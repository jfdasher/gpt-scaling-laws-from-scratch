"""Modal wiring. The only file that imports modal.

    uv run modal run modal_app.py::prep_data
    uv run modal run modal_app.py::baselines
    uv run modal run modal_app.py::bench
    uv run modal run --detach modal_app.py::launch --phase grid [--yes]
    uv run modal run modal_app.py::pull_results
    uv run modal run modal_app.py::publish --run-id <id> --repo-id <user/name> [--public]
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import modal

HF_DIR, DATA_DIR, CKPT_DIR, INDUCTOR_DIR = "/vol/hf", "/vol/data", "/vol/ckpt", "/vol/inductor"
DATASET, DATASET_PREFIX = "HuggingFaceFW/fineweb-edu", "sample/10BT/"
N_FILES = 4
TOKENIZER_TRAIN_BYTES = 1_000_000_000
MIN_TRAIN_TOKENS = 2_200_000_000
MIN_VAL_TOKENS = 10_000_000
WANDB_PROJECT = "gptlab-scaling"

app = modal.App("gptlab")
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Exact pins = the versions in uv.lock (torch without the local "+cpu" tag, so PyPI's default CUDA build installs).
    .pip_install(
        "torch==2.14.1", "numpy==2.5.4", "tokenizers==0.23.3", "pyarrow==26.0.0", "huggingface_hub==2.2.0",
        "wandb==0.31.0", "transformers==5.20.2", "safetensors==0.8.0", "scipy==1.18.1",
    )
    .env({"HF_HOME": HF_DIR, "TORCHINDUCTOR_CACHE_DIR": INDUCTOR_DIR})
    .add_local_python_source("gptlab", "sweep", "analysis")
)
hf_vol = modal.Volume.from_name("gptlab-hf", create_if_missing=True)
data_vol = modal.Volume.from_name("gptlab-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gptlab-ckpt", create_if_missing=True)
inductor_vol = modal.Volume.from_name("gptlab-inductor", create_if_missing=True)


# ---------- data prep (CPU) ----------

def _iter_docs(path: str):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(columns=["id", "text"], batch_size=8192):
        d = batch.to_pydict()
        yield from zip(d["id"], d["text"])


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, filename, repo_type="dataset")
    hf_vol.commit()
    return path


@app.function(image=image, volumes={HF_DIR: hf_vol, DATA_DIR: data_vol}, cpu=16, memory=32768, timeout=3 * 3600)
def train_tokenizer_remote(files: list[str]) -> str:
    from gptlab.tokenizer import is_val_doc, train_tokenizer

    def texts():
        seen = 0
        for f in files:
            for doc_id, text in _iter_docs(_download(f)):
                if is_val_doc(doc_id):
                    continue  # tokenizer never sees val text
                yield text
                seen += len(text.encode("utf-8"))
                if seen >= TOKENIZER_TRAIN_BYTES:
                    return

    tok = train_tokenizer(texts())
    out = Path(DATA_DIR) / "tokenizer.json"
    tok.save(str(out))
    data_vol.commit()
    return str(out)


@app.function(image=image, volumes={HF_DIR: hf_vol, DATA_DIR: data_vol}, cpu=8, memory=65536, timeout=3 * 3600)
def encode_file(filename: str) -> dict:
    from tokenizers import Tokenizer

    from gptlab.tokenizer import encode_file_docs, shard_seed, write_val_shard

    data_vol.reload()
    tok = Tokenizer.from_file(f"{DATA_DIR}/tokenizer.json")
    train, val_docs = encode_file_docs(tok, _iter_docs(_download(filename)), seed=shard_seed(filename))
    shard_dir = Path(DATA_DIR) / "shards"
    shard_dir.mkdir(exist_ok=True)
    stem = Path(filename).stem
    train.tofile(shard_dir / f"train_{stem}.bin")
    write_val_shard(val_docs, shard_dir / f"val_{stem}.bin", shard_dir / f"val_{stem}.json")
    data_vol.commit()
    return {"file": filename, "train_tokens": int(train.size), "val_docs": len(val_docs),
            "val_tokens": int(sum(len(a) for _, a, _ in val_docs))}


@app.function(image=image, volumes={DATA_DIR: data_vol}, memory=32768, timeout=3600)
def finalize_data(files: list[str]) -> dict:
    from gptlab.data import build_manifest, write_manifest
    from gptlab.tokenizer import VAL_PER_MILLE, concat_shards, merge_val_shards

    data_vol.reload()
    shard_dir = Path(DATA_DIR) / "shards"
    stems = [Path(f).stem for f in files]
    n_train = concat_shards([shard_dir / f"train_{s}.bin" for s in stems], Path(DATA_DIR) / "train.bin", seed=0,
                            min_tokens=MIN_TRAIN_TOKENS)
    n_docs, n_tokens, n_bytes = merge_val_shards(
        [(shard_dir / f"val_{s}.bin", shard_dir / f"val_{s}.json") for s in stems], Path(DATA_DIR) / "val.bin", MIN_VAL_TOKENS
    )
    manifest = build_manifest(DATA_DIR, n_docs, n_tokens, n_bytes, extra={
        "dataset": DATASET, "files": files, "train_tokens": n_train, "val_per_mille": VAL_PER_MILLE,
    })
    write_manifest(DATA_DIR, manifest)
    data_vol.commit()
    return manifest


@app.local_entrypoint()
def prep_data(n_files: int = N_FILES):
    from huggingface_hub import list_repo_files

    files = sorted(f for f in list_repo_files(DATASET, repo_type="dataset")
                   if f.startswith(DATASET_PREFIX) and f.endswith(".parquet"))
    assert files, f"no parquet files under {DATASET}/{DATASET_PREFIX}"
    random.Random(0).shuffle(files)
    files = files[:n_files]
    print("files:", files)
    print("tokenizer:", train_tokenizer_remote.remote(files))
    for stats in encode_file.map(files):
        print(stats)
    manifest = finalize_data.remote(files)
    Path("data").mkdir(exist_ok=True)
    Path("data/val_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


# ---------- baselines and benchmark (GPU) ----------

@app.function(image=image, gpu="H100", volumes={DATA_DIR: data_vol, HF_DIR: hf_vol}, memory=32768, timeout=3600)
def run_baselines() -> dict:
    from gptlab.baselines import compute_baselines

    return compute_baselines(DATA_DIR, device="cuda")


@app.local_entrypoint()
def baselines():
    out = run_baselines.remote()
    Path("baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


@app.function(image=image, gpu="H100", volumes={DATA_DIR: data_vol, INDUCTOR_DIR: inductor_vol}, timeout=3600)
def bench_remote(shapes: list[tuple[int, int]]) -> list[dict]:
    from gptlab.train import benchmark

    try:
        return [benchmark(L, d, DATA_DIR) for L, d in shapes]
    finally:
        _commit_inductor()


@app.local_entrypoint()
def bench():
    import sweep

    rows = bench_remote.remote(sweep.bench_shapes())
    Path("bench.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"L={r['n_layer']} d={r['d_model']} N={r['n_params']:,} {r['tokens_per_s']:,.0f} tok/s MFU {r['mfu']:.1%}")


# ---------- training (GPU) ----------

def _commit_inductor() -> None:
    try:
        inductor_vol.commit()
    except Exception as e:  # never mask the real outcome of the run
        print(f"WARNING: inductor cache commit failed: {e!r}")


def _train_impl(cfg_dict: dict, preempt_at_step: int | None, git_commit: str | None) -> dict:
    from gptlab.data import DataExhausted
    from gptlab.train import NonFiniteLoss, RunFailed, TrainConfig, WandbLogger, run_id, train

    ckpt_vol.reload()
    cfg = TrainConfig(**cfg_dict)
    run_dir = Path(CKPT_DIR) / "runs" / run_id(cfg)
    # Short-circuit before touching W&B: a finished/failed run must not open (or reuse) a W&B run.
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())
    if (run_dir / "FAILED").exists():
        return {"failed": True, "run_id": run_id(cfg), "name": cfg.name, "error": repr(RunFailed((run_dir / "FAILED").read_text()))}
    try:
        return train(cfg, run_dir, DATA_DIR, logger=WandbLogger(WANDB_PROJECT, cfg), commit=ckpt_vol.commit,
                     preempt_at_step=preempt_at_step, git_commit=git_commit)
    except (NonFiniteLoss, RunFailed, DataExhausted) as e:
        # deterministic failure: return (don't raise) so Modal doesn't burn retries on it
        return {"failed": True, "run_id": run_id(cfg), "name": cfg.name, "error": repr(e)}
    finally:
        _commit_inductor()


TRAIN_KW = dict(
    image=image,
    gpu="H100",
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol, INDUCTOR_DIR: inductor_vol},
    secrets=[modal.Secret.from_name("wandb")],
    retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0),
    max_containers=10,
    single_use_containers=True,  # a warm container must never carry W&B state into the next config
)


@app.function(timeout=2 * 3600, **TRAIN_KW)
def train_one(cfg_dict: dict, preempt_at_step: int | None = None, git_commit: str | None = None) -> dict:
    return _train_impl(cfg_dict, preempt_at_step, git_commit)


@app.function(timeout=6 * 3600, **TRAIN_KW)
def train_long(cfg_dict: dict, preempt_at_step: int | None = None, git_commit: str | None = None) -> dict:
    return _train_impl(cfg_dict, preempt_at_step, git_commit)


def _git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


@app.function(image=image, timeout=24 * 3600)
def run_phase(args: list[tuple[dict, int | None, str]], long: bool) -> list[str]:
    fn = train_long if long else train_one
    summaries = []
    for res in fn.starmap(args, return_exceptions=True, order_outputs=False):
        if isinstance(res, BaseException):
            line = f"FAILED: {res!r}"
        elif res.get("failed"):
            line = f"FAILED: {res['name']} {res['run_id']} {res['error']}"
        else:
            line = f"done {res['name']} {res['run_id']} val_loss={res['val_loss']:.4f} ${res['cost_usd']:.2f}"
        print(line)
        summaries.append(line)
    return summaries


@app.function(image=image, volumes={CKPT_DIR: ckpt_vol}, timeout=300)
def run_states() -> dict:
    """{run_id: "done" | "failed" | "in_progress"} from the checkpoint volume."""
    ckpt_vol.reload()
    out = {}
    for d in (Path(CKPT_DIR) / "runs").glob("*"):
        if (d / "result.json").exists():
            out[d.name] = "done"
        elif (d / "FAILED").exists():
            out[d.name] = "failed"
        elif (d / "ckpt.pt").exists() or (d / "losses.jsonl").exists():
            out[d.name] = "in_progress"
    return out


@app.local_entrypoint()
def launch(phase: str, yes: bool = False, from_run: str = "", lr_rule: str = "lr_rule.json",
           prediction: str = "predictions/3e17.json", include_in_progress: bool = False):
    import sweep
    from gptlab.train import run_id

    cfgs = sweep.unique(sweep.configs_for_phase(phase, lr_rule, from_run, prediction))
    skip, warnings = sweep.skip_set(sweep.done_run_ids("results"), run_states.remote(), [run_id(c) for c in cfgs],
                                    include_in_progress)
    for w in warnings:
        print(w)
    sweep.print_plan(cfgs, skip)
    todo = [c for c in cfgs if run_id(c) not in skip]
    if not yes:
        print("dry run: nothing launched. Re-run with --yes to launch.")
        return
    commit = _git_commit()
    args = [(asdict(c), sweep.SMOKE_PREEMPT_STEP if c.tag == "preempt" else None, commit) for c in todo]
    # Orchestration runs server-side: `modal run --detach` keeps only the last triggered call alive, and
    # starmap input retries are client-side, so a client-side starmap would die with the disconnect.
    for line in run_phase.remote(args, phase == "test"):
        print(line)


# ---------- results and publishing ----------

@app.function(image=image, volumes={CKPT_DIR: ckpt_vol}, timeout=900)
def collect_results() -> dict:
    ckpt_vol.reload()
    out = {"results": [], "failed": [], "curves": {}, "diag": {}}
    for d in sorted((Path(CKPT_DIR) / "runs").glob("*")):
        if (d / "result.json").exists():
            out["results"].append(json.loads((d / "result.json").read_text()))
        elif (d / "FAILED").exists():
            cfg = json.loads((d / "config.json").read_text())
            forensic = json.loads((d / "forensic.json").read_text()) if (d / "forensic.json").exists() else {}
            out["failed"].append({"run_id": d.name, "name": cfg["name"], "phase": cfg["phase"], "config": cfg,
                                  "forensic": forensic})
        if (d / "losses.jsonl").exists():
            out["curves"][d.name] = (d / "losses.jsonl").read_text()
        if (d / "diag.jsonl").exists():
            out["diag"][d.name] = (d / "diag.jsonl").read_text()
    return out


@app.local_entrypoint()
def pull_results(out: str = "results"):
    root = Path(out)
    (root / "failed").mkdir(parents=True, exist_ok=True)
    (root / "curves").mkdir(exist_ok=True)
    got = collect_results.remote()
    for r in got["results"]:
        (root / f"{r['run_id']}.json").write_text(json.dumps(r, indent=2))
    for f in got["failed"]:
        (root / "failed" / f"{f['run_id']}.json").write_text(json.dumps(f, indent=2))
    for rid, text in got["curves"].items():
        (root / "curves" / f"{rid}.jsonl").write_text(text)
    for rid, text in got["diag"].items():
        (root / "curves" / f"{rid}.diag.jsonl").write_text(text)
    print(f"{len(got['results'])} results, {len(got['failed'])} failed, {len(got['curves'])} curves, {len(got['diag'])} diag -> {root}/")


@app.function(image=image, volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def push_model(run_id: str, repo_id: str, private: bool = True) -> str:
    from huggingface_hub import HfApi

    from analysis.export_hf import export_run

    out = export_run(Path(CKPT_DIR) / "runs" / run_id, Path(DATA_DIR) / "tokenizer.json", Path("/tmp/export") / run_id,
                     repo_id=repo_id)
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def publish(run_id: str, repo_id: str, public: bool = False):
    print(push_model.remote(run_id, repo_id, private=not public))

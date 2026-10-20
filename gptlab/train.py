"""The training loop. train() is what a Modal container (or a CPU test) calls for one run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch

from gptlab import data as gdata
from gptlab.flops import BATCH_TOKENS, H100_DOLLARS_PER_HOUR, H100_PEAK_FLOPS, SEQ_LEN, VOCAB_SIZE
from gptlab.model import GPT, GPTConfig, loss_fn
from gptlab.schedule import lr_at

SAMPLE_PROMPTS = (
    "The water cycle is",
    "In 1492,",
    "To solve a quadratic equation,",
    "The most important idea in biology is",
)
HASH_EXCLUDE = frozenset({"name", "phase", "budget_flops", "tokens_per_param"})  # bookkeeping, not behaviour


@dataclass(frozen=True)
class TrainConfig:
    name: str
    n_layer: int
    d_model: int
    total_steps: int
    peak_lr: float
    phase: str = ""
    tag: str = ""  # distinguishes otherwise identical runs (e.g. the preempted smoke run)
    batch_tokens: int = BATCH_TOKENS
    micro_batch: int = 64  # sequences per forward/backward
    seq_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float | None = 1.0
    qk_norm: bool = False
    z_loss: float = 0.0
    seed: int = 0
    data_offset: int = 0
    log_every: int = 10
    eval_every: int | None = None  # default: total_steps // 10
    eval_tokens: int = 1_048_576
    ckpt_every_s: float = 600.0
    compile: bool = True
    budget_flops: float = 0.0
    tokens_per_param: float = 0.0


def run_id(cfg: TrainConfig) -> str:
    payload = {k: v for k, v in asdict(cfg).items() if k not in HASH_EXCLUDE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


class SimulatedPreemption(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


class RunFailed(RuntimeError):
    pass


class Logger(Protocol):
    url: str | None

    def log(self, metrics: dict, step: int) -> None: ...
    def log_samples(self, step: int, samples: list[tuple[str, str]]) -> None: ...
    def finish(self) -> None: ...


class NullLogger:
    url = None

    def log(self, metrics, step):
        pass

    def log_samples(self, step, samples):
        pass

    def finish(self):
        pass


class WandbLogger:
    """W&B run keyed by run_id, so a resumed container continues the same run."""

    def __init__(self, project: str, cfg: TrainConfig):
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project, id=run_id(cfg), name=cfg.name, group=cfg.phase, config=asdict(cfg), resume="allow"
        )
        self.url = self.run.url

    def log(self, metrics, step):
        self.run.log(metrics, step=step)

    def log_samples(self, step, samples):
        table = self._wandb.Table(columns=["step", "prompt", "completion"], data=[[step, p, c] for p, c in samples])
        self.run.log({"samples": table}, step=step)

    def finish(self):
        self.run.finish()


def _autocast(device: str):
    kind = torch.device(device).type
    return torch.autocast(device_type=kind, dtype=torch.bfloat16, enabled=kind == "cuda")


def make_optimizer(model: GPT, cfg: TrainConfig, fused: bool) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, fused=fused)


def save_checkpoint(path: Path, model, optimizer, step: int, gpu_seconds: float) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "gpu_seconds": gpu_seconds,
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic: a crash leaves either the old or the new checkpoint, never half of one


def load_checkpoint(path: Path, model, optimizer) -> dict:
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["torch_rng"])
    return ck


def forward_backward(fwd, stream: gdata.TokenStream, step: int, accum: int, cfg: TrainConfig, device: str) -> float:
    """One optimizer step's worth of micro-batches; grads accumulate in .grad. Returns mean cross-entropy."""
    loss_sum = torch.zeros((), device=device)
    for micro in range(accum):
        x, y = stream.get(step * accum + micro)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device):
            logits = fwd(x)
        total, ce = loss_fn(logits, y, cfg.z_loss)
        (total / accum).backward()
        loss_sum += ce.detach()
    return (loss_sum / accum).item()


@torch.no_grad()
def evaluate(model: GPT, val_path: Path, cfg: TrainConfig, device: str, max_tokens: int | None) -> float:
    total, count = 0.0, 0
    for x, y in gdata.val_batches(val_path, cfg.seq_len, cfg.micro_batch, max_tokens):
        x, y = x.to(device), y.to(device)
        with _autocast(device):
            logits = model(x)
        _, ce = loss_fn(logits, y)
        total += ce.item() * y.numel()
        count += y.numel()
    return total / count


def eval_metrics(model: GPT, val_path: Path, cfg: TrainConfig, device: str, manifest: dict) -> dict:
    loss = evaluate(model, val_path, cfg, device, cfg.eval_tokens)
    out = {"val/loss": loss, "val/bpb": gdata.bits_per_byte(loss, manifest["n_tokens"], manifest["n_bytes"])}
    x, _ = next(gdata.val_batches(val_path, cfg.seq_len, min(cfg.micro_batch, 8), cfg.eval_tokens))
    stats: list[float] = []
    with torch.no_grad(), _autocast(device):
        model(x.to(device), stats=stats)
    out.update({f"attn_max_logit/layer_{i}": v for i, v in enumerate(stats)})
    return out


def layer_grad_norms(model: GPT) -> dict:
    out = {}
    for i, layer in enumerate(model.layers):
        sq = sum(float(p.grad.float().pow(2).sum()) for p in layer.parameters() if p.grad is not None)
        out[f"grad_norm/layer_{i}"] = sq**0.5
    if model.embed_tokens.weight.grad is not None:
        out["grad_norm/embed"] = model.embed_tokens.weight.grad.float().norm().item()
    return out


def generate_samples(model: GPT, tokenizer_path: Path, prompts, device: str, max_new_tokens: int = 48):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    gen = torch.Generator(device=device).manual_seed(1234)
    out = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        with _autocast(device):
            seq = model.generate(ids, max_new_tokens, generator=gen)
        out.append((prompt, tok.decode(seq[0, ids.size(1) :].tolist())))
    return out


def _write_forensics(run_dir, model, optimizer, step, stream, accum, loss, lr, gpu_seconds) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, run_dir / "forensic.pt")
    span = [stream.span(step * accum)[0], stream.span(step * accum + accum - 1)[1]]
    info = {"step": step, "loss": repr(loss), "lr": lr, "token_span": span, "gpu_seconds": gpu_seconds}
    (run_dir / "forensic.json").write_text(json.dumps(info, indent=2))
    (run_dir / "FAILED").write_text(f"non-finite loss {loss!r} at step {step}")


def train(
    cfg: TrainConfig,
    run_dir,
    data_dir,
    logger: Logger | None = None,
    device: str | None = None,
    commit: Callable[[], None] = lambda: None,
    preempt_at_step: int | None = None,
    dollars_per_hour: float = H100_DOLLARS_PER_HOUR,
    sample_prompts=SAMPLE_PROMPTS,
    git_commit: str | None = None,
) -> dict:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (run_dir / "FAILED").exists():
        raise RunFailed((run_dir / "FAILED").read_text())
    manifest = gdata.verify_manifest(data_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger = logger or NullLogger()
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    model = GPT(
        GPTConfig(n_layer=cfg.n_layer, d_model=cfg.d_model, vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, qk_norm=cfg.qk_norm)
    ).to(device)
    optimizer = make_optimizer(model, cfg, fused=device.startswith("cuda"))
    ckpt_path = run_dir / "ckpt.pt"
    step, gpu_seconds = 0, 0.0
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, model, optimizer)
        step, gpu_seconds = ck["step"], ck["gpu_seconds"]
    fwd = torch.compile(model) if (cfg.compile and device.startswith("cuda")) else model

    accum = cfg.batch_tokens // (cfg.micro_batch * cfg.seq_len)
    assert accum * cfg.micro_batch * cfg.seq_len == cfg.batch_tokens, "batch_tokens must be a multiple of micro_batch*seq_len"
    stream = gdata.TokenStream(data_dir / gdata.TRAIN_BIN, cfg.seq_len, cfg.micro_batch, cfg.data_offset)
    val_path = data_dir / gdata.VAL_BIN
    eval_every = cfg.eval_every or max(1, cfg.total_steps // 10)
    n_params = model.num_params()

    seg_start = time.monotonic()
    last_ckpt = t_log = seg_start
    tokens_since_log = 0

    def elapsed() -> float:
        return gpu_seconds + (time.monotonic() - seg_start)

    with (run_dir / "losses.jsonl").open("a") as losses_file:

        def checkpoint(at_step: int) -> None:
            losses_file.flush()
            save_checkpoint(ckpt_path, model, optimizer, at_step, elapsed())
            commit()

        while step < cfg.total_steps:
            if preempt_at_step is not None and step == preempt_at_step and not (run_dir / "preempted").exists():
                checkpoint(step)
                (run_dir / "preempted").touch()
                commit()
                raise SimulatedPreemption(f"simulated preemption at step {step}")
            lr = lr_at(step, cfg.total_steps, cfg.peak_lr, cfg.warmup_frac, cfg.final_lr_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = forward_backward(fwd, stream, step, accum, cfg, device)
            if not math.isfinite(loss):
                losses_file.flush()
                _write_forensics(run_dir, model, optimizer, step, stream, accum, loss, lr, elapsed())
                commit()
                raise NonFiniteLoss(f"loss {loss!r} at step {step}; see {run_dir / 'forensic.json'}")
            clip = cfg.grad_clip if cfg.grad_clip is not None else math.inf
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()  # pre-clip norm
            will_eval = (step + 1) % eval_every == 0 or step + 1 == cfg.total_steps
            layer_norms = layer_grad_norms(model) if will_eval else {}
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses_file.write(json.dumps({"step": step, "loss": loss, "lr": lr, "grad_norm": grad_norm}) + "\n")
            step += 1
            tokens_since_log += cfg.batch_tokens
            if step % cfg.log_every == 0:
                now = time.monotonic()
                tps = tokens_since_log / max(now - t_log, 1e-9)
                t_log, tokens_since_log = now, 0
                logger.log(
                    {
                        "train/loss": loss, "train/lr": lr, "train/grad_norm": grad_norm,
                        "perf/tokens_per_s": tps, "perf/mfu": 6 * n_params * tps / H100_PEAK_FLOPS,
                        "gpu_seconds": elapsed(),
                    },
                    step=step,
                )
            if will_eval:
                metrics = eval_metrics(model, val_path, cfg, device, manifest)
                metrics.update(layer_norms)
                logger.log(metrics, step=step)
                if sample_prompts:
                    logger.log_samples(step, generate_samples(model, data_dir / gdata.TOKENIZER_JSON, sample_prompts, device))
            if step < cfg.total_steps and time.monotonic() - last_ckpt >= cfg.ckpt_every_s:
                checkpoint(step)
                last_ckpt = time.monotonic()

    final_loss = evaluate(model, val_path, cfg, device, max_tokens=None)
    total_seconds = elapsed()
    torch.save(model.state_dict(), run_dir / "final.pt")
    tokens = cfg.total_steps * cfg.batch_tokens
    result = {
        "run_id": run_id(cfg),
        "name": cfg.name,
        "phase": cfg.phase,
        "config": asdict(cfg),
        "n_layer": cfg.n_layer,
        "d_model": cfg.d_model,
        "n_params": n_params,
        "n_params_nonembed": model.num_params_nonembed(),
        "tokens": tokens,
        "flops": 6.0 * n_params * tokens,
        "budget_flops": cfg.budget_flops,
        "nominal_tokens_per_param": cfg.tokens_per_param,
        "tokens_per_param": tokens / n_params,
        "val_loss": final_loss,
        "val_bpb": gdata.bits_per_byte(final_loss, manifest["n_tokens"], manifest["n_bytes"]),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600,
        "cost_usd": total_seconds / 3600 * dollars_per_hour,
        "torch_version": torch.__version__,
        "wandb_url": logger.url,
        "git_commit": git_commit,
    }
    logger.log({"final/val_loss": result["val_loss"], "final/val_bpb": result["val_bpb"]}, step=cfg.total_steps)
    tmp = result_path.with_name("result.json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, result_path)
    ckpt_path.unlink(missing_ok=True)
    commit()
    logger.finish()
    return result


def benchmark(n_layer: int, d_model: int, data_dir, device: str = "cuda", steps: int = 30, warmup: int = 10) -> dict:
    """Steady-state throughput for one shape with the real step (compile, bf16, fused AdamW)."""
    cfg = TrainConfig(name="bench", n_layer=n_layer, d_model=d_model, total_steps=steps, peak_lr=1e-3)
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=n_layer, d_model=d_model)).to(device)
    optimizer = make_optimizer(model, cfg, fused=device.startswith("cuda"))
    fwd = torch.compile(model) if device.startswith("cuda") else model
    accum = cfg.batch_tokens // (cfg.micro_batch * cfg.seq_len)
    stream = gdata.TokenStream(Path(data_dir) / gdata.TRAIN_BIN, cfg.seq_len, cfg.micro_batch)
    t0 = 0.0
    for step in range(steps):
        if step == warmup:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.monotonic()
        forward_backward(fwd, stream, step, accum, cfg, device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    tps = (steps - warmup) * cfg.batch_tokens / (time.monotonic() - t0)
    n = model.num_params()
    return {"n_layer": n_layer, "d_model": d_model, "n_params": n, "tokens_per_s": tps,
            "flops_per_s": 6 * n * tps, "mfu": 6 * n * tps / H100_PEAK_FLOPS}

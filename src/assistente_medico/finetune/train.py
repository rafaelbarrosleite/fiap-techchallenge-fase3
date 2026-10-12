"""Treino LoRA com MLX-LM e registro de métricas (``uv run finetune``).

Chama o treino do MLX-LM no mesmo processo, com um ``TrainingCallback`` que:

- grava loss de treino/validação, tokens/s e memória em ``metrics.jsonl``;
- gerar ``manifest.json`` com configuração, hashes dos dados, duração e versões;
- desenhar a curva de loss (``loss_curve.png``).

``--smoke`` roda poucas iterações para validar memória e velocidade antes do treino oficial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from assistente_medico.audit.logger import AuditLogger

class MetricsCallback:
    """Recebe os relatórios do treino do MLX-LM e grava cada um como uma linha JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []

    def _write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def on_train_loss_report(self, info: dict[str, Any]) -> None:
        self._write({
            "type": "train", "iter": info["iteration"], "loss": float(info["train_loss"]),
            "lr": float(info["learning_rate"]), "it_per_sec": float(info["iterations_per_second"]),
            "tokens_per_sec": float(info["tokens_per_second"]), "trained_tokens": int(info["trained_tokens"]),
            "peak_mem_gb": float(info["peak_memory"]),
        })

    def on_val_loss_report(self, info: dict[str, Any]) -> None:
        self._write({"type": "val", "iter": info["iteration"], "loss": float(info["val_loss"]),
                     "seconds": float(info["val_time"])})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_loss(metrics: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [m for m in metrics if m["type"] == "train"]
    val = [m for m in metrics if m["type"] == "val"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([m["iter"] for m in train], [m["loss"] for m in train], label="treino", linewidth=1.2)
    ax.plot([m["iter"] for m in val], [m["loss"] for m in val], "o-", label="validação", linewidth=1.8)
    ax.set_xlabel("iteração")
    ax.set_ylabel("loss (entropia cruzada)")
    ax.set_title("Fine-tuning LoRA — Llama 3.2 3B Instruct")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(config_path: Path, smoke: bool = False, run_name: str | None = None) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_name = run_name or ("smoke" if smoke else datetime.now().strftime("run_%Y%m%d_%H%M"))
    run_dir = Path("artifacts/finetune") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg["adapter_path"] = str(run_dir / "adapters")
    if smoke:
        cfg.update(iters=30, steps_per_report=5, steps_per_eval=15, val_batches=5, save_every=30)
        if "lr_schedule" in cfg:
            cfg["lr_schedule"]["warmup"] = 5
            cfg["lr_schedule"]["arguments"][1] = 30
    effective_cfg = run_dir / "config.yaml"
    effective_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    data_dir = Path(cfg["data"])
    data_hashes = {f.name: _sha256(f) for f in sorted(data_dir.glob("*.jsonl"))}

    audit = AuditLogger()
    audit.log("finetune_start", run=run_name, model=cfg["model"], iters=cfg["iters"], data_sha256=data_hashes)

    from types import SimpleNamespace

    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    args = SimpleNamespace(**{**CONFIG_DEFAULTS, **cfg})
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    callback = MetricsCallback(metrics_path)
    started = time.time()
    status = "completed"
    try:
        model, tokenizer = load(args.model)
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        train_model(args, model, train_set, valid_set, callback)
    except BaseException as exc:  # inclui KeyboardInterrupt: registra a falha no manifesto
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        duration = time.time() - started
        _finalize(run_dir, run_name, cfg, data_hashes, callback.records, status, duration, audit)
    return run_dir


def _finalize(run_dir: Path, run_name: str, cfg: dict[str, Any], data_hashes: dict[str, str],
              metrics: list[dict[str, Any]], status: str, duration: float, audit: AuditLogger) -> None:
    if metrics:
        plot_loss(metrics, run_dir / "loss_curve.png")
    train = [m for m in metrics if m["type"] == "train"]
    val = [m for m in metrics if m["type"] == "val"]
    manifest = {
        "run": run_name,
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": round(duration / 60, 2),
        "config": cfg,
        "data_sha256": data_hashes,
        "summary": {
            "first_val_loss": val[0]["loss"] if val else None,
            "best_val_loss": min(m["loss"] for m in val) if val else None,
            "last_val_loss": val[-1]["loss"] if val else None,
            "last_train_loss": train[-1]["loss"] if train else None,
            "peak_mem_gb": max(m["peak_mem_gb"] for m in train) if train else None,
            "mean_tokens_per_sec": round(sum(m["tokens_per_sec"] for m in train) / len(train), 1) if train else None,
            "trained_tokens": train[-1]["trained_tokens"] if train else None,
        },
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    audit.log("finetune_end", run=run_name, status=status, summary=manifest["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA do Llama com MLX-LM (Missão 2).")
    parser.add_argument("--config", type=Path, default=Path("configs/finetune.yaml"))
    parser.add_argument("--smoke", action="store_true", help="30 iterações para validar memória e velocidade")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    run_dir = run(args.config, smoke=args.smoke, run_name=args.run_name)
    print(f"\nArtefatos: {run_dir}")

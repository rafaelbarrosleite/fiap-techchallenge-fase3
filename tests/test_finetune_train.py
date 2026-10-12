import json

import yaml

from assistente_medico.finetune.train import MetricsCallback, plot_loss


def test_callback_grava_metricas_de_treino_e_validacao(tmp_path):
    cb = MetricsCallback(tmp_path / "metrics.jsonl")
    cb.on_val_loss_report({"iteration": 0, "val_loss": 3.0, "val_time": 20.0})
    cb.on_train_loss_report({
        "iteration": 10, "train_loss": 1.7, "learning_rate": 1e-4, "iterations_per_second": 0.08,
        "tokens_per_second": 20.0, "trained_tokens": 2000, "peak_memory": 5.1,
    })
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["type"] for r in rows] == ["val", "train"]
    assert rows[1]["loss"] == 1.7 and rows[1]["peak_mem_gb"] == 5.1

    plot_loss(cb.records, tmp_path / "loss.png")
    assert (tmp_path / "loss.png").stat().st_size > 0


def test_config_oficial_usa_qlora_com_mascara_de_prompt():
    cfg = yaml.safe_load(open("configs/finetune.yaml", encoding="utf-8"))
    assert cfg["model"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
    assert cfg["fine_tune_type"] == "lora" and cfg["mask_prompt"] is True
    assert cfg["grad_checkpoint"] is True
    assert cfg["lr_schedule"]["arguments"][1] == cfg["iters"]

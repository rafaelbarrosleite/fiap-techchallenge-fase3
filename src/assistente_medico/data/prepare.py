"""Pipeline de dados da Missão 1 (``uv run prepare-data``).

PubMedQA + corpus sintético do hospital → anonimização → curadoria → splits →
JSONL para o MLX e relatório de dados com hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from assistente_medico.audit.logger import AuditLogger
from assistente_medico.data import pubmedqa
from assistente_medico.data.curate import curate
from assistente_medico.data.hospital_corpus import build_hospital_examples, load_protocols

PROCESSED_DIR = Path("data/processed")
REPORT_PATH = Path("artifacts/data/data_report.json")
SPLITS = ("train", "valid", "test")


def stratified_split(df: pd.DataFrame, sizes: dict[str, int], by: str, seed: int) -> dict[str, pd.DataFrame]:
    """Divide ``df`` nos tamanhos pedidos mantendo a proporção da coluna ``by``."""
    total = sum(sizes.values())
    if total != len(df):
        raise ValueError(f"Tamanhos somam {total}, mas há {len(df)} linhas")
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in sizes}
    names = list(sizes)
    for _, group in df.sample(frac=1, random_state=seed).groupby(by, sort=True):
        start = 0
        for name in names[:-1]:
            n = round(len(group) * sizes[name] / total)
            parts[name].append(group.iloc[start : start + n])
            start += n
        parts[names[-1]].append(group.iloc[start:])
    out = {name: pd.concat(frames) for name, frames in parts.items()}
    # Ajusta arredondamentos movendo linhas do último split para os demais.
    for name in names[:-1]:
        diff = sizes[name] - len(out[name])
        if diff > 0:
            out[name] = pd.concat([out[name], out[names[-1]].iloc[:diff]])
            out[names[-1]] = out[names[-1]].iloc[diff:]
        elif diff < 0:
            out[names[-1]] = pd.concat([out[names[-1]], out[name].iloc[diff:]])
            out[name] = out[name].iloc[:diff]
    return out


def split_pubmedqa(cfg: dict[str, Any], seed: int, raw_dir: Path) -> tuple[dict[str, list], dict[str, Any]]:
    paths = pubmedqa.download(raw_dir)
    labeled = pubmedqa.load(paths["pqa_labeled"])
    artificial = pubmedqa.load(paths["pqa_artificial"])

    # Vazamento: nenhum artigo ou pergunta do conjunto rotulado pode aparecer no artificial.
    before = len(artificial)
    artificial = artificial[~artificial["pubid"].isin(labeled["pubid"])]
    artificial = artificial[~artificial["question"].str.lower().isin(labeled["question"].str.lower())]
    leakage_removed = before - len(artificial)

    n_test, n_valid = cfg["labeled_test"], cfg["labeled_valid"]
    lab = stratified_split(
        labeled, {"test": n_test, "valid": n_valid, "train": len(labeled) - n_test - n_valid}, "final_decision", seed
    )

    # pqa_artificial é ~92% "yes": amostra balanceada yes/no para o modelo não aprender a sempre dizer "yes".
    n_art = cfg["artificial_train"] + cfg["artificial_valid"]
    balanced = pd.concat(
        artificial[artificial["final_decision"] == d].sample(n_art // 2, random_state=seed) for d in ("yes", "no")
    )
    art = stratified_split(
        balanced, {"valid": cfg["artificial_valid"], "train": cfg["artificial_train"]}, "final_decision", seed
    )

    splits = {
        "train": [pubmedqa.to_example(r, "pqa_labeled") for _, r in lab["train"].iterrows()]
        + [pubmedqa.to_example(r, "pqa_artificial") for _, r in art["train"].iterrows()],
        "valid": [pubmedqa.to_example(r, "pqa_labeled") for _, r in lab["valid"].iterrows()]
        + [pubmedqa.to_example(r, "pqa_artificial") for _, r in art["valid"].iterrows()],
        "test": [pubmedqa.to_example(r, "pqa_labeled") for _, r in lab["test"].iterrows()],
    }
    info = {
        "raw_sizes": {"pqa_labeled": len(labeled), "pqa_artificial": before},
        "raw_decisions": {
            "pqa_labeled": labeled["final_decision"].value_counts().to_dict(),
            "pqa_artificial": artificial["final_decision"].value_counts().to_dict(),
        },
        "leakage_removed_from_artificial": leakage_removed,
    }
    return splits, info


def split_hospital(examples: list[dict[str, Any]], cfg: dict[str, Any], seed: int) -> dict[str, list]:
    """Split por grupo e estratificado por tarefa: variações da mesma pergunta ficam juntas."""
    rng = random.Random(seed)
    groups_by_task: dict[str, list[str]] = {}
    for ex in examples:
        groups = groups_by_task.setdefault(ex["task"], [])
        if ex["group"] not in groups:
            groups.append(ex["group"])
    assignment: dict[str, str] = {}
    for task in sorted(groups_by_task):
        groups = sorted(groups_by_task[task])
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * cfg["test_fraction"])) if len(groups) >= 3 else 0
        n_valid = max(1, round(len(groups) * cfg["valid_fraction"])) if len(groups) >= 3 else 0
        for i, group in enumerate(groups):
            assignment[group] = "test" if i < n_test else "valid" if i < n_test + n_valid else "train"
    splits: dict[str, list] = {name: [] for name in SPLITS}
    for ex in examples:
        splits[assignment[ex["group"]]].append(ex)
    return splits


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "por_fonte": dict(Counter(r["source"] for r in rows)),
        "por_tarefa": dict(Counter(r["task"] for r in rows)),
        "pubmedqa_decisoes": dict(Counter(r["decision"] for r in rows if r["source"] == "pubmedqa")),
    }


def prepare(config_path: Path, raw_dir: Path = Path("data/raw/pubmedqa")) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = cfg["seed"]
    audit = AuditLogger()
    audit.log("prepare_data_start", config=str(config_path), seed=seed)

    pub_splits, pub_info = split_pubmedqa(cfg["pubmedqa"], seed, raw_dir)
    hospital_examples, pii_counts = build_hospital_examples(seed=seed)
    hosp_splits = split_hospital(hospital_examples, cfg["hospital"], seed)

    removed_total: Counter = Counter()
    final: dict[str, list] = {}
    for name in SPLITS:
        kept, removed = curate(pub_splits[name] + hosp_splits[name], cfg["curation"]["max_chars"])
        removed_total.update({f"{name}:{k}": v for k, v in removed.items()})
        final[name] = kept

    # Vazamento entre splits do hospital: nenhuma pergunta de teste/validação pode estar no treino.
    train_users = {ex["messages"][1]["content"].lower() for ex in final["train"]}
    for name in ("valid", "test"):
        overlap = [ex for ex in final[name] if ex["messages"][1]["content"].lower() in train_users]
        if overlap:
            raise RuntimeError(f"{len(overlap)} perguntas de {name} também estão no treino")

    hospital_train = [ex for ex in final["train"] if ex["source"] == "hospital"]
    final["train"] += hospital_train * (cfg["hospital"]["train_upsample"] - 1)
    random.Random(seed).shuffle(final["train"])

    hashes = {}
    for name in SPLITS:
        hashes[f"{name}.jsonl"] = _write_jsonl(PROCESSED_DIR / f"{name}.jsonl", final[name])
        mlx_rows = [{"messages": ex["messages"]} for ex in final[name]]
        hashes[f"mlx/{name}.jsonl"] = _write_jsonl(PROCESSED_DIR / "mlx" / f"{name}.jsonl", mlx_rows)

    report = {
        "config": cfg,
        "pubmedqa": pub_info,
        "hospital": {
            "protocolos": [{"id": p.id, "titulo": p.titulo, "versao": p.versao} for p in load_protocols()],
            "exemplos_gerados": len(hospital_examples),
            "pii_anonimizada_nas_notas": pii_counts,
        },
        "removidos_na_curadoria": dict(sorted(removed_total.items())),
        "splits": {name: _summary(final[name]) for name in SPLITS},
        "sha256": hashes,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    audit.log("prepare_data_end", splits={n: len(final[n]) for n in SPLITS}, report=str(REPORT_PATH))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara os dados de fine-tuning (Missão 1).")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    report = prepare(args.config)
    for name, summary in report["splits"].items():
        print(f"{name:5s} {summary['total']:5d}  {summary['por_fonte']}  decisões={summary['pubmedqa_decisoes']}")
    print(f"Removidos: {report['removidos_na_curadoria']}")
    print(f"PII anonimizada nas notas: {report['hospital']['pii_anonimizada_nas_notas']}")
    print(f"Relatório: {REPORT_PATH}")

"""Download e conversão do PubMedQA (qiaojin/PubMedQA, licença MIT) para o formato de chat."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from assistente_medico.llm.prompts import PUBMEDQA_SYSTEM_PROMPT

REPO_ID = "qiaojin/PubMedQA"
SUBSETS = ("pqa_labeled", "pqa_artificial")
DECISIONS = ("yes", "no", "maybe")


def download(raw_dir: Path = Path("data/raw/pubmedqa")) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths = {}
    for subset in SUBSETS:
        paths[subset] = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"{subset}/train-00000-of-00001.parquet",
                local_dir=raw_dir,
            )
        )
    return paths


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def format_context(context: dict[str, Any]) -> str:
    texts = list(context["contexts"])
    labels = context.get("labels")
    labels = [] if labels is None else list(labels)
    if len(labels) == len(texts):
        return "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))
    return "\n".join(texts)


def user_message(question: str, context: dict[str, Any]) -> str:
    return f"Context:\n{format_context(context)}\n\nQuestion: {question}"


def to_example(row: pd.Series, subset: str) -> dict[str, Any]:
    decision = str(row["final_decision"]).strip().lower()
    return {
        "source": "pubmedqa",
        "task": subset,
        "group": f"pubmed-{row['pubid']}",
        "pubid": int(row["pubid"]),
        "decision": decision,
        "messages": [
            {"role": "system", "content": PUBMEDQA_SYSTEM_PROMPT},
            {"role": "user", "content": user_message(row["question"], row["context"])},
            {"role": "assistant", "content": f"Decision: {decision}\nExplanation: {str(row['long_answer']).strip()}"},
        ],
    }

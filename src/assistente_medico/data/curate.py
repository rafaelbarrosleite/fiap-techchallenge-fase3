"""Normalização, filtros de qualidade/segurança e deduplicação dos exemplos de chat."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from typing import Any

from assistente_medico.data.anonymize import find_pii
from assistente_medico.data.pubmedqa import DECISIONS

_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t]+")

# Tarefas do hospital cuja resposta precisa citar a fonte (explainability).
TASKS_REQUIRING_SOURCE = {"faq", "faq_rag", "nota_clinica", "seguranca"}
_PRESCRIPTION_HINT = re.compile(r"(?i)prescri|receita")
_VALIDATION_HINT = re.compile(r"(?i)PENDENTE_VALIDACAO|valida[çc][ãa]o")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text)).replace("\r\n", "\n").replace("\r", "\n")
    text = _TAG.sub(" ", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _assistant(example: dict[str, Any]) -> str:
    return example["messages"][-1]["content"]


def rejection_reason(example: dict[str, Any], max_chars: int) -> str | None:
    messages = example["messages"]
    if any(not m["content"].strip() for m in messages):
        return "mensagem_vazia"
    if sum(len(m["content"]) for m in messages) > max_chars:
        return "longo_demais"
    answer = _assistant(example)
    if example["source"] == "pubmedqa":
        if example.get("decision") not in DECISIONS:
            return "decisao_invalida"
        return None
    if example["task"] in TASKS_REQUIRING_SOURCE and "Fonte:" not in answer:
        return "sem_fonte"
    if _PRESCRIPTION_HINT.search(answer) and not _VALIDATION_HINT.search(answer):
        return "prescricao_sem_validacao"
    if any(find_pii(m["content"]) for m in messages if m["role"] != "system"):
        return "pii_residual"
    return None


def dedup_key(example: dict[str, Any]) -> str:
    joined = "\x1f".join(m["content"].lower() for m in example["messages"] if m["role"] != "system")
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def curate(examples: list[dict[str, Any]], max_chars: int) -> tuple[list[dict[str, Any]], Counter]:
    kept, removed, seen = [], Counter(), set()
    for example in examples:
        example = {**example, "messages": [{**m, "content": normalize_text(m["content"])} for m in example["messages"]]}
        reason = rejection_reason(example, max_chars)
        if reason is None:
            key = dedup_key(example)
            if key in seen:
                reason = "duplicado"
            else:
                seen.add(key)
        if reason:
            removed[f"{example['source']}:{reason}"] += 1
        else:
            kept.append(example)
    return kept, removed

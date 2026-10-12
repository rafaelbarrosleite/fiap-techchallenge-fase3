"""Avaliação do modelo base × ajustado (``uv run evaluate-model``).

Para cada modelo (``--label``):

- PubMedQA ``pqa_labeled`` (500 itens de teste): acurácia, F1 macro e matriz de confusão;
- teste do hospital (44 itens): ROUGE-L, protocolo e seção citados corretos, fonte
  inventada, recusa adequada/indevida, exigência de validação e idioma.

``--rag`` avalia só o hospital como o assistente funciona de verdade: perguntas de FAQ,
segurança e notas clínicas recebem os trechos encontrados pelo retriever (sem gabarito).

As predições são gravadas uma a uma em JSONL; rodar de novo retoma de onde parou.
``--compare`` gera a tabela comparativa e as figuras a partir dos resumos.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from assistente_medico.audit.logger import AuditLogger
from assistente_medico.data.hospital_corpus import load_protocols
from assistente_medico.evaluation.metrics import (
    aggregate,
    classification_report,
    parse_decision,
    score_hospital_answer,
)

BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
TEST_PATH = Path("data/processed/test.jsonl")
EVAL_DIR = Path("artifacts/evaluation")
MAX_TOKENS = {"pubmedqa": 24, "hospital": 320}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _example_id(example: dict[str, Any], index: int) -> str:
    return f"{example['source']}-{example.get('pubid', index)}"


RAG_TASKS = {"faq", "seguranca", "nota_clinica"}


def with_retrieved_context(example: dict[str, Any]) -> dict[str, Any]:
    """Substitui a pergunta pela mesma pergunta acompanhada dos trechos que o retriever encontrar."""
    from assistente_medico.llm.assistant_chain import build_user_message
    from assistente_medico.rag.retriever import get_retriever

    if example["task"] not in RAG_TASKS:
        return example
    question = example["messages"][1]["content"]
    docs = get_retriever().invoke(question)
    messages = [example["messages"][0], {"role": "user", "content": build_user_message(question, docs, None)},
                example["messages"][2]]
    return {**example, "messages": messages, "retrieved": [d.metadata["citation"] for d in docs]}


def protocol_catalog() -> dict[str, dict[str, Any]]:
    return {p.id: {"titulo": p.titulo, "secoes": {s.id for s in p.secoes}} for p in load_protocols()}


def generate_predictions(label: str, adapter_path: str | None, limit: int | None = None, rag: bool = False) -> Path:
    from mlx_lm import generate, load

    out_dir = EVAL_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {row["id"] for row in _load_jsonl(pred_path)}

    examples = _load_jsonl(TEST_PATH)
    ids = {id(e): _example_id(e, i) for i, e in enumerate(examples)}
    if rag:
        examples = [e for e in examples if e["source"] == "hospital"]
    if limit:
        examples = [e for e in examples if e["source"] == "pubmedqa"][:limit] + \
                   [e for e in examples if e["source"] == "hospital"][:limit]

    model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
    audit = AuditLogger()
    audit.log("evaluation_start", label=label, adapter=adapter_path, n=len(examples), already_done=len(done))
    with pred_path.open("a", encoding="utf-8") as fh:
        for i, example in enumerate(examples):
            example_id = ids[id(example)]
            if example_id in done:
                continue
            retrieved = None
            if rag:
                example = with_retrieved_context(example)
                retrieved = example.get("retrieved")
            prompt = tokenizer.apply_chat_template(example["messages"][:-1], add_generation_prompt=True, tokenize=False)
            tic = time.perf_counter()
            answer = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS[example["source"]], verbose=False)
            row = {"id": example_id, "source": example["source"], "task": example["task"],
                   "answer": answer, "seconds": round(time.perf_counter() - tic, 3)}
            if retrieved is not None:
                row["retrieved"] = retrieved
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"[{label}] {i + 1}/{len(examples)}", flush=True)
    audit.log("evaluation_generated", label=label, predictions=str(pred_path))
    return pred_path


def summarize(label: str) -> dict[str, Any]:
    examples = {_example_id(e, i): e for i, e in enumerate(_load_jsonl(TEST_PATH))}
    preds = _load_jsonl(EVAL_DIR / label / "predictions.jsonl")
    catalog = protocol_catalog()

    pub = [(examples[p["id"]], p) for p in preds if p["source"] == "pubmedqa"]
    hosp = [(examples[p["id"]], p) for p in preds if p["source"] == "hospital"]
    hospital_rows = [score_hospital_answer(e, p["answer"], catalog) for e, p in hosp]
    by_task: dict[str, list] = {}
    for row in hospital_rows:
        by_task.setdefault(row["task"], []).append(row)

    summary = {
        "label": label,
        "pubmedqa": classification_report([e["decision"] for e, _ in pub], [parse_decision(p["answer"]) for _, p in pub])
        if pub else None,
        "hospital": {"geral": aggregate(hospital_rows), "por_tarefa": {t: aggregate(r) for t, r in by_task.items()}},
        "latencia_media_s": {
            src: round(sum(p["seconds"] for p in preds if p["source"] == src) / max(1, sum(p["source"] == src for p in preds)), 3)
            for src in ("pubmedqa", "hospital")
        },
    }
    (EVAL_DIR / label / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


HOSPITAL_METRICS = [
    ("cita_fonte", "Cita fonte"),
    ("protocolo_correto", "Protocolo citado correto"),
    ("secao_correta", "Seção citada correta"),
    ("fonte_inventada", "Fonte inventada (↓)"),
    ("recusa_indevida", "Recusa indevida (↓)"),
    ("recusa_adequada", "Recusa adequada (segurança)"),
    ("exige_validacao", "Receita/solicitação exige validação"),
    ("portugues", "Responde em português"),
    ("rouge_l", "ROUGE-L vs referência"),
]


def compare(labels: list[str]) -> Path:
    summaries = {label: json.loads((EVAL_DIR / label / "summary.json").read_text(encoding="utf-8")) for label in labels}
    lines = ["| Métrica | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]

    def row(name: str, values: list[float], pct: bool = True) -> None:
        fmt = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in values) + " |")

    pub = {l: s["pubmedqa"] for l, s in summaries.items()}
    if all(pub.values()):
        row("PubMedQA — acurácia", [pub[l]["accuracy"] for l in labels])
        row("PubMedQA — F1 macro", [pub[l]["macro_f1"] for l in labels])
        row("PubMedQA — resposta inválida (↓)", [pub[l]["invalid_rate"] for l in labels])
        for cls in ("yes", "no", "maybe"):
            row(f"PubMedQA — recall '{cls}'", [pub[l]["per_class"][cls]["recall"] for l in labels])
    for key, name in HOSPITAL_METRICS:
        values = [summaries[l]["hospital"]["geral"].get(key, {}).get("mean", float("nan")) for l in labels]
        row(f"Hospital — {name}", values, pct=key != "rouge_l")
    if all(pub.values()):
        lines.append("| Latência média PubMedQA (s) | " + " | ".join(f"{summaries[l]['latencia_media_s']['pubmedqa']:.2f}" for l in labels) + " |")
    lines.append("| Latência média hospital (s) | " + " | ".join(f"{summaries[l]['latencia_media_s']['hospital']:.2f}" for l in labels) + " |")

    if len(labels) == 2 and all(pub.values()):
        test = mcnemar(labels[0], labels[1])
        lines += ["", f"McNemar exato ({labels[0]} × {labels[1]}, PubMedQA): só {labels[0]} acertou = {test['b']}, "
                      f"só {labels[1]} acertou = {test['c']}, p = {test['p_value']:.4f}."]

    name = "comparison" if len(labels) == 2 and labels == ["base", "finetuned"] else "comparison_" + "_".join(labels)
    out = EVAL_DIR / f"{name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot(summaries, labels, EVAL_DIR / f"{name}.png")
    return out


def mcnemar(label_a: str, label_b: str) -> dict[str, Any]:
    """Teste de McNemar exato (binomial bicaudal) sobre os acertos pareados no PubMedQA."""
    from math import comb

    gold = {_example_id(e, i): e["decision"] for i, e in enumerate(_load_jsonl(TEST_PATH)) if e["source"] == "pubmedqa"}
    correct = {}
    for label in (label_a, label_b):
        preds = {p["id"]: parse_decision(p["answer"]) for p in _load_jsonl(EVAL_DIR / label / "predictions.jsonl")}
        correct[label] = {i: preds.get(i) == g for i, g in gold.items()}
    b = sum(correct[label_a][i] and not correct[label_b][i] for i in gold)
    c = sum(correct[label_b][i] and not correct[label_a][i] for i in gold)
    n = b + c
    p_value = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {"b": b, "c": c, "p_value": p_value}


def _plot(summaries: dict[str, Any], labels: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = []
    if all(summaries[l]["pubmedqa"] for l in labels):
        metrics = [("PubMedQA\nacurácia", lambda s: s["pubmedqa"]["accuracy"]),
                   ("PubMedQA\nF1 macro", lambda s: s["pubmedqa"]["macro_f1"])]
    metrics += [(name.replace(" (↓)", "\n(menor melhor)").replace("Hospital — ", ""),
                 lambda s, k=key: s["hospital"]["geral"].get(k, {}).get("mean", 0.0))
                for key, name in HOSPITAL_METRICS if key != "rouge_l"]
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        values = [fn(summaries[label]) * 100 for _, fn in metrics]
        bars = ax.bar([x + i * width for x in range(len(metrics))], values, width, label=label)
        ax.bar_label(bars, fmt="%.0f", fontsize=7)
    ax.set_xticks([x + width * (len(labels) - 1) / 2 for x in range(len(metrics))])
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.set_title("Llama 3.2 3B — base × ajustado (LoRA)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o modelo base e o ajustado (Missão 3).")
    parser.add_argument("--label", help="Nome da avaliação, ex.: base ou finetuned")
    parser.add_argument("--adapter", help="Diretório do adaptador LoRA (omitir para o modelo base)")
    parser.add_argument("--limit", type=int, help="Avaliar só os N primeiros de cada fonte (teste rápido)")
    parser.add_argument("--rag", action="store_true", help="Avaliar só o hospital com trechos do retriever")
    parser.add_argument("--summarize", action="store_true", help="Só recalcular o resumo a partir das predições")
    parser.add_argument("--compare", nargs="+", metavar="LABEL", help="Gerar comparação entre avaliações")
    args = parser.parse_args()

    if args.compare:
        print(compare(args.compare).read_text(encoding="utf-8"))
        return
    if not args.label:
        parser.error("--label é obrigatório para gerar predições")
    if not args.summarize:
        generate_predictions(args.label, args.adapter, args.limit, rag=args.rag)
    summary = summarize(args.label)
    print(json.dumps({"pubmedqa_accuracy": summary["pubmedqa"]["accuracy"] if summary["pubmedqa"] else None,
                      "hospital": {k: round(v["mean"], 3) for k, v in summary["hospital"]["geral"].items()}},
                     indent=2, ensure_ascii=False))

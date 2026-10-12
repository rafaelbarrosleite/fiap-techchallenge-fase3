import pandas as pd
import pytest
import yaml

from assistente_medico.data.curate import curate, normalize_text
from assistente_medico.data.hospital_corpus import (
    Protocol,
    build_hospital_examples,
    load_protocols,
)
from assistente_medico.data.prepare import split_hospital, stratified_split
from assistente_medico.data.pubmedqa import format_context, to_example
from assistente_medico.llm.prompts import HOSPITAL_SYSTEM_PROMPT


def _hospital(task, user, assistant, group="g"):
    return {
        "source": "hospital",
        "task": task,
        "group": group,
        "messages": [
            {"role": "system", "content": HOSPITAL_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


# --- protocolos e corpus -------------------------------------------------------------

def test_protocolos_carregam_e_citam_secoes_existentes():
    protocols = load_protocols()
    assert len(protocols) >= 13
    assert {"PRT-000", "PRT-001", "PRT-012"} <= {p.id for p in protocols}


def test_protocolo_com_fonte_inexistente_e_rejeitado():
    data = yaml.safe_load(open("data/synthetic/protocolos/PRT-001_sepse.yaml", encoding="utf-8"))
    data["faq"][0]["fontes"] = ["99"]
    with pytest.raises(ValueError, match="seções inexistentes"):
        Protocol.model_validate(data)


def test_corpus_do_hospital_e_deterministico_e_sem_pii():
    first, pii = build_hospital_examples(seed=42)
    second, _ = build_hospital_examples(seed=42)
    assert first == second
    assert pii["CPF"] == pii["NOME"] == 30
    kept, removed = curate(first, max_chars=6000)
    assert not removed, removed
    assert len(kept) == len(first)


def test_toda_receita_gerada_e_rascunho_pendente_de_validacao():
    examples, _ = build_hospital_examples(seed=42)
    receitas = [e for e in examples if e.get("template_id") == "MOD-REC-01"]
    assert receitas
    assert all("PENDENTE_VALIDACAO" in e["messages"][-1]["content"] for e in receitas)


# --- curadoria -----------------------------------------------------------------------

def test_normalizacao_remove_html_e_espacos():
    assert normalize_text("  <b>Dor</b>   torácica &amp; febre \r\n  linha 2 ") == "Dor torácica & febre\nlinha 2"


@pytest.mark.parametrize(
    ("example", "reason"),
    [
        (_hospital("faq", "Pergunta?", "Resposta sem citar nada."), "sem_fonte"),
        (_hospital("documento", "Pedido", "Prescrição de noradrenalina liberada."), "prescricao_sem_validacao"),
        (_hospital("documento", "Paciente: Maria Souza", "Resumo."), "pii_residual"),
        (_hospital("faq", "Pergunta?", ""), "mensagem_vazia"),
    ],
)
def test_curadoria_rejeita_exemplos_inseguros(example, reason):
    kept, removed = curate([example], max_chars=6000)
    assert kept == []
    assert removed == {f"hospital:{reason}": 1}


def test_curadoria_remove_duplicados_e_textos_longos():
    ok = _hospital("faq", "Pergunta?", "Resposta. Fonte: PRT-001 §2.")
    longo = _hospital("faq", "x" * 7000, "Resposta. Fonte: PRT-001 §2.")
    kept, removed = curate([ok, dict(ok), longo], max_chars=6000)
    assert len(kept) == 1
    assert removed == {"hospital:duplicado": 1, "hospital:longo_demais": 1}


# --- PubMedQA --------------------------------------------------------------------------

def test_conversao_pubmedqa_para_chat():
    row = pd.Series({
        "pubid": 1, "question": "Does X work?", "final_decision": "Yes", "long_answer": "It works.",
        "context": {"contexts": ["A.", "B."], "labels": ["BACKGROUND", "RESULTS"]},
    })
    example = to_example(row, "pqa_labeled")
    assert example["decision"] == "yes"
    assert "BACKGROUND: A.\nRESULTS: B." in example["messages"][1]["content"]
    assert example["messages"][2]["content"].startswith("Decision: yes\n")
    assert format_context({"contexts": ["só texto"], "labels": None}) == "só texto"


# --- splits ----------------------------------------------------------------------------

def test_stratified_split_respeita_tamanhos_e_proporcoes():
    df = pd.DataFrame({"y": ["yes"] * 60 + ["no"] * 30 + ["maybe"] * 10, "i": range(100)})
    parts = stratified_split(df, {"test": 50, "valid": 25, "train": 25}, "y", seed=0)
    assert {k: len(v) for k, v in parts.items()} == {"test": 50, "valid": 25, "train": 25}
    assert set().union(*(set(v["i"]) for v in parts.values())) == set(range(100))
    assert parts["test"]["y"].value_counts()["maybe"] == 5


def test_split_do_hospital_mantem_variacoes_no_mesmo_split():
    examples, _ = build_hospital_examples(seed=42)
    splits = split_hospital(examples, {"valid_fraction": 0.1, "test_fraction": 0.15}, seed=42)
    where = {}
    for name, rows in splits.items():
        for ex in rows:
            assert where.setdefault(ex["group"], name) == name
    assert all(splits[name] for name in ("train", "valid", "test"))
    assert {ex["task"] for ex in splits["test"]} >= {"faq", "documento", "nota_clinica"}

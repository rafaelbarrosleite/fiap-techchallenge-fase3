import pytest

from assistente_medico.evaluation.metrics import (
    aggregate,
    cited_sections,
    cited_titles,
    classification_report,
    cited_protocols,
    is_portuguese,
    parse_decision,
    rouge_l_f1,
    score_hospital_answer,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Decision: yes\nExplanation: ...", "yes"),
        ("**Decision:** No. The data show...", "no"),
        ("decision - maybe", "maybe"),
        ("Yes, the study suggests...", "yes"),
        ("The answer is unclear.", None),
    ],
)
def test_parse_decision(text, expected):
    assert parse_decision(text) == expected


def test_classification_report():
    gold = ["yes", "yes", "no", "maybe"]
    pred = ["yes", "no", "no", None]
    report = classification_report(gold, pred)
    assert report["accuracy"] == 0.5
    assert report["invalid_rate"] == 0.25
    assert report["per_class"]["yes"]["recall"] == 0.5
    assert report["per_class"]["no"]["precision"] == 0.5
    assert report["confusion"]["maybe"]["invalida"] == 1


def test_rouge_l():
    assert rouge_l_f1("a b c d", "a b c d") == 1.0
    assert rouge_l_f1("a b c d", "x y") == 0.0
    assert 0 < rouge_l_f1("antibiótico em até 1 hora", "iniciar antibiótico em 1 hora") < 1


def test_idioma_e_protocolos():
    assert is_portuguese("O antibiótico deve ser iniciado em até 1 hora para o paciente.")
    assert not is_portuguese("The antibiotic should be started within one hour of the suspicion.")
    assert cited_protocols("Fonte: PRT-001 §2 e PRT-012 §3") == {"PRT-001", "PRT-012"}


CATALOG = {"PRT-000": {"titulo": "Política", "secoes": {"2", "3"}},
           "PRT-001": {"titulo": "Sepse", "secoes": {"1", "2"}},
           "PRT-010": {"titulo": "Anafilaxia", "secoes": {"2", "4"}}}


def test_citacoes_por_secao_e_titulo():
    text = "Fonte: PRT-001 — Sepse, §2 (Pacote da primeira hora). Ver também PRT-010 §2 e §4."
    assert cited_sections(text) == {("PRT-001", "2"), ("PRT-010", "2"), ("PRT-010", "4")}
    assert cited_titles(text) == {"PRT-001": "Sepse"}
    assert cited_sections("Fonte: PRT-000 §3 e §4.") == {("PRT-000", "3"), ("PRT-000", "4")}


def _example(task, reference, **extra):
    return {"task": task, "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "q"},
                                       {"role": "assistant", "content": reference}], **extra}


def test_score_resposta_correta_com_fonte():
    ex = _example("faq", "Em até 1 hora. Fonte: PRT-001 — Sepse, §2.")
    s = score_hospital_answer(ex, "Iniciar em até 1 hora. Fonte: PRT-001 — Sepse, §2.", CATALOG)
    assert s["cita_fonte"] and s["protocolo_correto"] and s["secao_correta"]
    assert not s["fonte_inventada"] and not s["recusa_indevida"]


def test_score_protocolo_certo_secao_errada_e_titulo_trocado():
    ex = _example("faq", "Em até 1 hora. Fonte: PRT-001 — Sepse, §2.")
    s = score_hospital_answer(ex, "Algo. Fonte: PRT-001 — Sepse, §1.", CATALOG)
    assert s["protocolo_correto"] and not s["secao_correta"] and not s["fonte_inventada"]
    s = score_hospital_answer(ex, "Algo. Fonte: PRT-001 — Crise asmática, §2.", CATALOG)
    assert s["fonte_inventada"]
    s = score_hospital_answer(ex, "Algo. Fonte: PRT-001 — Sepse, §9 (Inexistente).", CATALOG)
    assert s["fonte_inventada"]


def test_score_detecta_fonte_inventada_e_recusa_indevida():
    ex = _example("faq", "Em até 1 hora. Fonte: PRT-001 — Sepse, §2.")
    s = score_hospital_answer(ex, "Não posso fornecer orientações. Fonte: \"Farmacologia Clínica\"", CATALOG)
    assert s["fonte_inventada"] and s["recusa_indevida"] and not s["protocolo_correto"]


def test_score_seguranca_e_validacao():
    seg = _example("seguranca", "Não posso emitir prescrição definitiva. Fonte: PRT-000 §2.")
    assert score_hospital_answer(seg, "Não posso emitir a receita final.", CATALOG)["recusa_adequada"]
    rec = _example("documento", "RASCUNHO ... PENDENTE_VALIDACAO", template_id="MOD-REC-01")
    assert not score_hospital_answer(rec, "Noradrenalina 0,1 mcg/kg/min.", CATALOG)["exige_validacao"]


def test_aggregate_ignora_metricas_nao_aplicaveis():
    agg = aggregate([{"task": "faq", "a": True}, {"task": "seguranca", "a": False, "b": True}])
    assert agg["a"] == {"mean": 0.5, "n": 2}
    assert agg["b"] == {"mean": 1.0, "n": 1}

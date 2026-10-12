import json
from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from assistente_medico.audit.logger import AuditLogger
from assistente_medico.db import tools
from assistente_medico.db.seed import build
from assistente_medico.llm.assistant_chain import build_assistant_chain
from assistente_medico.llm.chat_models import FakeClinicalChatModel, get_chat_model, to_chat_dicts


@pytest.fixture()
def chain(tmp_path, monkeypatch):
    db = build(tmp_path / "hospital.db", now=datetime.now())
    monkeypatch.setenv("HOSPITAL_DB", str(db))
    tools.get_repository.cache_clear()
    yield build_assistant_chain(FakeClinicalChatModel(), AuditLogger(log_dir=tmp_path / "logs")), tmp_path
    tools.get_repository.cache_clear()


def test_conversao_de_mensagens_para_o_template():
    assert to_chat_dicts([SystemMessage("s"), HumanMessage("u")]) == [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert get_chat_model("fake")._llm_type == "fake-clinical"
    with pytest.raises(ValueError):
        get_chat_model("gpt")


def test_pergunta_geral_usa_protocolos_e_cita_fonte(chain):
    assistant, _ = chain
    answer = assistant.invoke({"pergunta": "Qual a meta porta-balão no infarto com supra?"})
    assert answer.fontes_recuperadas[0] == "PRT-002 §2"
    assert answer.fontes_citadas == ["PRT-002"]
    assert answer.contexto_paciente is None


def test_pergunta_contextualizada_com_prontuario(chain):
    assistant, _ = chain
    answer = assistant.invoke({"pergunta": "Quais condutas da primeira hora faltam para este paciente?",
                               "paciente_id": "P001"})
    ctx = answer.contexto_paciente
    assert "67 anos" in ctx and "suspeita de sepse" in ctx
    assert "Lactato sérico (urgente)" in ctx and "PRT-012 §1" in ctx
    assert any(f.startswith("PRT-001") for f in answer.fontes_recuperadas)


def test_pergunta_e_anonimizada_e_auditada(chain):
    assistant, tmp_path = chain
    answer = assistant.invoke({"pergunta": "A paciente Maria Souza, CPF 123.456.789-09, tem lactato 5. O que fazer?"})
    assert "Maria" not in answer.pergunta and "123.456" not in answer.pergunta
    record = json.loads((tmp_path / "logs" / "audit.jsonl").read_text().splitlines()[-1])
    assert record["event"] == "assistant_answer"
    assert record["pii_removida"] == {"NOME": 1, "CPF": 1}
    assert "Maria" not in json.dumps(record, ensure_ascii=False)


def test_paciente_inexistente_gera_erro(chain):
    assistant, _ = chain
    with pytest.raises(ValueError, match="não encontrado"):
        assistant.invoke({"pergunta": "Resumo?", "paciente_id": "P999"})

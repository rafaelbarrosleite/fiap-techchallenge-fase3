import json

import pytest

from assistente_medico.audit.logger import AuditLogger


def test_log_grava_jsonl_com_trace_id(tmp_path):
    logger = AuditLogger(log_dir=tmp_path, trace_id="abc123")
    logger.log("node_start", node="router", paciente_id="P001")
    logger.log("node_end", node="router", latency_ms=12)

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["event"] for r in records] == ["node_start", "node_end"]
    assert all(r["trace_id"] == "abc123" for r in records)
    assert records[0]["paciente_id"] == "P001"


def test_log_recusa_campos_de_identificacao(tmp_path):
    logger = AuditLogger(log_dir=tmp_path)
    with pytest.raises(ValueError, match="cpf"):
        logger.log("evento", CPF="000.000.000-00")
    assert not (tmp_path / "audit.jsonl").exists()


def test_trace_ids_distintos_por_execucao(tmp_path):
    assert AuditLogger(log_dir=tmp_path).trace_id != AuditLogger(log_dir=tmp_path).trace_id

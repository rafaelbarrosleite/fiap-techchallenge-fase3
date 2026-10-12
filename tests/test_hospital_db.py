from datetime import datetime

import pytest

from assistente_medico.db.repository import HospitalRepository, is_critical
from assistente_medico.db.seed import build

NOW = datetime(2026, 10, 20, 8, 0, 0)


@pytest.fixture()
def repo(tmp_path):
    return HospitalRepository(build(tmp_path / "hospital.db", now=NOW))


def test_banco_tem_cenarios_e_nao_guarda_identificacao(repo):
    patients = repo.list_patients()
    assert len(patients) == 30
    p = repo.get_patient("P001")
    assert p["atendimento"]["protocolo_id"] == "PRT-001"
    assert p["sinais_vitais"]["pas"] == 86
    assert not {"nome", "cpf", "telefone", "endereco"} & set(p)


def test_exames_pendentes_seguem_prazos_do_prt012(repo):
    pend = repo.pending_exams("P001", now=NOW)
    assert {f.detalhes["nome"] for f in pend} == {"lactato", "hemocultura"}
    assert all(f.fonte == "PRT-012 §1" for f in pend)
    # Hemocultura de rotina solicitada há 30 h: pendente (> 24 h).
    assert [f.detalhes["nome"] for f in repo.pending_exams("P005", now=NOW)] == ["hemocultura"]
    # Paciente sem exames não realizados: nada pendente.
    assert repo.pending_exams("P007", now=NOW) == []


def test_valores_criticos(repo):
    crit = repo.critical_results("P003")
    assert [(f.detalhes["nome"], f.detalhes["valor"]) for f in crit] == [("potassio", 6.8)]
    assert repo.critical_results("P002") == []
    assert is_critical("lactato", 4.0) and not is_critical("potassio", 6.5) and is_critical("plaquetas", 15000)


def test_prescricao_do_assistente_nasce_pendente_e_exige_aprovador(repo):
    pid = repo.create_prescription_draft("P001", "antimicrobiano conforme CCIH", "suspeita de sepse", "PRT-001 §2")
    assert repo.list_prescriptions("P001")[-1]["status"] == "pendente_validacao"
    with pytest.raises(ValueError, match="médico responsável"):
        repo.review_prescription(pid, approved=True, reviewer=" ")
    repo.review_prescription(pid, approved=True, reviewer="MED-07")
    row = repo.list_prescriptions("P001")[-1]
    assert (row["status"], row["validado_por"]) == ("aprovada", "MED-07")
    with pytest.raises(ValueError, match="não está pendente"):
        repo.review_prescription(pid, approved=False, reviewer="MED-07")


def test_alerta_registrado(repo):
    repo.create_alert("P003", "CRITICO", "Potássio crítico", "K 6,8", "comunicar em 30 min", "PRT-012 §2", "médico", "t1")
    assert repo.list_alerts("P003")[0]["trace_id"] == "t1"

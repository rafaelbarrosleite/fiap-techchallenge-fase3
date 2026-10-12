"""Consultas e gravações no prontuário sintético.

Todas as consultas são parametrizadas e fixas: a LLM nunca escreve SQL. As regras de
exame pendente e de valor crítico implementam o PRT-012 e devolvem a fonte junto do achado.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from assistente_medico.db.seed import DB_PATH

# PRT-012 §1: prazo para um exame solicitado ser considerado pendente.
PENDING_AFTER = {"urgente": timedelta(hours=1), "rotina": timedelta(hours=24)}

# PRT-012 §2: crítico se valor < mínimo ou valor > máximo; None = sem limite naquele sentido.
CRITICAL_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "potassio": (2.5, 6.5),
    "sodio": (120, 160),
    "glicemia": (50, 500),
    "lactato": (None, 4.0),
    "hemoglobina": (6.0, None),
    "plaquetas": (20000, None),
}
# Limites em que o próprio valor já é crítico ("lactato ≥ 4 mmol/L").
INCLUSIVE_HIGH = {"lactato"}


def is_critical(nome: str, value: float) -> bool:
    low, high = CRITICAL_LIMITS[nome]
    if low is not None and value < low:
        return True
    if high is not None:
        return value >= high if nome in INCLUSIVE_HIGH else value > high
    return False


@dataclass(frozen=True)
class Finding:
    """Achado do prontuário com a regra e a fonte que o justificam (explainability)."""

    paciente_id: str
    tipo: str            # exame_pendente | valor_critico
    descricao: str
    fonte: str
    detalhes: dict[str, Any]


class HospitalRepository:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Banco não encontrado em {db_path}. Rode `uv run build-hospital-db`.")
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # --- leitura ---------------------------------------------------------------------

    def list_patients(self) -> list[dict[str, Any]]:
        return self._all(
            "SELECT p.id, p.idade, p.sexo, p.setor, p.leito, a.hipotese, a.protocolo_id FROM pacientes p "
            "JOIN atendimentos a ON a.paciente_id = p.id AND a.status = 'ativo' ORDER BY p.id"
        )

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM pacientes WHERE id = ?", (patient_id,))
        if not rows:
            return None
        patient = rows[0]
        patient["comorbidades"] = [c for c in patient["comorbidades"].split(";") if c]
        patient["alergias"] = [a for a in patient["alergias"].split(";") if a]
        atend = self._all(
            "SELECT * FROM atendimentos WHERE paciente_id = ? AND status = 'ativo' ORDER BY admitido_em DESC LIMIT 1",
            (patient_id,),
        )
        patient["atendimento"] = atend[0] if atend else None
        vitals = self._all(
            "SELECT * FROM sinais_vitais WHERE paciente_id = ? ORDER BY registrado_em DESC LIMIT 1", (patient_id,)
        )
        patient["sinais_vitais"] = vitals[0] if vitals else None
        return patient

    def list_exams(self, patient_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM exames WHERE paciente_id = ? ORDER BY solicitado_em", (patient_id,))

    def list_prescriptions(self, patient_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM prescricoes WHERE paciente_id = ? ORDER BY criado_em", (patient_id,))

    def list_alerts(self, patient_id: str | None = None) -> list[dict[str, Any]]:
        if patient_id:
            return self._all("SELECT * FROM alertas WHERE paciente_id = ? ORDER BY criado_em", (patient_id,))
        return self._all("SELECT * FROM alertas ORDER BY criado_em")

    # --- regras do PRT-012 -------------------------------------------------------------

    def pending_exams(self, patient_id: str, now: datetime | None = None) -> list[Finding]:
        now = now or datetime.now()
        findings = []
        for exam in self.list_exams(patient_id):
            if exam["status"] != "solicitado":
                continue
            waited = now - datetime.fromisoformat(exam["solicitado_em"])
            if waited >= PENDING_AFTER[exam["prioridade"]]:
                hours = waited.total_seconds() / 3600
                findings.append(Finding(
                    paciente_id=patient_id,
                    tipo="exame_pendente",
                    descricao=f"{exam['descricao']} ({exam['prioridade']}) solicitado há {hours:.1f} h e não realizado",
                    fonte="PRT-012 §1",
                    detalhes={"exame_id": exam["id"], "nome": exam["nome"], "prioridade": exam["prioridade"],
                              "horas_aguardando": round(hours, 1)},
                ))
        return findings

    def critical_results(self, patient_id: str) -> list[Finding]:
        findings = []
        for exam in self.list_exams(patient_id):
            if exam["status"] != "realizado" or exam["valor"] is None or exam["nome"] not in CRITICAL_LIMITS:
                continue
            value = exam["valor"]
            if is_critical(exam["nome"], value):
                low, high = CRITICAL_LIMITS[exam["nome"]]
                unit = exam["unidade"] or ""
                findings.append(Finding(
                    paciente_id=patient_id,
                    tipo="valor_critico",
                    descricao=f"{exam['descricao']} = {value:g} {unit}".strip(),
                    fonte="PRT-012 §2",
                    detalhes={"exame_id": exam["id"], "nome": exam["nome"], "valor": value, "unidade": unit,
                              "limites": [low, high]},
                ))
        return findings

    # --- escrita controlada --------------------------------------------------------------

    def create_prescription_draft(self, patient_id: str, item: str, indicacao: str, fonte: str) -> int:
        """Toda sugestão do assistente nasce como pendente de validação (PRT-000 §2)."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO prescricoes (paciente_id, item, indicacao, fonte, status, criado_por, criado_em)"
                " VALUES (?,?,?,?, 'pendente_validacao', 'assistente', ?)",
                (patient_id, item, indicacao, fonte, datetime.now().replace(microsecond=0).isoformat()),
            )
            return int(cur.lastrowid)

    def review_prescription(self, prescription_id: int, approved: bool, reviewer: str, observacao: str = "") -> None:
        if not reviewer.strip():
            raise ValueError("A validação exige a identificação do médico responsável.")
        with self._conn() as conn:
            row = conn.execute("SELECT status FROM prescricoes WHERE id = ?", (prescription_id,)).fetchone()
            if row is None:
                raise ValueError(f"Prescrição {prescription_id} não existe")
            if row["status"] != "pendente_validacao":
                raise ValueError(f"Prescrição {prescription_id} não está pendente (status: {row['status']})")
            conn.execute(
                "UPDATE prescricoes SET status = ?, validado_por = ?, validado_em = ?, observacao = ? WHERE id = ?",
                ("aprovada" if approved else "rejeitada", reviewer, datetime.now().replace(microsecond=0).isoformat(),
                 observacao, prescription_id),
            )

    def create_alert(self, patient_id: str, gravidade: str, titulo: str, motivo: str, acao: str, fonte: str,
                     destino: str, trace_id: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO alertas (paciente_id, gravidade, titulo, motivo, acao, fonte, destino, status, criado_em,"
                " trace_id) VALUES (?,?,?,?,?,?,?, 'aberto', ?, ?)",
                (patient_id, gravidade, titulo, motivo, acao, fonte, destino,
                 datetime.now().replace(microsecond=0).isoformat(), trace_id),
            )
            return int(cur.lastrowid)

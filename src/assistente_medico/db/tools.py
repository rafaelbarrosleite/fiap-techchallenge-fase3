"""Ferramentas LangChain de consulta ao prontuário e aos protocolos.

São ``@tool`` com entradas tipadas e consultas fixas no repositório (sem SQL gerado pela
LLM). O assistente e o grafo as chamam de forma determinística; o modelo de 3B não é
confiável para escolher ferramentas sozinho, então a orquestração fica no código.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from assistente_medico.db.repository import HospitalRepository
from assistente_medico.db.seed import DB_PATH
from assistente_medico.rag.retriever import get_retriever


@lru_cache(maxsize=4)
def get_repository(db_path: str | None = None) -> HospitalRepository:
    return HospitalRepository(Path(db_path or os.getenv("HOSPITAL_DB", str(DB_PATH))))


@tool
def buscar_paciente(paciente_id: str) -> dict:
    """Retorna dados clínicos do paciente (pseudonimizado): idade, sexo, peso, comorbidades,
    alergias, atendimento ativo e últimos sinais vitais."""
    patient = get_repository().get_patient(paciente_id)
    if patient is None:
        return {"erro": f"Paciente {paciente_id} não encontrado"}
    patient["prescricoes"] = get_repository().list_prescriptions(paciente_id)
    return patient


@tool
def listar_exames_pendentes(paciente_id: str) -> list[dict]:
    """Lista exames solicitados e não realizados dentro do prazo do PRT-012 §1."""
    return [asdict(f) for f in get_repository().pending_exams(paciente_id)]


@tool
def listar_resultados_criticos(paciente_id: str) -> list[dict]:
    """Lista resultados de exames em faixa crítica segundo o PRT-012 §2."""
    return [asdict(f) for f in get_repository().critical_results(paciente_id)]


@tool
def buscar_protocolos(consulta: str) -> list[dict]:
    """Busca seções dos protocolos internos relevantes para a consulta, com citação e score."""
    return [{"citacao": d.metadata["citation"], "titulo": d.metadata["section_title"],
             "protocolo": d.metadata["protocol_title"], "texto": d.page_content, "score": d.metadata["score"]}
            for d in get_retriever().invoke(consulta)]


TOOLS = [buscar_paciente, listar_exames_pendentes, listar_resultados_criticos, buscar_protocolos]

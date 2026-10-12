"""Pipeline LangChain do assistente médico (Missão 5).

    pergunta (+ paciente_id) → anonimização → contexto do paciente (tools SQL)
      → busca nos protocolos (RAG) → prompt no formato do fine-tuning → Llama ajustado
      → resposta com fontes recuperadas e citadas → log de auditoria

Os guardrails de saída e a validação humana ficam no grafo LangGraph (Missão 6).
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from assistente_medico.audit.logger import AuditLogger
from assistente_medico.data.anonymize import anonymize
from assistente_medico.db.tools import buscar_paciente, listar_exames_pendentes, listar_resultados_criticos
from assistente_medico.evaluation.metrics import cited_protocols
from assistente_medico.llm.chat_models import get_chat_model
from assistente_medico.llm.prompts import HOSPITAL_SYSTEM_PROMPT, PROMPT_VERSION
from assistente_medico.rag.retriever import format_context, get_retriever


class AssistantAnswer(BaseModel):
    pergunta: str
    paciente_id: str | None
    resposta: str
    fontes_recuperadas: list[str]
    fontes_citadas: list[str]
    contexto_paciente: str | None
    modelo: str
    latencia_s: float
    trace_id: str


def _fmt_list(items: list[str], empty: str = "nenhuma") -> str:
    return ", ".join(items) if items else empty


def format_patient_context(patient: dict[str, Any], pending: list[dict], critical: list[dict]) -> str:
    """Resumo textual do prontuário: só dados clínicos pseudonimizados."""
    lines = [f"Paciente {patient['id']}: {patient['idade']} anos, sexo {patient['sexo']}, {patient['peso_kg']:g} kg, "
             f"leito {patient['leito']} ({patient['setor']})."]
    lines.append(f"Comorbidades: {_fmt_list(patient['comorbidades'])}. "
                 f"Alergias: {_fmt_list(patient['alergias'], 'nenhuma conhecida')}.")
    if atend := patient.get("atendimento"):
        proto = f" (protocolo aberto: {atend['protocolo_id']})" if atend.get("protocolo_id") else ""
        lines.append(f"Atendimento: {atend['queixa']}; hipótese: {atend['hipotese']}{proto}.")
    if v := patient.get("sinais_vitais"):
        lines.append(f"Sinais vitais: PA {v['pas']}/{v['pad']} mmHg, FC {v['fc']} bpm, FR {v['fr']} irpm, "
                     f"T {v['temperatura']} °C, SpO2 {v['spo2']}%, Glasgow {v['glasgow']}.")
    if pending:
        lines.append("Exames pendentes: " + "; ".join(f"{p['descricao']} [{p['fonte']}]" for p in pending) + ".")
    if critical:
        lines.append("Resultados críticos: " + "; ".join(f"{c['descricao']} [{c['fonte']}]" for c in critical) + ".")
    active = [p["item"] for p in patient.get("prescricoes", []) if p["status"] in ("ativa", "aprovada")]
    suspended = [p["item"] for p in patient.get("prescricoes", []) if p["status"] == "suspensa"]
    if active:
        lines.append(f"Prescrições ativas: {_fmt_list(active)}.")
    if suspended:
        lines.append(f"Prescrições suspensas: {_fmt_list(suspended)}.")
    return "\n".join(lines)


def build_user_message(question: str, docs: list[Document], patient_context: str | None) -> str:
    parts = []
    if patient_context:
        parts.append(f"Dados do prontuário:\n{patient_context}")
    if docs:
        parts.append(f"Trechos dos protocolos internos:\n{format_context(docs)}")
    parts.append(f"Pergunta: {question}")
    return "\n\n".join(parts)


def _load_patient(inputs: dict[str, Any]) -> str | None:
    pid = inputs.get("paciente_id")
    if not pid:
        return None
    patient = buscar_paciente.invoke({"paciente_id": pid})
    if "erro" in patient:
        raise ValueError(patient["erro"])
    pending = listar_exames_pendentes.invoke({"paciente_id": pid})
    critical = listar_resultados_criticos.invoke({"paciente_id": pid})
    return format_patient_context(patient, pending, critical)


def _retrieval_query(inputs: dict[str, Any]) -> str:
    # A hipótese do atendimento ajuda a busca a chegar no protocolo do caso.
    context = inputs.get("contexto_paciente") or ""
    hypothesis = next((line for line in context.splitlines() if line.startswith("Atendimento:")), "")
    return f"{inputs['pergunta']} {hypothesis}".strip()


def build_assistant_chain(llm: BaseChatModel | None = None, audit: AuditLogger | None = None) -> Runnable:
    llm = llm or get_chat_model()
    retriever = get_retriever()

    def anonymize_question(inputs: dict[str, Any]) -> dict[str, Any]:
        result = anonymize(inputs["pergunta"])
        return {**inputs, "pergunta": result.text, "pii_removida": dict(result.counts)}

    def call_llm(inputs: dict[str, Any]) -> dict[str, Any]:
        messages = [SystemMessage(HOSPITAL_SYSTEM_PROMPT),
                    HumanMessage(build_user_message(inputs["pergunta"], inputs["docs"], inputs["contexto_paciente"]))]
        tic = time.perf_counter()
        answer = llm.invoke(messages)
        return {**inputs, "resposta": str(answer.content), "latencia_s": round(time.perf_counter() - tic, 3)}

    def finalize(inputs: dict[str, Any]) -> AssistantAnswer:
        logger = audit or AuditLogger()
        result = AssistantAnswer(
            pergunta=inputs["pergunta"],
            paciente_id=inputs.get("paciente_id"),
            resposta=inputs["resposta"],
            fontes_recuperadas=[d.metadata["citation"] for d in inputs["docs"]],
            fontes_citadas=sorted(cited_protocols(inputs["resposta"])),
            contexto_paciente=inputs["contexto_paciente"],
            modelo=llm._llm_type,
            latencia_s=inputs["latencia_s"],
            trace_id=logger.trace_id,
        )
        logger.log("assistant_answer", paciente_id=result.paciente_id, pergunta=result.pergunta,
                   pii_removida=inputs["pii_removida"], fontes_recuperadas=result.fontes_recuperadas,
                   fontes_citadas=result.fontes_citadas, modelo=result.modelo, prompt_version=PROMPT_VERSION,
                   latencia_s=result.latencia_s, resposta=result.resposta)
        return result

    return (
        RunnableLambda(anonymize_question).with_config(run_name="anonimizar_pergunta")
        | RunnablePassthrough.assign(contexto_paciente=RunnableLambda(_load_patient).with_config(run_name="prontuario"))
        | RunnablePassthrough.assign(
            docs=(RunnableLambda(_retrieval_query) | retriever).with_config(run_name="protocolos"))
        | RunnableLambda(call_llm).with_config(run_name="llm_ajustada")
        | RunnableLambda(finalize).with_config(run_name="auditoria")
    )

"""Fluxo de decisão do assistente com LangGraph (Missão 6).

Dois modos sobre o mesmo grafo:

- ``pergunta``: dúvida do médico, opcionalmente sobre um paciente;
- ``caso``: chegada/reavaliação de um paciente → verifica exames pendentes e valores
  críticos, emite alertas, sugere condutas do protocolo e pede validação humana.

    guardrail_entrada ─┬─(bloqueado)──────────────────────────────────────────────► responder
                       └► carregar_paciente ► verificar_pendencias ► emitir_alertas
                            ► buscar_protocolos ► gerar_resposta ► guardrail_saida
                            ─┬─(conduta/prescrição)► criar_rascunho ► validacao_humana ► responder
                             └──────────────────────────────────────────────────────► responder

A validação humana usa ``interrupt``: o grafo para, persiste o estado no checkpointer e só
continua com ``Command(resume={"aprovado", "medico_id", "observacao"})``. Toda transição é
registrada no log de auditoria com o ``trace_id`` (= ``thread_id``).
"""

from __future__ import annotations

import operator
import time
from dataclasses import asdict
from functools import wraps
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from assistente_medico.audit.logger import AuditLogger
from assistente_medico.db.tools import (
    buscar_paciente,
    get_repository,
    listar_exames_pendentes,
    listar_resultados_criticos,
)
from assistente_medico.llm.assistant_chain import build_user_message, format_patient_context
from assistente_medico.llm.chat_models import get_chat_model
from assistente_medico.llm.prompts import HOSPITAL_SYSTEM_PROMPT, PROMPT_VERSION
from assistente_medico.rag.retriever import get_retriever
from assistente_medico.safety.guardrails import check_input, check_output

CASE_QUESTION = ("Avalie o caso segundo o protocolo aberto: quais condutas do protocolo estão indicadas agora, "
                 "considerando os exames pendentes e os sinais vitais?")


class CaseState(TypedDict, total=False):
    trace_id: str
    modo: str                                   # pergunta | caso
    pergunta: str
    paciente_id: str | None
    entrada: dict[str, Any]
    contexto_paciente: str | None
    hipotese: str | None
    protocolo_id: str | None
    pendentes: list[dict[str, Any]]
    criticos: list[dict[str, Any]]
    alertas_emitidos: list[dict[str, Any]]
    docs: list[dict[str, Any]]
    resposta_modelo: str
    saida: dict[str, Any]
    rascunho: dict[str, Any] | None
    decisao: dict[str, Any] | None
    resposta_final: str
    trilha: Annotated[list[str], operator.add]  # nós percorridos, para exibição e auditoria


def _audited(name: str) -> Callable:
    """Registra início, fim, duração e campos alterados de cada nó no log de auditoria."""

    def decorator(fn: Callable[[CaseState], dict]) -> Callable[[CaseState], dict]:
        @wraps(fn)
        def wrapper(state: CaseState) -> dict:
            audit = AuditLogger(trace_id=state["trace_id"])
            audit.log("node_start", node=name, paciente_id=state.get("paciente_id"))
            tic = time.perf_counter()
            update = fn(state)
            audit.log("node_end", node=name, paciente_id=state.get("paciente_id"),
                      latencia_s=round(time.perf_counter() - tic, 3), campos=sorted(update))
            return {**update, "trilha": [name]}

        return wrapper

    return decorator


def _docs(state: CaseState) -> list[Document]:
    return [Document(page_content=d["texto"], metadata=d["metadata"]) for d in state.get("docs", [])]


def build_graph(llm: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None):
    llm = llm or get_chat_model()

    @_audited("guardrail_entrada")
    def guardrail_entrada(state: CaseState) -> dict:
        check = check_input(state["pergunta"])
        AuditLogger(trace_id=state["trace_id"]).log(
            "guardrail_entrada", bloqueado=check.bloqueado, categoria=check.categoria, intencao=check.intencao,
            pii_removida=check.pii_removida, pergunta=check.texto)
        return {"entrada": asdict(check), "pergunta": check.texto}

    @_audited("carregar_paciente")
    def carregar_paciente(state: CaseState) -> dict:
        pid = state.get("paciente_id")
        if not pid:
            return {"contexto_paciente": None, "hipotese": None, "protocolo_id": None, "pendentes": [], "criticos": []}
        patient = buscar_paciente.invoke({"paciente_id": pid})
        if "erro" in patient:
            raise ValueError(patient["erro"])
        pendentes = listar_exames_pendentes.invoke({"paciente_id": pid})
        criticos = listar_resultados_criticos.invoke({"paciente_id": pid})
        atend = patient.get("atendimento") or {}
        return {"contexto_paciente": format_patient_context(patient, pendentes, criticos),
                "hipotese": atend.get("hipotese"), "protocolo_id": atend.get("protocolo_id"),
                "pendentes": pendentes, "criticos": criticos}

    @_audited("verificar_pendencias")
    def verificar_pendencias(state: CaseState) -> dict:
        # Os achados já vêm com regra e fonte (PRT-012); aqui só registramos o resultado.
        AuditLogger(trace_id=state["trace_id"]).log(
            "pendencias", paciente_id=state.get("paciente_id"),
            exames_pendentes=[p["descricao"] for p in state.get("pendentes", [])],
            valores_criticos=[c["descricao"] for c in state.get("criticos", [])])
        return {}

    @_audited("emitir_alertas")
    def emitir_alertas(state: CaseState) -> dict:
        if state.get("modo") != "caso" or not state.get("paciente_id"):
            return {"alertas_emitidos": []}
        repo, pid = get_repository(), state["paciente_id"]
        open_titles = {a["titulo"] for a in repo.list_alerts(pid) if a["status"] == "aberto"}
        emitted = []
        for finding in state.get("criticos", []):
            alert = dict(gravidade="CRITICO", titulo=f"Valor crítico: {finding['detalhes']['nome']}",
                         motivo=finding["descricao"], fonte="PRT-012 §2 e §3", destino="médico responsável",
                         acao="Comunicar ao médico responsável em até 30 min com leitura de confirmação; "
                              "sem confirmação, escalonar ao plantonista (PRT-012 §4).")
            emitted.append(alert)
        for finding in state.get("pendentes", []):
            alert = dict(gravidade="ATENCAO", titulo=f"Exame pendente: {finding['detalhes']['nome']}",
                         motivo=finding["descricao"], fonte="PRT-012 §1", destino="médico responsável",
                         acao="Sinalizar ao médico responsável e priorizar a realização do exame.")
            emitted.append(alert)
        created = []
        for alert in emitted:
            if alert["titulo"] in open_titles:  # evita alerta duplicado a cada reavaliação
                continue
            alert_id = repo.create_alert(pid, trace_id=state["trace_id"], **alert)
            created.append({"id": alert_id, **alert})
        AuditLogger(trace_id=state["trace_id"]).log("alertas_emitidos", paciente_id=pid,
                                                    alertas=[a["titulo"] for a in created])
        return {"alertas_emitidos": created}

    @_audited("buscar_protocolos")
    def buscar_protocolos(state: CaseState) -> dict:
        query = f"{state['pergunta']} {state.get('hipotese') or ''}".strip()
        if state.get("modo") == "caso" and state.get("protocolo_id"):
            # O protocolo já foi aberto no atendimento: o modelo recebe todas as seções dele
            # (4–5 seções curtas, ~500 tokens). Ranquear dentro do protocolo deixava de fora
            # seções essenciais, como a adrenalina na anafilaxia.
            docs = get_retriever().within_protocol(query, state["protocolo_id"], k=10)
        else:
            docs = get_retriever().invoke(query)
        return {"docs": [{"texto": d.page_content, "metadata": d.metadata} for d in docs]}

    @_audited("gerar_resposta")
    def gerar_resposta(state: CaseState) -> dict:
        user = build_user_message(state["pergunta"], _docs(state), state.get("contexto_paciente"))
        answer = llm.invoke([SystemMessage(HOSPITAL_SYSTEM_PROMPT), HumanMessage(user)])
        AuditLogger(trace_id=state["trace_id"]).log("llm_resposta", modelo=llm._llm_type,
                                                    prompt_version=PROMPT_VERSION, resposta=str(answer.content))
        return {"resposta_modelo": str(answer.content)}

    @_audited("guardrail_saida")
    def guardrail_saida(state: CaseState) -> dict:
        retrieved = [d["metadata"]["citation"] for d in state.get("docs", [])]
        excerpts = {d["metadata"]["citation"]: d["texto"] for d in state.get("docs", [])}
        out = check_output(state["resposta_modelo"], retrieved, excerpts=excerpts)
        AuditLogger(trace_id=state["trace_id"]).log("guardrail_saida", acao=out.acao, problemas=out.problemas,
                                                    fontes_recuperadas=retrieved, fontes_validas=out.fontes_validas)
        return {"saida": asdict(out)}

    @_audited("criar_rascunho")
    def criar_rascunho(state: CaseState) -> dict:
        pid = state.get("paciente_id")
        fontes = ", ".join(state["saida"]["fontes_validas"]) or "sem fonte"
        item = "Plano de conduta sugerido pelo assistente" if state["modo"] == "caso" else "Prescrição sugerida pelo assistente"
        rascunho = {"item": item, "indicacao": state.get("hipotese") or state["pergunta"], "fonte": fontes,
                    "texto": state["saida"]["texto"], "id": None}
        if pid:
            rascunho["id"] = get_repository().create_prescription_draft(pid, item, rascunho["indicacao"], fontes)
        return {"rascunho": rascunho}

    @_audited("validacao_humana")
    def validacao_humana(state: CaseState) -> dict:
        payload = {"paciente_id": state.get("paciente_id"), "rascunho": state["rascunho"],
                   "mensagem": "Conduta sugerida pelo assistente. Aprove ou rejeite informando seu identificador."}
        decision = interrupt(payload)
        while not str(decision.get("medico_id", "")).strip():
            decision = interrupt({**payload, "erro": "A validação exige a identificação do médico responsável."})
        decision = {"aprovado": bool(decision["aprovado"]), "medico_id": decision["medico_id"].strip(),
                    "observacao": decision.get("observacao", "")}
        if state["rascunho"].get("id") is not None:
            get_repository().review_prescription(state["rascunho"]["id"], decision["aprovado"],
                                                 decision["medico_id"], decision["observacao"])
        AuditLogger(trace_id=state["trace_id"]).log("validacao_humana", paciente_id=state.get("paciente_id"),
                                                    rascunho_id=state["rascunho"].get("id"), **decision)
        return {"decisao": decision}

    @_audited("responder")
    def responder(state: CaseState) -> dict:
        entrada = state["entrada"]
        if entrada["bloqueado"]:
            text = entrada["resposta"]
        else:
            text = state["saida"]["texto"]
            if decision := state.get("decisao"):
                status = "APROVADA" if decision["aprovado"] else "REJEITADA"
                text += f"\n\nValidação humana: conduta {status} por {decision['medico_id']}."
            elif state.get("rascunho"):
                text += "\n\nStatus: PENDENTE_VALIDACAO — aguardando o médico responsável."
        AuditLogger(trace_id=state["trace_id"]).log("resposta_final", paciente_id=state.get("paciente_id"),
                                                    bloqueado=entrada["bloqueado"], resposta=text)
        return {"resposta_final": text}

    def after_input(state: CaseState) -> str:
        return "responder" if state["entrada"]["bloqueado"] else "carregar_paciente"

    def after_output(state: CaseState) -> str:
        needs_validation = state["modo"] == "caso" or state["entrada"]["intencao"] == "prescricao"
        return "criar_rascunho" if needs_validation and state["saida"]["acao"] != "bloqueada" else "responder"

    graph = StateGraph(CaseState)
    for name, fn in [("guardrail_entrada", guardrail_entrada), ("carregar_paciente", carregar_paciente),
                     ("verificar_pendencias", verificar_pendencias), ("emitir_alertas", emitir_alertas),
                     ("buscar_protocolos", buscar_protocolos), ("gerar_resposta", gerar_resposta),
                     ("guardrail_saida", guardrail_saida), ("criar_rascunho", criar_rascunho),
                     ("validacao_humana", validacao_humana), ("responder", responder)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "guardrail_entrada")
    graph.add_conditional_edges("guardrail_entrada", after_input, ["responder", "carregar_paciente"])
    graph.add_edge("carregar_paciente", "verificar_pendencias")
    graph.add_edge("verificar_pendencias", "emitir_alertas")
    graph.add_edge("emitir_alertas", "buscar_protocolos")
    graph.add_edge("buscar_protocolos", "gerar_resposta")
    graph.add_edge("gerar_resposta", "guardrail_saida")
    graph.add_conditional_edges("guardrail_saida", after_output, ["criar_rascunho", "responder"])
    graph.add_edge("criar_rascunho", "validacao_humana")
    graph.add_edge("validacao_humana", "responder")
    graph.add_edge("responder", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# --- execução --------------------------------------------------------------------------

def start(graph, thread_id: str, modo: str, pergunta: str | None = None, paciente_id: str | None = None) -> dict:
    """Inicia um caso. Retorna o estado; se houver ``__interrupt__``, aguarda validação humana."""
    if modo not in ("pergunta", "caso"):
        raise ValueError("modo deve ser 'pergunta' ou 'caso'")
    if modo == "caso" and not paciente_id:
        raise ValueError("o modo caso exige paciente_id")
    inputs = {"trace_id": thread_id, "modo": modo, "pergunta": pergunta or CASE_QUESTION,
              "paciente_id": paciente_id, "trilha": []}
    return graph.invoke(inputs, {"configurable": {"thread_id": thread_id}})


def resume(graph, thread_id: str, aprovado: bool, medico_id: str, observacao: str = "") -> dict:
    decision = {"aprovado": aprovado, "medico_id": medico_id, "observacao": observacao}
    return graph.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})


def pending_interrupt(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__") or []
    return interrupts[0].value if interrupts else None

"""Linha de comando do assistente (``uv run assistente``), executando o grafo LangGraph.

Exemplos:
    uv run assistente --listar-pacientes
    uv run assistente "Qual a meta porta-balão no infarto com supra?"
    uv run assistente "Quais condutas faltam para este paciente?" --paciente P001
    uv run assistente --caso P001                      # fluxo automatizado com validação humana
    uv run assistente --validar <trace_id> --aprovar --medico MED-07
    uv run assistente ... --provider fake              # sem GPU, resposta determinística
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from assistente_medico import __version__
from assistente_medico.db.tools import get_repository
from assistente_medico.graph.workflow import build_graph, pending_interrupt, resume, start
from assistente_medico.llm.chat_models import get_chat_model

AVISO = "Uso acadêmico. Não é ferramenta clínica; toda conduta exige validação do médico responsável."
CHECKPOINTS = Path("data/checkpoints.sqlite")


def open_graph(provider: str | None):
    CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(CHECKPOINTS, check_same_thread=False))
    return build_graph(get_chat_model(provider), checkpointer=saver)


def print_result(result: dict, trace_id: str) -> None:
    if result.get("contexto_paciente"):
        print("— Contexto do prontuário —\n" + result["contexto_paciente"] + "\n")
    for alert in result.get("alertas_emitidos") or []:
        print(f"⚠ ALERTA {alert['gravidade']}: {alert['titulo']} — {alert['motivo']} [{alert['fonte']}]")
    if result.get("alertas_emitidos"):
        print()
    if (payload := pending_interrupt(result)) is not None:
        print("— Sugestão aguardando validação humana —\n" + payload["rascunho"]["texto"] + "\n")
        if payload.get("erro"):
            print(f"! {payload['erro']}")
        print(f"Status: PENDENTE_VALIDACAO | trace_id: {trace_id}")
        return
    print("— Resposta —\n" + result["resposta_final"] + "\n")
    saida = result.get("saida") or {}
    if saida:
        print(f"Guardrail de saída: {saida['acao']} {saida['problemas'] or ''}".rstrip())
    print(f"Caminho: {' → '.join(result.get('trilha', []))}")
    print(f"trace_id: {trace_id}")


def ask_validation(graph, trace_id: str, result: dict) -> dict:
    while pending_interrupt(result) is not None:
        choice = input("Aprovar a conduta? [a]provar / [r]ejeitar / [d]epois: ").strip().lower()
        if choice.startswith("d"):
            print(f"Validação adiada. Retome com: uv run assistente --validar {trace_id} --aprovar --medico <ID>")
            return result
        medico = input("Identificador do médico responsável: ")
        obs = input("Observação (opcional): ")
        result = resume(graph, trace_id, aprovado=choice.startswith("a"), medico_id=medico, observacao=obs)
        print_result(result, trace_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Assistente médico {__version__} — Tech Challenge Fase 3")
    parser.add_argument("pergunta", nargs="?", help="Pergunta clínica")
    parser.add_argument("--paciente", help="ID pseudonimizado do paciente (ex.: P001)")
    parser.add_argument("--caso", metavar="PACIENTE", help="Executa o fluxo automatizado do paciente")
    parser.add_argument("--validar", metavar="TRACE_ID", help="Retoma uma validação pendente")
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--aprovar", action="store_true")
    decision.add_argument("--rejeitar", action="store_true")
    parser.add_argument("--medico", help="Identificador do médico que valida")
    parser.add_argument("--observacao", default="")
    parser.add_argument("--provider", choices=["mlx", "mlx-base", "fake"], default=None,
                        help="mlx = Llama ajustado (padrão); mlx-base = sem LoRA; fake = offline")
    parser.add_argument("--nao-interativo", action="store_true", help="Não pergunta a validação no terminal")
    parser.add_argument("--listar-pacientes", action="store_true")
    args = parser.parse_args()

    if args.listar_pacientes:
        for p in get_repository().list_patients():
            print(f"{p['id']}  {p['idade']:>3} {p['sexo']}  {p['leito']:<6} {p['hipotese']} [{p['protocolo_id'] or '-'}]")
        return

    print(AVISO + "\n")
    graph = open_graph(args.provider)
    if args.validar:
        if not (args.aprovar or args.rejeitar) or not args.medico:
            parser.error("--validar exige --aprovar ou --rejeitar e --medico")
        result = resume(graph, args.validar, aprovado=args.aprovar, medico_id=args.medico, observacao=args.observacao)
        print_result(result, args.validar)
        return

    trace_id = uuid.uuid4().hex
    if args.caso:
        result = start(graph, trace_id, "caso", paciente_id=args.caso)
    elif args.pergunta:
        result = start(graph, trace_id, "pergunta", args.pergunta, args.paciente)
    else:
        parser.error("informe uma pergunta, --caso, --validar ou --listar-pacientes")
    print_result(result, trace_id)
    if not args.nao_interativo:
        ask_validation(graph, trace_id, result)

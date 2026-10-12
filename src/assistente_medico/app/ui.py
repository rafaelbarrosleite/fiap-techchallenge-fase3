"""Interface de demonstração em Streamlit (Missão 7).

    uv run streamlit run src/assistente_medico/app/ui.py

Executa o mesmo grafo LangGraph da linha de comando, com o mesmo checkpointer SQLite
(uma validação iniciada aqui pode ser concluída pela CLI e vice-versa).
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from assistente_medico.db import tools
from assistente_medico.db.seed import build
from assistente_medico.graph.workflow import build_graph, pending_interrupt, resume, start
from assistente_medico.llm.chat_models import get_chat_model

CHECKPOINTS = Path("data/checkpoints.sqlite")
AUDIT_LOG = Path(os.getenv("AUDIT_LOG_DIR", "logs")) / "audit.jsonl"
EVAL_DIR = Path("artifacts/evaluation")
FINETUNE_DIR = Path("artifacts/finetune/official")
PROVIDERS = {"Llama 3.2 3B ajustado (MLX)": "mlx", "Llama 3.2 3B original (MLX)": "mlx-base", "Fake (offline)": "fake"}

st.set_page_config(page_title="Assistente Médico — Fase 3", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Carregando o modelo...")
def get_graph(provider: str):
    CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(CHECKPOINTS, check_same_thread=False))
    return build_graph(get_chat_model(provider), checkpointer=saver)


def repo():
    return tools.get_repository()


# --- componentes ------------------------------------------------------------------------

def show_alerts(alerts: list[dict]) -> None:
    for alert in alerts:
        text = f"**{alert['titulo']}** — {alert['motivo']}  \n{alert['acao']} _[{alert['fonte']}]_"
        (st.error if alert["gravidade"] == "CRITICO" else st.warning)(text, icon="🚨" if alert["gravidade"] == "CRITICO" else "⚠️")


def show_sources(result: dict) -> None:
    docs = result.get("docs") or []
    if not docs:
        return
    with st.expander(f"Fontes consultadas ({len(docs)})"):
        for d in docs:
            meta = d["metadata"]
            st.markdown(f"**{meta['citation']}** — {meta['protocol_title']} › {meta['section_title']} "
                        f"(v{meta['protocol_version']}, score {meta.get('score', '-')})")
            st.caption(d["texto"])


def show_result(result: dict, trace_id: str) -> None:
    if result.get("alertas_emitidos"):
        st.subheader("Alertas emitidos")
        show_alerts(result["alertas_emitidos"])
    entrada = result.get("entrada") or {}
    if entrada.get("bloqueado"):
        st.error(f"Pedido recusado pelo guardrail de entrada — categoria: **{entrada['categoria']}**", icon="⛔")
    payload = pending_interrupt(result)
    st.subheader("Sugestão aguardando validação" if payload else "Resposta")
    text = payload["rascunho"]["texto"] if payload else result.get("resposta_final", "")
    with st.container(border=True):
        st.markdown(text.replace("\n", "  \n"))
    saida = result.get("saida")
    if saida:
        badge = {"aprovada": "🟢", "corrigida": "🟡", "bloqueada": "🔴"}[saida["acao"]]
        st.caption(f"Guardrail de saída: {badge} {saida['acao']} {', '.join(saida['problemas']) or ''}")
    show_sources(result)
    st.caption(f"Caminho no grafo: {' → '.join(result.get('trilha', []))}")
    st.caption(f"trace_id: `{trace_id}`")


def validation_form(graph, key: str) -> None:
    pending = st.session_state.get(key)
    if not pending or pending_interrupt(pending["result"]) is None:
        return
    payload = pending_interrupt(pending["result"])
    st.subheader("Validação humana (obrigatória)")
    if payload.get("erro"):
        st.error(payload["erro"])
    st.info(f"Rascunho #{payload['rascunho'].get('id')} — {payload['rascunho']['item']} | "
            f"indicação: {payload['rascunho']['indicacao']} | fontes: {payload['rascunho']['fonte']}", icon="🩺")
    with st.form(f"form-{key}"):
        medico = st.text_input("Identificador do médico responsável", value=st.session_state.get("medico_id", ""))
        obs = st.text_area("Observação", height=70)
        col1, col2 = st.columns(2)
        approve = col1.form_submit_button("✅ Aprovar", width="stretch")
        reject = col2.form_submit_button("❌ Rejeitar", width="stretch")
    if approve or reject:
        st.session_state["medico_id"] = medico
        with st.spinner("Registrando a decisão..."):
            result = resume(graph, pending["trace_id"], aprovado=approve, medico_id=medico, observacao=obs)
        st.session_state[key] = {"trace_id": pending["trace_id"], "result": result}
        st.rerun()


# --- barra lateral ----------------------------------------------------------------------

with st.sidebar:
    st.title("🩺 Assistente Médico")
    st.caption("Tech Challenge Fase 3 — Llama 3.2 3B + LoRA, LangChain e LangGraph")
    provider = PROVIDERS[st.selectbox("Modelo", list(PROVIDERS))]
    st.session_state["medico_id"] = st.text_input("Médico (para validações)", value=st.session_state.get("medico_id", "MED-07"))
    if st.button("Reiniciar prontuário de demonstração", help="Recria o banco sintético com horários relativos a agora"):
        build()
        tools.get_repository.cache_clear()
        for key in ("pergunta", "caso"):
            st.session_state.pop(key, None)
        st.success("Prontuário recriado.")
    st.divider()
    st.warning("Uso acadêmico. Não é ferramenta clínica; toda conduta exige validação do médico responsável.", icon="⚠️")

graph = get_graph(provider)
patients = repo().list_patients()
labels = {p["id"]: f"{p['id']} — {p['idade']}{p['sexo']}, {p['leito']}: {p['hipotese']}" for p in patients}

tab_q, tab_case, tab_panel, tab_audit, tab_eval = st.tabs(
    ["💬 Pergunta clínica", "🏥 Fluxo do paciente", "📋 Painel", "🔎 Auditoria", "📈 Modelo e avaliação"])

# --- pergunta ---------------------------------------------------------------------------

with tab_q:
    with st.form("pergunta-form"):
        question = st.text_area("Pergunta do médico", placeholder="Ex.: Qual a meta porta-balão no infarto com supra?")
        pid = st.selectbox("Paciente (opcional)", [None, *labels], format_func=lambda x: "— sem paciente —" if x is None else labels[x])
        submitted = st.form_submit_button("Perguntar", type="primary")
    if submitted and question.strip():
        trace_id = uuid.uuid4().hex
        with st.spinner("Executando o fluxo..."):
            st.session_state["pergunta"] = {"trace_id": trace_id, "result": start(graph, trace_id, "pergunta", question, pid)}
    if current := st.session_state.get("pergunta"):
        show_result(current["result"], current["trace_id"])
        validation_form(graph, "pergunta")

# --- caso -------------------------------------------------------------------------------

with tab_case:
    st.markdown("Ao receber um paciente, o fluxo verifica exames pendentes e valores críticos, emite alertas, "
                "sugere condutas do protocolo aberto e **para até um médico validar**.")
    pid = st.selectbox("Paciente", list(labels), format_func=labels.get, key="case-patient")
    patient = repo().get_patient(pid)
    with st.expander("Prontuário", expanded=False):
        st.json({k: v for k, v in patient.items() if k not in ("atendimento", "sinais_vitais")})
        st.write("Atendimento:", patient["atendimento"])
        st.write("Sinais vitais:", patient["sinais_vitais"])
        st.dataframe(pd.DataFrame(repo().list_exams(pid))[["descricao", "prioridade", "solicitado_em", "realizado_em",
                                                            "valor", "unidade", "status"]], hide_index=True)
    if st.button("▶ Executar fluxo do paciente", type="primary"):
        trace_id = uuid.uuid4().hex
        with st.spinner("Verificando pendências, consultando protocolos e gerando sugestão..."):
            st.session_state["caso"] = {"trace_id": trace_id, "result": start(graph, trace_id, "caso", paciente_id=pid)}
    if current := st.session_state.get("caso"):
        show_result(current["result"], current["trace_id"])
        validation_form(graph, "caso")

# --- painel -----------------------------------------------------------------------------

with tab_panel:
    st.subheader("Alertas")
    alerts = pd.DataFrame(repo().list_alerts())
    if alerts.empty:
        st.info("Nenhum alerta emitido.")
    else:
        st.dataframe(alerts[["criado_em", "paciente_id", "gravidade", "titulo", "fonte", "status"]], hide_index=True,
                     width="stretch")
    st.subheader("Prescrições e condutas")
    rows = [r for p in patients for r in repo().list_prescriptions(p["id"])]
    if rows:
        st.dataframe(pd.DataFrame(rows)[["paciente_id", "item", "status", "criado_por", "validado_por", "fonte"]],
                     hide_index=True, width="stretch")

# --- auditoria --------------------------------------------------------------------------

with tab_audit:
    if not AUDIT_LOG.exists():
        st.info("Nenhum evento registrado ainda.")
    else:
        events = [json.loads(line) for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
        traces = list(dict.fromkeys(e["trace_id"] for e in reversed(events) if e["event"] == "node_start"))
        default = [t for t in (st.session_state.get(k, {}).get("trace_id") for k in ("caso", "pergunta")) if t in traces]
        trace = st.selectbox("trace_id", traces, index=traces.index(default[0]) if default else 0)
        selected = [e for e in events if e["trace_id"] == trace]
        st.dataframe(pd.DataFrame([{"horário": e["ts"][11:19], "evento": e["event"], "nó": e.get("node", ""),
                                    "detalhes": json.dumps({k: v for k, v in e.items()
                                                            if k not in ("ts", "trace_id", "event", "node")},
                                                           ensure_ascii=False)[:300]} for e in selected]),
                     hide_index=True, width="stretch")
        with st.expander("JSON bruto"):
            st.json(selected)

# --- avaliação --------------------------------------------------------------------------

with tab_eval:
    if (FINETUNE_DIR / "loss_curve.png").exists():
        st.subheader("Fine-tuning (QLoRA, MLX)")
        manifest = json.loads((FINETUNE_DIR / "manifest.json").read_text(encoding="utf-8"))
        st.image(str(FINETUNE_DIR / "loss_curve.png"), width=700)
        st.caption(f"{manifest['duration_minutes']:.0f} min · loss de validação "
                   f"{manifest['summary']['first_val_loss']:.2f} → {manifest['summary']['last_val_loss']:.2f} · "
                   f"pico de memória {manifest['summary']['peak_mem_gb']:.1f} GB")
    for title, name in [("Base × ajustado: PubMedQA e hospital (sem RAG)", "comparison.md"),
                        ("Hospital com RAG (como o assistente funciona)", "comparison_base_finetuned_base_rag_finetuned_rag.md")]:
        if (EVAL_DIR / name).exists():
            st.subheader(title)
            st.markdown((EVAL_DIR / name).read_text(encoding="utf-8"))
    safety = sorted(EVAL_DIR.glob("safety/*/summary.json"))
    if safety:
        st.subheader("Bateria de segurança")
        st.caption("v1 = regras antes da correção (teste independente); mlx/fake = regras atuais.")
        st.dataframe(pd.DataFrame([{"execução": str(p.parent.relative_to(EVAL_DIR / "safety")),
                                    **{k: v for k, v in json.loads(p.read_text()).items()
                                       if isinstance(v, (int, float, str))}} for p in safety]),
                     hide_index=True, width="stretch")

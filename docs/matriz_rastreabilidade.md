# Matriz de rastreabilidade

Fonte: [`references/8IADT - Fase 3 - Tech challenge.pdf`](<references/8IADT - Fase 3 - Tech challenge.pdf>). Cada requisito aponta para a implementação, o teste automatizado e a evidência gerada.

## Requisitos técnicos

| ID | Requisito (PDF) | Implementação | Teste | Evidência | Status |
|---|---|---|---|---|---|
| R1 | Fine-tuning de LLM (LLaMA, Falcon ou outra) | QLoRA do Llama 3.2 3B Instruct com MLX-LM — `finetune/train.py`, `configs/finetune.yaml` | `test_finetune_train.py` | `artifacts/finetune/official/` (manifesto, métricas, curva); adaptador na Release v0.7.0 | ✅ |
| R1.1 | … com protocolos médicos do hospital | 13 protocolos sintéticos com seções citáveis — `data/synthetic/protocolos/` | `test_data_pipeline.py` (validação de fontes) | Exemplos `faq`/`faq_rag` no treino | ✅ |
| R1.2 | … com perguntas frequentes de médicos | FAQ por protocolo (2–3 variações) + PubMedQA | `test_data_pipeline.py` | `data_report.json` | ✅ |
| R1.3 | … com modelos de laudos, receitas e procedimentos | `data/synthetic/modelos_documentos.yaml` | `test_toda_receita_gerada_e_rascunho_pendente_de_validacao` | Exemplos `documento` | ✅ |
| R1.4 | Preprocessing, anonimização e curadoria | `data/anonymize.py`, `data/curate.py`, `data/prepare.py` | `test_anonymize.py`, `test_data_pipeline.py` | `data_report.json`: 30 PII de cada tipo anonimizadas, 0 residual | ✅ |
| R2.1 | Pipeline LangChain integrando a LLM customizada | `llm/chat_models.py` (`BaseChatModel` MLX), `llm/assistant_chain.py` | `test_assistant_chain.py` | Relatório §5.1 | ✅ |
| R2.2 | Consultas em base estruturada (prontuários) | SQLite + tools `@tool` — `db/`; fatos do prontuário respondidos sem LLM (`responder_prontuario`) | `test_hospital_db.py`, `test_pergunta_sobre_prontuario_e_respondida_pelas_regras_sem_o_modelo` | `uv run build-hospital-db` | ✅ |
| R2.3 | Contextualizar com informações atualizadas do paciente | `format_patient_context` + nós `carregar_paciente`/`verificar_pendencias` | `test_pergunta_contextualizada_com_prontuario` | Exemplo P004 (K = 3,0) no relatório §6.6 | ✅ |
| R3.1 | Limites de atuação (nunca prescrever sem validação humana) | Guardrail de entrada + rascunho `PENDENTE_VALIDACAO` + `interrupt()` — `safety/`, `graph/workflow.py` | `test_guardrails.py`, `test_graph_workflow.py` | Bateria de segurança: v1 20/21 contidos, 0/15 falso bloqueio | ✅ |
| R3.2 | Logging detalhado para rastreamento e auditoria | `audit/logger.py`; eventos por nó com `trace_id` | `test_audit_logger.py`, `test_pergunta_simples_responde_com_fonte_sem_validacao` | `logs/audit.jsonl`; aba Auditoria | ✅ |
| R3.3 | Explainability (indicar a fonte) | RAG por seção + guardrail de citação e de aderência + "Fontes consultadas" | `test_guardrails.py`, `test_rag_retriever.py` | Fonte inventada 2,3% (ajustado + RAG); aderência barra 8/10 respostas com seção errada | ✅ |
| R4.1 | Projeto modularizado em Python | `src/assistente_medico/` com `uv`, `pyproject.toml`, `uv.lock` | 92 testes | `uv run pytest` | ✅ |
| R4.2 | Instruções completas no README | `README.md` | — | — | ✅ |

## Entregáveis

| ID | Entregável (PDF) | Onde | Status |
|---|---|---|---|
| E1 | Código: pipeline de fine-tuning | `src/assistente_medico/data/`, `finetune/` | ✅ |
| E2 | Código: integração com LangChain | `src/assistente_medico/llm/`, `db/tools.py`, `rag/` | ✅ |
| E3 | Código: fluxos do LangGraph | `src/assistente_medico/graph/workflow.py` | ✅ |
| E4 | Dataset anonimizado ou sintético | `data/synthetic/` (versionado) + `uv run prepare-data` | ✅ |
| E5 | Relatório: processo de fine-tuning | [`relatorio_tecnico.md`](relatorio_tecnico.md) §3–4 | ✅ |
| E6 | Relatório: descrição do assistente | [`relatorio_tecnico.md`](relatorio_tecnico.md) §5 | ✅ |
| E7 | Relatório: diagrama do fluxo LangChain | [`diagrama_fluxo.md`](diagrama_fluxo.md) e relatório §2 e §5.2 | ✅ |
| E8 | Relatório: avaliação e análise | [`relatorio_tecnico.md`](relatorio_tecnico.md) §6–8 | ✅ |
| E9 | Vídeo de até 15 min (treino, fluxo automatizado, pergunta contextualizada, logs e validação) | Roteiro em [`roteiro_video.md`](roteiro_video.md); link no README | ⏳ Missão 9 |

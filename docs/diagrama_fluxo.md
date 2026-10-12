# Diagrama do fluxo LangChain + LangGraph

O LangChain fornece as peças (modelo de chat, retriever, ferramentas de prontuário). O LangGraph define a ordem, as decisões e a pausa para validação humana. Código: [`graph/workflow.py`](../src/assistente_medico/graph/workflow.py).

## Visão funcional

```mermaid
flowchart TD
    IN[/"Médico: pergunta<br/>ou paciente (modo caso)"/] --> GE{{"guardrail_entrada<br/>anonimiza + classifica pedido"}}
    GE -- "fora do escopo<br/>(ignorar regras, dose letal,<br/>dados pessoais, prescrição sem validação...)" --> RESP
    GE -- permitido --> CP["carregar_paciente<br/>tools: buscar_paciente,<br/>listar_exames_pendentes,<br/>listar_resultados_criticos"]
    CP --> VP["verificar_pendencias<br/>regras PRT-012 §1 e §2"]
    VP --> AL["emitir_alertas<br/>(modo caso; sem duplicar)"]
    AL --> BP["buscar_protocolos<br/>BM25 por seção → citações"]
    BP --> GR["gerar_resposta<br/>Llama 3.2 3B + LoRA (MLX)"]
    GR --> GS{{"guardrail_saida<br/>citação ⊆ trechos recuperados?<br/>fonte existe? PII? aviso?"}}
    GS -- "pergunta informativa" --> RESP
    GS -- "modo caso ou<br/>pedido de prescrição" --> RA["criar_rascunho<br/>status PENDENTE_VALIDACAO"]
    RA --> HV[["validacao_humana<br/>interrupt() — aguarda médico<br/>aprovar/rejeitar com ID"]]
    HV --> RESP["responder<br/>texto final + fontes + status"]
    RESP --> OUT[/"Resposta auditada"/]

    LOG[("logs/audit.jsonl<br/>trace_id por caso")]
    DB[("data/hospital.db<br/>prontuário pseudonimizado")]
    CP -.-> DB
    AL -.-> DB
    RA -.-> DB
    HV -.-> DB
    GE & CP & VP & AL & BP & GR & GS & RA & HV & RESP -.-> LOG
```

## Grafo compilado (gerado pelo LangGraph)

Gerado com `build_graph().get_graph().draw_mermaid()`. Arestas tracejadas são condicionais.

```mermaid
graph TD;
	__start__([__start__]):::first
	guardrail_entrada(guardrail_entrada)
	carregar_paciente(carregar_paciente)
	verificar_pendencias(verificar_pendencias)
	emitir_alertas(emitir_alertas)
	buscar_protocolos(buscar_protocolos)
	gerar_resposta(gerar_resposta)
	guardrail_saida(guardrail_saida)
	criar_rascunho(criar_rascunho)
	validacao_humana(validacao_humana)
	responder(responder)
	__end__([__end__]):::last
	__start__ --> guardrail_entrada;
	buscar_protocolos --> gerar_resposta;
	carregar_paciente --> verificar_pendencias;
	criar_rascunho --> validacao_humana;
	emitir_alertas --> buscar_protocolos;
	gerar_resposta --> guardrail_saida;
	guardrail_entrada -.-> carregar_paciente;
	guardrail_entrada -.-> responder;
	guardrail_saida -.-> criar_rascunho;
	guardrail_saida -.-> responder;
	validacao_humana --> responder;
	verificar_pendencias --> emitir_alertas;
	responder --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Decisões

| Ponto | Decisão | Motivo |
|---|---|---|
| Escolha de ferramentas | Orquestração fixa no grafo, não pela LLM | O modelo de 3B não é confiável para decidir sozinho quais ferramentas chamar; o fluxo determinístico é auditável |
| Validação humana | `interrupt()` + checkpointer | O grafo para, persiste o estado e só continua com a decisão de um médico identificado |
| Citações | Guardrail compara a citação do modelo com os trechos recuperados | A avaliação (M3) mostrou que o modelo ajustado erra a seção citada; a lista de fontes exibida vem sempre do sistema |
| Alertas | Gerados por regra (PRT-012), não pela LLM | Pendências e valores críticos são fatos do prontuário; não dependem de geração de texto |
| Auditoria | `node_start`/`node_end` de cada nó + eventos de decisão, com `trace_id` | Reconstrói o caminho completo de cada caso. Ao retomar após o `interrupt`, o nó de validação é reexecutado, por isso aparece com dois `node_start` |

# Tech Challenge — Fase 3 — Assistente Médico com LLaMA, LangChain e LangGraph

Projeto acadêmico da Pós Tech FIAP (8IADT). Assistente virtual médico que usa uma LLaMA ajustada com LoRA no PubMedQA e em dados sintéticos do hospital, consulta prontuários estruturados via LangChain e coordena fluxos de decisão com LangGraph, sempre com validação humana antes de qualquer prescrição.

> Uso exclusivamente acadêmico. Não é ferramenta clínica e não substitui avaliação médica.

Estado: **Missões 0–7 concluídas** (dados, fine-tuning, avaliação, prontuário, RAG, LangChain, fluxo LangGraph com validação humana e interface). Pendentes: relatório final e vídeo. Plano completo em [`docs/plano_implementacao.md`](docs/plano_implementacao.md).

## Instalação

Requisitos: Python 3.11–3.13 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
uv run pytest
```

## Dados (Missão 1)

```bash
uv run prepare-data
```

O comando baixa o [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (licença MIT, sem token), gera o corpus sintético do hospital, anonimiza, faz a curadoria e grava:

- `data/processed/{train,valid,test}.jsonl`: exemplos com metadados (fonte, tarefa, grupo);
- `data/processed/mlx/{train,valid,test}.jsonl`: somente `messages`, no formato de chat do MLX-LM;
- `artifacts/data/data_report.json`: contagens, PII anonimizada, itens removidos e hashes SHA-256.

| Fonte | Uso |
|---|---|
| `pqa_artificial` (211 mil, rótulo automático, ~93% "yes") | Amostra balanceada yes/no: 4.000 treino + 200 validação |
| `pqa_labeled` (1.000, rotulado por especialistas) | 500 teste, 250 validação e 250 treino (estratificado por decisão) |
| Corpus do hospital (`data/synthetic/`) | 13 protocolos fictícios, FAQ, modelos de laudo/receita/alerta, notas clínicas anonimizadas e exemplos de segurança |

Decisões de dados:

- **Idioma**: o PubMedQA fica em inglês, e o corpus do hospital em português. O assistente responde em português.
- **Divisão sem vazamento**: variações da mesma pergunta ficam no mesmo split, e há uma checagem de sobreposição entre treino e teste.
- **Anonimização** ([`anonymize.py`](src/assistente_medico/data/anonymize.py)): usa regras determinísticas para CPF, CNS, RG, telefone, e-mail, CEP, datas, campos rotulados, profissionais e prenomes. As notas clínicas são geradas com dados pessoais falsos (Faker) e passam pelo anonimizador antes de entrar no treino.
- **Curadoria**: normalização, deduplicação e limite de tamanho. Respostas clínicas sem fonte, prescrições sem marca de validação e qualquer PII residual são descartadas.

Os protocolos são fictícios, inspirados em diretrizes públicas, e servem só para fins acadêmicos.

## Fine-tuning (Missão 2)

```bash
uv run finetune --smoke      # 30 iterações para validar memória e velocidade
uv run finetune --run-name official
```

QLoRA (rank 16, 16 camadas) sobre `mlx-community/Llama-3.2-3B-Instruct-4bit` com MLX-LM em um MacBook M4 de 16 GB: 1.000 iterações em 3 h 43 min, loss de validação de 2,97 para 1,58. Os artefatos ficam em `artifacts/finetune/official/`: `metrics.jsonl`, `loss_curve.png`, `manifest.json` e os adaptadores.

## Avaliação (Missão 3)

```bash
uv run evaluate-model --label base
uv run evaluate-model --label finetuned --adapter artifacts/finetune/official/adapters
uv run evaluate-model --label finetuned_rag --rag --adapter artifacts/finetune/official/adapters
uv run evaluate-model --compare base finetuned base_rag finetuned_rag
```

Resultados em `artifacts/evaluation/`. O PubMedQA subiu de 73,0% para 75,6% (McNemar p = 0,07, não significativo). No hospital, com RAG, o modelo ajustado reduz fonte inventada de 21% para 2% e passa a gerar 100% das receitas como rascunho pendente de validação.

## Assistente e fluxo LangGraph (Missões 4–6)

```bash
uv run build-hospital-db                                   # prontuário sintético pseudonimizado
uv run assistente --listar-pacientes
uv run assistente "Qual a meta porta-balão no infarto com supra?"
uv run assistente "Esta paciente pode iniciar insulina agora?" --paciente P004
uv run assistente --caso P001                              # alertas + sugestão + validação humana
uv run assistente --validar <trace_id> --aprovar --medico MED-07
uv run evaluate-safety --provider mlx                      # bateria adversarial
```

Use `--provider fake` para rodar sem GPU. O diagrama e as decisões do fluxo estão em [`docs/diagrama_fluxo.md`](docs/diagrama_fluxo.md). Cada caso deixa a trilha completa em `logs/audit.jsonl`: nós percorridos, decisões dos guardrails, resposta do modelo e médico validador, todos com o mesmo `trace_id`.

## Interface de demonstração (Missão 7)

```bash
uv run streamlit run src/assistente_medico/app/ui.py
```

Abas:

- **Pergunta clínica**: pergunta com ou sem paciente, com resposta, fontes, guardrail e caminho no grafo;
- **Fluxo do paciente**: alertas, sugestão de conduta e formulário de validação humana;
- **Painel**: alertas e prescrições gravados no prontuário;
- **Auditoria**: eventos de cada `trace_id`;
- **Modelo e avaliação**: curva de loss, comparativos e bateria de segurança.

A barra lateral escolhe o modelo (ajustado, original ou fake) e recria o prontuário de demonstração, com horários relativos ao momento atual. A interface usa o mesmo checkpointer da CLI: uma validação iniciada em uma pode ser concluída na outra.

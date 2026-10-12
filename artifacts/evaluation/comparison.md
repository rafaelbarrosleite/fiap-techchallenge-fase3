| Métrica | base | finetuned |
|---|---:|---:|
| PubMedQA — acurácia | 73.0% | 75.6% |
| PubMedQA — F1 macro | 52.0% | 52.8% |
| PubMedQA — resposta inválida (↓) | 0.0% | 0.0% |
| PubMedQA — recall 'yes' | 87.3% | 86.6% |
| PubMedQA — recall 'no' | 72.8% | 82.2% |
| PubMedQA — recall 'maybe' | 1.8% | 0.0% |
| Hospital — Cita fonte | 59.1% | 93.2% |
| Hospital — Protocolo citado correto | 22.0% | 61.0% |
| Hospital — Seção citada correta | 22.0% | 34.1% |
| Hospital — Fonte inventada (↓) | 38.6% | 11.4% |
| Hospital — Recusa indevida (↓) | 7.1% | 4.8% |
| Hospital — Recusa adequada (segurança) | 100.0% | 100.0% |
| Hospital — Receita/solicitação exige validação | 0.0% | 100.0% |
| Hospital — Responde em português | 100.0% | 100.0% |
| Hospital — ROUGE-L vs referência | 0.167 | 0.571 |
| Latência média PubMedQA (s) | 1.24 | 1.41 |
| Latência média hospital (s) | 5.08 | 2.70 |

McNemar exato (base × finetuned, PubMedQA): só base acertou = 16, só finetuned acertou = 29, p = 0.0725.

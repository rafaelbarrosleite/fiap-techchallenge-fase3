"""Corpus sintético do hospital: protocolos, FAQ, modelos de documentos e segurança.

Os YAML em ``data/synthetic/`` são a fonte única. Este módulo:

- valida os protocolos (``Protocol``);
- gera os exemplos de chat para o fine-tuning, cada um com ``group`` — variações
  da mesma pergunta compartilham o grupo e caem no mesmo split (evita vazamento);
- gera notas clínicas brutas com dados pessoais falsos e as anonimiza, para que o
  pipeline de anonimização seja exercitado de verdade.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml
from faker import Faker
from pydantic import BaseModel, model_validator

from assistente_medico.data.anonymize import anonymize
from assistente_medico.llm.prompts import CLINICAL_DISCLAIMER, HOSPITAL_SYSTEM_PROMPT

SYNTHETIC_DIR = Path("data/synthetic")


class Section(BaseModel):
    id: str
    titulo: str
    texto: str


class FaqItem(BaseModel):
    perguntas: list[str]
    resposta: str
    fontes: list[str]


class Protocol(BaseModel):
    id: str
    titulo: str
    versao: str
    area: str
    secoes: list[Section]
    faq: list[FaqItem]

    @model_validator(mode="after")
    def _fontes_existem(self) -> "Protocol":
        ids = {s.id for s in self.secoes}
        for item in self.faq:
            missing = set(item.fontes) - ids
            if missing:
                raise ValueError(f"{self.id}: FAQ cita seções inexistentes {sorted(missing)}")
        return self

    @property
    def institucional(self) -> bool:
        return self.area.startswith("Institucional")

    def section(self, section_id: str) -> Section:
        return next(s for s in self.secoes if s.id == section_id)

    def cite(self, section_ids: list[str]) -> str:
        parts = [f"§{sid} ({self.section(sid).titulo})" for sid in section_ids]
        return f"Fonte: {self.id} — {self.titulo}, " + "; ".join(parts) + "."


def load_protocols(base_dir: Path = SYNTHETIC_DIR) -> list[Protocol]:
    files = sorted((base_dir / "protocolos").glob("*.yaml"))
    protocols = [Protocol.model_validate(yaml.safe_load(f.read_text(encoding="utf-8"))) for f in files]
    ids = [p.id for p in protocols]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs de protocolo duplicados")
    return protocols


def _chat(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": HOSPITAL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _example(task: str, group: str, user: str, assistant: str, **meta: Any) -> dict[str, Any]:
    return {"source": "hospital", "task": task, "group": group, "messages": _chat(user, assistant), **meta}


def build_faq_examples(protocols: list[Protocol]) -> list[dict[str, Any]]:
    """Perguntas frequentes em dois formatos: pergunta direta e pergunta com trecho recuperado (RAG)."""
    examples = []
    for proto in protocols:
        for i, item in enumerate(proto.faq):
            group = f"{proto.id}-faq{i}"
            answer = item.resposta.strip() + "\n\n" + proto.cite(item.fontes)
            if not proto.institucional:
                answer += f"\n{CLINICAL_DISCLAIMER}"
            context = "\n".join(
                f"[{proto.id} §{sid} — {proto.section(sid).titulo}] {proto.section(sid).texto.strip()}"
                for sid in item.fontes
            )
            for j, question in enumerate(item.perguntas):
                meta = {"protocol_id": proto.id, "sections": item.fontes}
                examples.append(_example("faq", group, question, answer, **meta))
                if j == 0:
                    user = f"Trechos dos protocolos internos:\n{context}\n\nPergunta: {question}"
                    examples.append(_example("faq_rag", group, user, answer, **meta))
    return examples


def _patient(rng: random.Random) -> dict[str, Any]:
    alergias = rng.choice(["nenhuma conhecida", "dipirona", "penicilina", "sulfa", "látex"])
    return {"paciente_id": f"P{rng.randint(1, 999):03d}", "idade": rng.randint(18, 92), "alergias": alergias}


def build_document_examples(base_dir: Path = SYNTHETIC_DIR, per_case: int = 3, seed: int = 42) -> list[dict[str, Any]]:
    """Exemplos de preenchimento dos modelos de receita, solicitação, laudo, alerta e POP."""
    spec = yaml.safe_load((base_dir / "modelos_documentos.yaml").read_text(encoding="utf-8"))
    templates = {m["id"]: m for m in spec["modelos"]}
    cases = spec["casos"]
    rng = random.Random(seed)
    examples = []

    requests = {
        "MOD-REC-01": ("receita", "Prepare um rascunho de prescrição de {medicamento} para o paciente {paciente_id} "
                                  "({idade} anos, alergias: {alergias}). Indicação: {indicacao}."),
        "MOD-SOL-01": ("solicitacao_exame", "Gere a solicitação de {exame} ({prioridade}) para o paciente "
                                            "{paciente_id}, {idade} anos. Motivo: {indicacao}."),
        "MOD-LAU-01": ("laudo_rx", "Redija o laudo preliminar da radiografia de tórax do paciente {paciente_id}. "
                                   "Achados descritos: {achados}"),
        "MOD-LAU-02": ("laudo_ecg", "Redija o laudo preliminar do ECG do paciente {paciente_id}: ritmo {ritmo}, "
                                    "FC {fc} bpm, {achados}"),
        "MOD-ALT-01": ("alerta", "Monte o alerta para a equipe: paciente {paciente_id}, leito {leito}, {motivo}."),
    }
    for template_id, (case_key, request) in requests.items():
        for c_idx, case in enumerate(cases[case_key]):
            for _ in range(per_case):
                fields = {**_patient(rng), "leito": rng.randint(1, 40), **case}
                user = request.format(**fields)
                assistant = templates[template_id]["texto"].format(**fields).strip()
                group = f"{template_id}-case{c_idx}"
                examples.append(_example("documento", group, user, assistant, template_id=template_id))

    pop = templates["MOD-POP-01"]["texto"].strip()
    for k, question in enumerate(["Qual o POP de coleta de hemoculturas?", "Como coletar hemocultura segundo o hospital?"]):
        examples.append(_example("documento", "MOD-POP-01", question, pop, template_id="MOD-POP-01"))
    return examples


def build_safety_examples(base_dir: Path = SYNTHETIC_DIR) -> list[dict[str, Any]]:
    spec = yaml.safe_load((base_dir / "seguranca.yaml").read_text(encoding="utf-8"))
    examples = []
    for item in spec["exemplos"]:
        for k, request in enumerate(item["pedidos"]):
            # Cada pedido é seu próprio grupo, para que todas as categorias apareçam no treino.
            group = f"seg-{item['categoria']}-{k}"
            examples.append(_example("seguranca", group, request, item["resposta"].strip(), categoria=item["categoria"]))
    return examples


_NOTE_SCENARIOS = [
    ("febre de 38,9 °C, FC 112, PA 84/50 mmHg e lactato 4,2 mmol/L, com foco urinário provável",
     "Suspeita de sepse com hipotensão e lactato ≥ 4 mmol/L.",
     "Abrir PRT-001: lactato e hemoculturas, antimicrobiano em até 1 hora e cristaloide 30 mL/kg; alerta crítico ao plantonista.",
     "PRT-001 §2 e §5"),
    ("dor torácica opressiva há 40 minutos, sudorese e ECG com supra de ST em parede inferior",
     "Síndrome coronariana aguda com supradesnivelamento de ST.",
     "Acionar hemodinâmica imediatamente (meta porta-balão ≤ 90 min) e alertar o cardiologista.",
     "PRT-002 §2 e §5"),
    ("hemiparesia direita e afasia iniciadas há 2 horas, glicemia capilar 132 mg/dL",
     "Déficit neurológico focal agudo dentro da janela terapêutica.",
     "Código AVC: tomografia de crânio em até 25 minutos e avaliação do neurologista para trombólise.",
     "PRT-003 §2 e §3"),
    ("glicemia 486 mg/dL, pH 7,12, bicarbonato 9 mEq/L, cetonúria positiva e potássio 3,0 mEq/L",
     "Cetoacidose diabética com potássio abaixo de 3,3 mEq/L.",
     "Hidratação com SF 0,9%; repor potássio e adiar a insulina até K ≥ 3,3 mEq/L.",
     "PRT-007 §2 e §3"),
    ("urticária generalizada, sibilos e PA 78/40 mmHg minutos após antibiótico endovenoso",
     "Quadro compatível com anafilaxia.",
     "Adrenalina intramuscular sem atraso, suspender o antibiótico, registrar alergia e observar por 6 horas.",
     "PRT-010 §2 e §4"),
    ("tosse produtiva, febre, confusão mental, ureia 62 mg/dL, FR 32 irpm, 71 anos",
     "Pneumonia comunitária com CURB-65 igual a 4.",
     "Internação com avaliação de UTI e antibioticoterapia conforme CCIH.",
     "PRT-004 §2 e §3"),
]


def build_clinical_note_examples(n_notes: int = 30, seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Notas de admissão com PII falsa → anonimização → tarefa de resumo para passagem de plantão."""
    fake = Faker("pt_BR")
    Faker.seed(seed)
    rng = random.Random(seed)
    examples, totals = [], {}
    for i in range(n_notes):
        quadro, impressao, conduta, fonte = _NOTE_SCENARIOS[i % len(_NOTE_SCENARIOS)]
        patient = _patient(rng)
        raw = (
            f"Paciente: {fake.name()}\nCPF: {fake.cpf()} | Telefone: {fake.cellphone_number()}\n"
            f"Endereço: {fake.street_address()}\n"
            f"Admitido em {fake.date(pattern='%d/%m/%Y')} com {quadro}. "
            f"Idade {patient['idade']} anos. Avaliado pela Dra. {fake.first_name_female()} {fake.last_name()}."
        )
        result = anonymize(raw)
        for key, value in result.counts.items():
            totals[key] = totals.get(key, 0) + value
        user = f"Resuma a nota de admissão abaixo para a passagem de plantão.\n\n{result.text}"
        assistant = (
            f"Resumo para passagem de plantão:\n- Quadro: {quadro}.\n- Impressão: {impressao}\n"
            f"- Conduta sugerida pelo protocolo: {conduta}\nFonte: {fonte}.\n{CLINICAL_DISCLAIMER}"
        )
        group = f"nota-cenario{i % len(_NOTE_SCENARIOS)}"
        examples.append(_example("nota_clinica", group, user, assistant, anonymized=True))
    return examples, totals


def build_hospital_examples(base_dir: Path = SYNTHETIC_DIR, seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, int]]:
    protocols = load_protocols(base_dir)
    notes, pii_counts = build_clinical_note_examples(seed=seed)
    examples = (
        build_faq_examples(protocols)
        + build_document_examples(base_dir, seed=seed)
        + build_safety_examples(base_dir)
        + notes
    )
    return examples, pii_counts

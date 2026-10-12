"""Guardrails determinísticos de entrada e saída (Missão 6).

Ficam fora do modelo de propósito: são regras auditáveis, testáveis e que não mudam
com o fine-tuning. Implementam a política PRT-000:

- entrada: pedidos fora do escopo são recusados com a resposta padronizada da política;
  pedidos de prescrição seguem, mas viram rascunho com validação humana obrigatória;
- saída: só valem citações que existem e que estão entre os trechos recuperados. Citação
  divergente indica conteúdo não confiável (na avaliação, o modelo trocava o conteúdo junto
  com a seção), então a resposta gerada é descartada e o médico recebe os trechos do protocolo
  na íntegra. O mesmo vale para resposta **não aderente** ao texto citado: pouca sobreposição
  de vocabulário (< 0,5) ou número que não aparece nem nos trechos nem no contexto do paciente.
  Resposta clínica sem fonte é bloqueada; dados pessoais são removidos; o aviso de validação é
  garantido.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from assistente_medico.data.anonymize import anonymize
from assistente_medico.data.hospital_corpus import SYNTHETIC_DIR, load_protocols
from assistente_medico.evaluation.metrics import cited_sections, cited_titles
from assistente_medico.llm.prompts import CLINICAL_DISCLAIMER
from assistente_medico.rag.retriever import stem_tokens

# --- entrada ----------------------------------------------------------------------------

# Ordem importa: a primeira categoria que casar define a recusa.
_INPUT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("ignorar_regras", re.compile(
        r"(?i)\b(ignor\w*|desconsider\w*|esque[çc]a)\b.{0,40}\b(instru[çc]\w*|regras?|protocolos?|pol[íi]tica)"
        r"|\bfinja (que|ser)\b|\bsem (regras|restri[çc]\w*)\b|\bmodo (desenvolvedor|livre|irrestrito)\b"
        r"|\bjailbreak\b|n[ãa]o precisa mais seguir")),
    ("dose_letal", re.compile(
        r"(?i)\b(letal|letais|mat(ar|aria|e|asse)|caus\w* (uma )?parada|overdose intencional|tirar a vida)\b")),
    ("dados_pessoais", re.compile(
        r"(?i)\b(cpf|telefone|celular|endere[çc]o|e-?mail|nome completo|\brg\b|cns)\b.{0,60}\bpaciente"
        r"|\bpaciente\b.{0,60}\b(cpf|telefone|celular|endere[çc]o|e-?mail|nome completo)\b")),
    ("ocultar_registro", re.compile(
        r"(?i)\b(apag\w*|delet\w*|remov\w*|omit\w*|ocult\w*|esconda)\b.{0,50}\b(prontu[áa]rio|registro|log|auditoria|erro)"
        r"|\bn[ãa]o (registre|grave|logue)\b")),
    ("orientacao_direta_paciente", re.compile(
        r"(?i)\bsou (o |a )?paciente\b|\b(eu )?(tomo|devo tomar|posso tomar)\b.{0,40}\b(em casa|hoje|rem[ée]dio)")),
    ("prescricao_direta", re.compile(
        r"(?i)\b(prescrev\w*|prescri[çc][ãa]o|receita|libere|emita)\b.{0,80}"
        r"(sem (esperar|validar|valida[çc][ãa]o|aprova[çc][ãa]o|o m[ée]dico|revis[ãa]o)|n[ãa]o precisa (de )?aprova"
        r"|direto no sistema|receita final|definitiv)")),
]
_PRESCRIPTION_REQUEST = re.compile(r"(?i)\b(prescrev\w*|prescri[çc][ãa]o|receita|rascunho de prescri)")
# Perguntas sobre fatos do prontuário: respondidas pelas regras do PRT-012, sem o modelo.
RECORD_PENDING = re.compile(r"(?i)\b(exames?\s+(est[ãa]o\s+|ainda\s+)?pendentes?|pend[êe]ncias?)\b")
RECORD_CRITICAL = re.compile(r"(?i)\b(resultados?|valor(es)?|exames?)\s+cr[íi]ticos?\b")


@dataclass
class InputCheck:
    bloqueado: bool
    categoria: str | None = None
    resposta: str | None = None
    intencao: str = "pergunta"           # pergunta | prescricao | consulta_prontuario
    pii_removida: dict[str, int] = field(default_factory=dict)
    texto: str = ""                      # pergunta já anonimizada


@lru_cache(maxsize=1)
def _refusals(base_dir: Path = SYNTHETIC_DIR) -> dict[str, str]:
    spec = yaml.safe_load((base_dir / "seguranca.yaml").read_text(encoding="utf-8"))
    return {item["categoria"]: " ".join(item["resposta"].split()) for item in spec["exemplos"]}


def check_input(question: str) -> InputCheck:
    anon = anonymize(question)
    for categoria, pattern in _INPUT_RULES:
        if pattern.search(question):
            return InputCheck(bloqueado=True, categoria=categoria, resposta=_refusals()[categoria],
                              pii_removida=dict(anon.counts), texto=anon.text)
    if _PRESCRIPTION_REQUEST.search(question):
        intencao = "prescricao"
    elif RECORD_PENDING.search(question) or RECORD_CRITICAL.search(question):
        intencao = "consulta_prontuario"
    else:
        intencao = "pergunta"
    return InputCheck(bloqueado=False, intencao=intencao, pii_removida=dict(anon.counts), texto=anon.text)


# --- saída -----------------------------------------------------------------------------

_SOURCE_LINE = re.compile(r"(?im)^\s*(fonte|fontes|refer[êe]ncia)s?\s*:.*$|\s*Fonte:\s*[^\n]*$")
_REFUSAL = re.compile(r"(?i)\b(n[ãa]o posso|n[ãa]o forne[çc]o|n[ãa]o encontrei protocolo|n[ãa]o oriento)\b")
UNVERIFIED_HEADER = ("A resposta gerada não pôde ser validada contra os protocolos consultados (citação ou conteúdo "
                     "divergente) e foi descartada. Trechos dos protocolos para consulta direta:")
NO_SOURCE_ANSWER = ("Não encontrei protocolo interno que cubra esse tema, então não posso responder com segurança. "
                    "Recomendo consultar o especialista responsável ou a diretriz da área. Fonte: PRT-000 §4.")


# Aderência ao texto citado. Calibração (docs/relatorio_tecnico.md §6.5): as respostas de
# referência do FAQ têm suporte entre 0,62 e 1,0 contra as próprias seções. Com 0,6, respostas
# corretas do modelo com paráfrase (0,56) eram bloqueadas; com 0,5 elas passam, e as erradas entre
# 0,5 e 0,6 observadas na avaliação continuam barradas pela checagem de números.
SUPPORT_THRESHOLD = 0.5
_CITATION_TEXT = re.compile(
    r"(?i)\b(fontes?|refer[êe]ncias?)\s*:[^\n]*|Conduta sujeita à validação[^\n]*"
    r"|PRT-\d{3}(\s*(—|-)[^,§\n]*)?(,?\s*§\s*\d+(\s*\([^)]*\))?(\s*(e|;|,)\s*)?)*")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")


def _answer_body(text: str) -> str:
    return _LIST_MARKER.sub(" ", _CITATION_TEXT.sub(" ", text))


def _numbers(text: str) -> set[str]:
    return {n.replace(".", ",") for n in _NUMBER.findall(text)}


def grounding(answer: str, reference: str, numbers_reference: str | None = None) -> tuple[float, list[str]]:
    """(fração das palavras de conteúdo da resposta presentes em ``reference``, números sem suporte).

    Os números são conferidos contra ``numbers_reference`` (padrão: ``reference``): tudo o que o
    modelo recebeu pode legitimamente aparecer, mesmo que de uma seção que ele não citou.
    """
    body = _answer_body(answer)
    words = {t for t in stem_tokens(body) if not t[0].isdigit() and len(t) > 2}
    ref = set(stem_tokens(reference))
    support = len(words & ref) / len(words) if words else 1.0
    return support, sorted(_numbers(body) - _numbers(numbers_reference or reference))


@dataclass
class OutputCheck:
    acao: str                            # aprovada | corrigida | bloqueada
    texto: str
    problemas: list[str] = field(default_factory=list)
    fontes_validas: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict]:
    return {p.id: {"titulo": p.titulo, "secoes": {s.id: s.titulo for s in p.secoes}} for p in load_protocols()}


def _parse_citation(citation: str) -> tuple[str, str]:
    pid, sec = citation.split(" §")
    return pid, sec


def format_sources(citations: list[str]) -> str:
    catalog = _catalog()
    parts = []
    for citation in citations:
        pid, sec = _parse_citation(citation)
        parts.append(f"{citation} ({catalog[pid]['titulo']} — {catalog[pid]['secoes'][sec]})")
    return "Fontes consultadas: " + "; ".join(parts) + "."


def check_output(answer: str, retrieved: list[str], clinical: bool = True,
                 excerpts: dict[str, str] | None = None, context: str = "") -> OutputCheck:
    """Valida a resposta do modelo contra os trechos realmente entregues a ele (``retrieved``).

    ``excerpts`` (citação → texto) habilita a checagem de aderência e o fallback extrativo;
    ``context`` (pergunta + prontuário) é a outra fonte legítima de fatos e números.
    """
    problems: list[str] = []
    catalog = _catalog()

    anon = anonymize(answer)
    text = anon.text
    if anon.total:
        problems.append("pii_na_saida")

    is_refusal = bool(_REFUSAL.search(text))
    if not retrieved and not is_refusal:
        return OutputCheck(acao="bloqueada", texto=NO_SOURCE_ANSWER, problemas=problems + ["sem_fonte_recuperada"])

    cited = cited_sections(text)
    allowed = {_parse_citation(c) for c in retrieved} | {("PRT-000", s) for s in catalog["PRT-000"]["secoes"]}
    invented = {f"{p} §{s}" for p, s in cited if p not in catalog or s not in catalog[p]["secoes"]}
    invented |= {p for p, title in cited_titles(text).items() if p in catalog and title != catalog[p]["titulo"]}
    not_retrieved = {f"{p} §{s}" for p, s in cited if (p, s) not in allowed} - invented
    if invented:
        problems.append("citacao_inventada")
    if not_retrieved:
        problems.append("citacao_nao_recuperada")
    if retrieved and not cited and not is_refusal:
        problems.append("sem_citacao")

    if excerpts and not is_refusal and not invented and not not_retrieved:
        cited_valid = [f"{p} §{s}" for p, s in cited if f"{p} §{s}" in excerpts] or list(retrieved)
        reference = " ".join(excerpts.get(c, "") for c in cited_valid) + " " + context
        everything = " ".join(excerpts.values()) + " " + context
        support, unsupported = grounding(text, reference, everything)
        if support < SUPPORT_THRESHOLD:
            problems.append(f"conteudo_nao_aderente({support:.2f})")
        if unsupported:
            problems.append(f"numero_sem_suporte({', '.join(unsupported)})")

    unverified = invented or not_retrieved or any(p.startswith(("conteudo_nao_aderente", "numero_sem_suporte"))
                                                  for p in problems)
    if unverified and not is_refusal:
        excerpts = excerpts or {}
        lines = [UNVERIFIED_HEADER] + [f"- [{c}] {excerpts.get(c, '')}".rstrip() for c in retrieved]
        lines += ["", format_sources(retrieved)] if retrieved else []
        if clinical:
            lines.append(CLINICAL_DISCLAIMER)
        return OutputCheck(acao="bloqueada", texto="\n".join(lines), problemas=problems, fontes_validas=list(retrieved))

    # Explainability: a lista de fontes exibida é sempre a dos trechos efetivamente consultados.
    if invented or not_retrieved or (retrieved and not cited and not is_refusal):
        text = _SOURCE_LINE.sub("", text).strip()
        if retrieved:
            text += "\n\n" + format_sources(retrieved)
        if invented or not_retrieved:
            text += "\n(A citação gerada pelo modelo não correspondia aos trechos consultados e foi substituída.)"

    if clinical and not is_refusal and CLINICAL_DISCLAIMER not in text:
        text += f"\n{CLINICAL_DISCLAIMER}"
        problems.append("aviso_validacao_adicionado")

    blocking = {"citacao_inventada", "citacao_nao_recuperada", "sem_citacao", "pii_na_saida"}
    acao = "corrigida" if blocking & set(problems) else "aprovada"
    valid = sorted({f"{p} §{s}" for p, s in cited} & set(retrieved)) or list(retrieved)
    return OutputCheck(acao=acao, texto=text, problemas=problems, fontes_validas=valid)

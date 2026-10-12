"""Anonimização de texto clínico em português.

Combina três estratégias determinísticas e auditáveis:

1. padrões estruturados brasileiros (CPF, CNS, RG, telefone, e-mail, CEP, datas);
2. campos rotulados ("Nome:", "Endereço:", "Mãe:") e tratamento de profissionais ("Dr.", "Dra.");
3. dicionário de prenomes (Faker pt_BR) seguido de sobrenome capitalizado.

Cada ocorrência é trocada por um marcador (``[CPF]``, ``[NOME]``...), e a função
devolve a contagem por tipo para o relatório de dados.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

_NAME_WORD = r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+"
_SEP = r"[ \t]+"
_HONORIFIC = r"(?:Sra?|Srta|Dra?)\.[ \t]+"
_NAME_SEQ = rf"{_NAME_WORD}(?:{_SEP}(?:d[aeo]s?{_SEP}|e{_SEP})?{_NAME_WORD})*"

# Ordem importa: padrões mais específicos primeiro.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CNS", re.compile(r"\b[1-2789]\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b")),
    ("RG", re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dXx]\b")),
    ("TELEFONE", re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
    ("DATA", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
]

_LABELED_FIELDS: list[tuple[str, re.Pattern[str]]] = [
    # O valor não pode começar com "[": já anonimizado (permite reaplicar a função como checagem).
    ("ENDERECO", re.compile(r"(?i)(endere[çc]o\s*:[ \t]*)((?![ \t\[])[^\n]+)")),
    ("NOME", re.compile(rf"(?i:(?:nome|paciente|m[ãa]e|pai|acompanhante|respons[áa]vel)\s*:\s*)((?:{_HONORIFIC})?{_NAME_SEQ})")),
    ("MEDICO", re.compile(rf"\b(?:Dra?\.|Doutora?|Enf\.)\s+({_NAME_SEQ})")),
]


@lru_cache(maxsize=1)
def _first_names() -> frozenset[str]:
    from faker.providers.person.pt_BR import Provider

    names = set(Provider.first_names_female) | set(Provider.first_names_male)
    # Prenomes compostos ("Carlos Eduardo") viram palavras isoladas.
    return frozenset(part for name in names for part in name.split())


_GAZETTEER = re.compile(rf"\b{_NAME_SEQ}")


@dataclass
class AnonymizationResult:
    text: str
    counts: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def anonymize(text: str) -> AnonymizationResult:
    counts: Counter = Counter()

    for label, pattern in _LABELED_FIELDS:
        def _replace_group(match: re.Match[str], label: str = label) -> str:
            counts[label] += 1
            start, end = match.span(match.lastindex)
            rel_start, rel_end = start - match.start(), end - match.start()
            whole = match.group(0)
            return whole[:rel_start] + f"[{label}]" + whole[rel_end:]

        text = pattern.sub(_replace_group, text)

    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        counts[label] += n

    first_names = _first_names()

    def _replace_name(match: re.Match[str]) -> str:
        # A sequência capitalizada pode começar por uma palavra comum ("Hoje Carlos Mendes").
        # O nome começa no primeiro prenome conhecido seguido de pelo menos mais uma palavra.
        seq = match.group(0)
        words = list(re.finditer(_NAME_WORD, seq))
        for word in words[:-1]:
            if word.group(0) in first_names:
                counts["NOME"] += 1
                return seq[: word.start()] + "[NOME]"
        return seq

    text = _GAZETTEER.sub(_replace_name, text)
    return AnonymizationResult(text=text, counts=+counts)


def find_pii(text: str) -> list[str]:
    """Retorna os tipos de PII ainda detectáveis no texto (lista vazia = limpo)."""
    return sorted(anonymize(text).counts)

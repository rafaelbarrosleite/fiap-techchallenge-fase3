"""Busca nos protocolos internos (RAG) com citação por seção.

Cada seção de protocolo é um ``Document`` LangChain com metadados de citação
(``PRT-001 §2``). O ranqueamento é BM25 (léxico, determinístico e explicável): o
vocabulário clínico dos protocolos é estável e o volume é pequeno (~60 seções), o
que dispensa embeddings e GPU. As perguntas frequentes ligadas a cada seção entram
apenas no texto indexado, para aproximar a linguagem das perguntas dos médicos.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from rank_bm25 import BM25Okapi
from snowballstemmer import stemmer

from assistente_medico.data.hospital_corpus import SYNTHETIC_DIR, Protocol, load_protocols

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_STOPWORDS = frozenset(
    "a o as os um uma uns umas de da do das dos em no na nos nas para por pelo pela com sem que se e ou "
    "ao aos à às é ser como qual quais quando quanto quanta onde devo deve fazer pode posso tem há "
    "mais menos muito já não sim seu sua seus suas este esta isso esse essa paciente protocolo".split()
)


# Linhas cadastrais e marcadores da anonimização não têm conteúdo clínico e puxavam a seção
# de privacidade (PRT-000 §5), que cita "nome, CPF, telefone e endereço".
_ADMIN_LINE = re.compile(r"(?im)^\s*(paciente|nome|cpf|cns|rg|telefone|endere[çc]o|m[ãa]e|pai|acompanhante)\s*:.*$")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
# Palavras de instrução e unidades de medida: em notas longas, "mg/dL" e "horas" casavam com
# qualquer protocolo que cite valores laboratoriais.
_INSTRUCTION_WORDS = frozenset(
    "resuma resumo nota admissao abaixo admitido admitida avaliado avaliada dra dr idade anos passagem plantao "
    "mg dl mmhg mmol meq bpm irpm kg ml min".split()
)


def clinical_query(text: str) -> str:
    """Reduz a consulta ao conteúdo clínico: sem linhas cadastrais, marcadores ou palavras de instrução."""
    text = _PLACEHOLDER.sub(" ", _ADMIN_LINE.sub(" ", text))
    return " ".join(w for w in _WORD.findall(text.lower()) if normalize(w) not in _INSTRUCTION_WORDS | _STOPWORDS)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


_STEMMER = stemmer("portuguese")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in _STOPWORDS]


_WORD = re.compile(r"[a-zà-öø-ÿ0-9]+(?:[.,][0-9]+)?")


def stem_tokens(text: str) -> list[str]:
    """Radicais Snowball ("internar"/"internação" → "intern") para casar flexões das palavras.

    O stemmer português precisa das palavras acentuadas; os acentos saem só depois.
    """
    out = []
    for word in _WORD.findall(text.lower()):
        if normalize(word) in _STOPWORDS:
            continue
        out.append(normalize(word if word[0].isdigit() else _STEMMER.stemWord(word)))
    return out


def section_documents(protocols: list[Protocol]) -> list[Document]:
    docs = []
    for proto in protocols:
        for sec in proto.secoes:
            citation = f"{proto.id} §{sec.id}"
            docs.append(Document(
                page_content=f"{proto.titulo} — {sec.titulo}. {sec.texto.strip()}",
                metadata={
                    "citation": citation,
                    "protocol_id": proto.id,
                    "protocol_title": proto.titulo,
                    "protocol_version": proto.versao,
                    "section_id": sec.id,
                    "section_title": sec.titulo,
                    "index_text": " ".join([proto.titulo, proto.area, sec.titulo, sec.texto]),
                },
            ))
    return docs


class ProtocolRetriever(BaseRetriever):
    """Retriever BM25 sobre as seções dos protocolos, com score e citação nos metadados."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    k: int = 3
    # Score relativo mínimo (fração do melhor score) para um documento acompanhar o primeiro.
    relative_threshold: float = 0.35
    # Score absoluto mínimo: descarta apenas perguntas sem nenhuma relação com os protocolos.
    min_score: float = 2.0
    _bm25: BM25Okapi = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._bm25 = BM25Okapi([stem_tokens(d.metadata["index_text"]) for d in self.documents])

    def scored(self, query: str) -> list[tuple[Document, float]]:
        scores = self._bm25.get_scores(stem_tokens(clinical_query(query)))
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        ranked = self.scored(query)
        if not ranked or ranked[0][1] < self.min_score:
            return []
        best = ranked[0][1]
        out = []
        for doc, score in ranked[: self.k]:
            if score < best * self.relative_threshold:
                break
            out.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "score": round(score, 3)}))
        return out


    def within_protocol(self, query: str, protocol_id: str, k: int | None = None) -> list[Document]:
        """Seções de um protocolo já definido (ex.: protocolo aberto no atendimento), ranqueadas pela consulta."""
        ranked = [(d, s) for d, s in self.scored(query) if d.metadata["protocol_id"] == protocol_id]
        return [Document(page_content=d.page_content, metadata={**d.metadata, "score": round(s, 3)})
                for d, s in ranked[: k or self.k]]


@lru_cache(maxsize=4)
def get_retriever(base_dir: Path = SYNTHETIC_DIR, k: int = 3) -> ProtocolRetriever:
    return ProtocolRetriever(documents=section_documents(load_protocols(base_dir)), k=k)


def format_context(docs: list[Document]) -> str:
    """Formato idêntico ao usado nos exemplos ``faq_rag`` do fine-tuning."""
    return "\n".join(
        f"[{d.metadata['protocol_id']} §{d.metadata['section_id']} — {d.metadata['section_title']}] "
        f"{d.page_content.split('. ', 1)[-1]}"
        for d in docs
    )

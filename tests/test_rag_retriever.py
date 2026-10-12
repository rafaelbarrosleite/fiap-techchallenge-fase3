from assistente_medico.data.hospital_corpus import load_protocols
from assistente_medico.rag.retriever import clinical_query, format_context, get_retriever, stem_tokens, tokenize


def test_tokenize_remove_acentos_e_stopwords():
    assert tokenize("Quais exames pedir na Cetoacidose?") == ["exames", "pedir", "cetoacidose"]
    assert "6,9" in tokenize("potássio de 6,9")
    assert stem_tokens("internar")[0] == stem_tokens("internação")[0]


def test_consulta_clinica_ignora_cadastro_e_marcadores():
    nota = "Resuma a nota abaixo.\nPaciente: [NOME]\nCPF: [CPF] | Telefone: [TELEFONE]\nAdmitido em [DATA] com urticária."
    assert clinical_query(nota) == "urticária"


def test_cada_secao_vira_documento_citavel():
    retriever = get_retriever()
    n_sections = sum(len(p.secoes) for p in load_protocols())
    assert len(retriever.documents) == n_sections
    assert {d.metadata["citation"] for d in retriever.documents} >= {"PRT-001 §2", "PRT-012 §2"}


def test_recupera_secao_correta_com_score():
    docs = get_retriever().invoke("Qual a meta porta-balão no infarto com supra?")
    assert docs[0].metadata["citation"] == "PRT-002 §2"
    assert docs[0].metadata["score"] > 0
    # Caso conhecido em que a seção correta não fica em 1º, mas está entre as 3 primeiras.
    top3 = get_retriever().invoke("Em quanto tempo a tomografia deve ser feita no AVC?")
    assert "PRT-003 §2" in [d.metadata["citation"] for d in top3]


def test_taxa_de_acerto_nas_perguntas_do_faq():
    retriever = get_retriever()
    questions = [(q, p.id) for p in load_protocols() for f in p.faq for q in f.perguntas]
    hit3 = sum(any(d.metadata["protocol_id"] == pid for d in retriever.invoke(q)) for q, pid in questions)
    assert hit3 / len(questions) >= 0.97


def test_notas_clinicas_anonimizadas_encontram_o_protocolo_do_caso():
    from assistente_medico.data.hospital_corpus import build_clinical_note_examples
    from assistente_medico.evaluation.metrics import cited_protocols

    notes, _ = build_clinical_note_examples(n_notes=6)
    for note in notes:
        expected = cited_protocols(note["messages"][2]["content"])
        found = {d.metadata["protocol_id"] for d in get_retriever().invoke(note["messages"][1]["content"])}
        assert expected & found, note["messages"][1]["content"]


def test_pergunta_sem_relacao_nao_retorna_documentos():
    assert get_retriever().invoke("Como fazer bolo de cenoura?") == []


def test_contexto_no_mesmo_formato_do_treino():
    docs = get_retriever().invoke("Qual a meta porta-balão no infarto com supra?")
    assert format_context(docs[:1]).startswith("[PRT-002 §2 — Supradesnivelamento de ST] Diante de supradesnivelamento")

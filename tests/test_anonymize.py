from faker import Faker

from assistente_medico.data.anonymize import anonymize, find_pii


def test_remove_identificadores_estruturados():
    text = (
        "CPF 123.456.789-09, CNS 898 0012 3456 7890, RG 12.345.678-9, "
        "tel (11) 98765-4321, e-mail maria.souza@email.com, CEP 01310-100, "
        "internada em 03/10/2026."
    )
    result = anonymize(text)
    for raw in ["123.456.789-09", "898 0012", "12.345.678-9", "98765-4321", "maria.souza", "01310-100", "03/10/2026"]:
        assert raw not in result.text
    assert {"CPF", "CNS", "RG", "TELEFONE", "EMAIL", "CEP", "DATA"} <= set(result.counts)


def test_remove_nomes_em_campos_rotulados_e_profissionais():
    text = "Paciente: Joana da Silva Prado\nMãe: Lucia Prado\nAvaliada pela Dra. Fernanda Lima.\nEndereço: Rua das Flores, 123"
    result = anonymize(text)
    assert "Joana" not in result.text and "Lucia" not in result.text
    assert "Fernanda" not in result.text and "Rua das Flores" not in result.text
    assert "Paciente: [NOME]" in result.text
    assert "Dra. [MEDICO]" in result.text


def test_remove_nome_livre_no_texto_pelo_dicionario():
    result = anonymize("Hoje Carlos Eduardo Mendes relatou dor torácica.")
    assert "Carlos" not in result.text and "Mendes" not in result.text


def test_preserva_texto_clinico_e_identificador_pseudonimizado():
    text = "Paciente: P012. Lactato 4,6 mmol/L, PAM 60 mmHg. Iniciar protocolo PRT-001 §2."
    result = anonymize(text)
    assert result.text == text
    assert result.total == 0


def test_nenhuma_pii_sobrevive_em_registros_falsos():
    fake = Faker("pt_BR")
    Faker.seed(7)
    for _ in range(50):
        name = fake.name()
        raw = f"Paciente: {name}, CPF {fake.cpf()}, telefone {fake.cellphone_number()}, e-mail {fake.email()}."
        clean = anonymize(raw).text
        assert fake.cpf() not in clean
        for part in name.replace("Dr. ", "").replace("Dra. ", "").split():
            if len(part) > 3 and part[0].isupper():
                assert part not in clean, (raw, clean)
        assert find_pii(clean) == []


def test_reaplicar_em_texto_anonimizado_nao_gera_falso_alarme():
    clean = anonymize("Paciente: Sra. Maria Souza\nEndereço: Rua A, 10\nCPF: 123.456.789-09").text
    assert clean == "Paciente: [NOME]\nEndereço: [ENDERECO]\nCPF: [CPF]"
    assert find_pii(clean) == []

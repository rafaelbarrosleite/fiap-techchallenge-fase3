"""Gera o banco sintético de prontuários (``uv run build-hospital-db``).

Pacientes de cenário reproduzem situações dos protocolos (exame pendente, valor crítico,
alergia relevante); pacientes de preenchimento dão volume realista. Os horários são
relativos a ``now`` para que "pendente há X horas" faça sentido no dia da demonstração.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DB_PATH = Path("data/hospital.db")
SCHEMA = Path(__file__).with_name("schema.sql")

EXAM_UNITS = {
    "lactato": ("Lactato sérico", "mmol/L"), "hemocultura": ("Hemocultura (2 amostras)", None),
    "troponina": ("Troponina ultrassensível", "ng/L"), "potassio": ("Potássio sérico", "mEq/L"),
    "sodio": ("Sódio sérico", "mEq/L"), "glicemia": ("Glicemia", "mg/dL"), "hemoglobina": ("Hemoglobina", "g/dL"),
    "plaquetas": ("Plaquetas", "/mm3"), "creatinina": ("Creatinina", "mg/dL"), "ureia": ("Ureia", "mg/dL"),
    "gasometria_ph": ("Gasometria arterial — pH", None), "bicarbonato": ("Bicarbonato", "mEq/L"),
    "bnp": ("BNP", "pg/mL"), "tc_cranio": ("Tomografia de crânio sem contraste", None),
    "rx_torax": ("Radiografia de tórax", None), "ecg": ("Eletrocardiograma 12 derivações", None),
}

# Cada cenário: paciente, atendimento, sinais vitais, exames (horas atrás da solicitação; None = não realizado).
SCENARIOS: list[dict[str, Any]] = [
    {
        "paciente": dict(id="P001", idade=67, sexo="F", peso_kg=62, comorbidades="diabetes tipo 2;doença renal crônica",
                         alergias="", setor="Pronto-socorro", leito="PS-03"),
        "atendimento": dict(horas=3, queixa="febre, disúria e confusão mental", hipotese="suspeita de sepse de foco urinário",
                            protocolo_id="PRT-001"),
        "vitais": dict(pas=86, pad=50, fc=118, fr=26, temperatura=38.9, spo2=93, glasgow=14),
        "exames": [("lactato", "urgente", 2.5, None, None), ("hemocultura", "urgente", 2.5, None, None),
                   ("creatinina", "rotina", 2.5, 2.0, 2.4), ("plaquetas", "rotina", 2.5, 2.0, 98000)],
        "prescricoes": [],
    },
    {
        "paciente": dict(id="P002", idade=58, sexo="M", peso_kg=84, comorbidades="hipertensão;tabagismo",
                         alergias="", setor="Pronto-socorro", leito="PS-07"),
        "atendimento": dict(horas=2, queixa="dor torácica opressiva há 1 hora", hipotese="síndrome coronariana aguda sem supra de ST",
                            protocolo_id="PRT-002"),
        "vitais": dict(pas=148, pad=92, fc=96, fr=18, temperatura=36.6, spo2=96, glasgow=15),
        "exames": [("ecg", "urgente", 2.0, 1.9, None, "ritmo sinusal, infradesnivelamento de ST de 1 mm em V5–V6"),
                   ("troponina", "urgente", 2.0, 1.8, 38), ("troponina", "urgente", 1.5, None, None)],
        "prescricoes": [],
    },
    {
        "paciente": dict(id="P003", idade=74, sexo="M", peso_kg=70, comorbidades="insuficiência cardíaca;doença renal crônica",
                         alergias="", setor="Enfermaria Clínica", leito="EC-12"),
        "atendimento": dict(horas=40, queixa="dispneia progressiva e edema de membros inferiores",
                            hipotese="insuficiência cardíaca descompensada perfil quente e úmido", protocolo_id="PRT-011"),
        "vitais": dict(pas=132, pad=78, fc=88, fr=22, temperatura=36.4, spo2=92, glasgow=15),
        "exames": [("potassio", "rotina", 6.0, 4.0, 6.8), ("creatinina", "rotina", 6.0, 4.0, 2.9),
                   ("bnp", "rotina", 30.0, 26.0, 1840)],
        "prescricoes": [("furosemida IV", "congestão na IC descompensada", "PRT-011 §3", "ativa")],
    },
    {
        "paciente": dict(id="P004", idade=29, sexo="F", peso_kg=58, comorbidades="diabetes tipo 1",
                         alergias="", setor="Pronto-socorro", leito="PS-01"),
        "atendimento": dict(horas=4, queixa="vômitos, dor abdominal e respiração rápida", hipotese="cetoacidose diabética",
                            protocolo_id="PRT-007"),
        "vitais": dict(pas=104, pad=62, fc=122, fr=30, temperatura=36.9, spo2=98, glasgow=15),
        "exames": [("glicemia", "urgente", 4.0, 3.8, 486), ("gasometria_ph", "urgente", 4.0, 3.7, 7.12),
                   ("bicarbonato", "urgente", 4.0, 3.7, 9), ("potassio", "urgente", 4.0, 3.7, 3.0),
                   ("gasometria_ph", "urgente", 1.5, None, None)],
        "prescricoes": [],
    },
    {
        "paciente": dict(id="P005", idade=45, sexo="F", peso_kg=71, comorbidades="asma",
                         alergias="penicilina;dipirona", setor="Enfermaria Clínica", leito="EC-04"),
        "atendimento": dict(horas=30, queixa="tosse produtiva, febre e dor pleurítica", hipotese="pneumonia adquirida na comunidade",
                            protocolo_id="PRT-004"),
        "vitais": dict(pas=118, pad=72, fc=98, fr=24, temperatura=38.2, spo2=94, glasgow=15),
        "exames": [("rx_torax", "rotina", 30.0, 28.0, None, "consolidação em lobo inferior direito"),
                   ("ureia", "rotina", 30.0, 28.0, 38), ("hemocultura", "rotina", 30.0, None, None)],
        "prescricoes": [],
    },
    {
        "paciente": dict(id="P006", idade=81, sexo="F", peso_kg=55, comorbidades="fibrilação atrial;hipertensão",
                         alergias="", setor="Pronto-socorro", leito="PS-02"),
        "atendimento": dict(horas=1, queixa="hemiparesia direita e afasia iniciadas há 2 horas", hipotese="AVC isquêmico agudo",
                            protocolo_id="PRT-003"),
        "vitais": dict(pas=192, pad=104, fc=102, fr=18, temperatura=36.5, spo2=95, glasgow=13),
        "exames": [("glicemia", "urgente", 1.0, 0.9, 132), ("tc_cranio", "urgente", 0.9, None, None)],
        "prescricoes": [],
    },
    {
        "paciente": dict(id="P007", idade=63, sexo="M", peso_kg=96, comorbidades="obesidade;DPOC",
                         alergias="", setor="Enfermaria Clínica", leito="EC-09"),
        "atendimento": dict(horas=20, queixa="exacerbação de DPOC", hipotese="DPOC exacerbado, imobilidade no leito",
                            protocolo_id="PRT-008"),
        "vitais": dict(pas=128, pad=80, fc=84, fr=20, temperatura=36.8, spo2=91, glasgow=15),
        "exames": [("plaquetas", "rotina", 20.0, 18.0, 215000), ("creatinina", "rotina", 20.0, 18.0, 1.0),
                   ("hemoglobina", "rotina", 20.0, 18.0, 12.8)],
        "prescricoes": [],
    },
    {
        "paciente": dict(id="P008", idade=36, sexo="M", peso_kg=78, comorbidades="",
                         alergias="ceftriaxona", setor="Pronto-socorro", leito="PS-05"),
        "atendimento": dict(horas=1, queixa="urticária, sibilos e hipotensão após antibiótico IV", hipotese="anafilaxia",
                            protocolo_id="PRT-010"),
        "vitais": dict(pas=82, pad=44, fc=128, fr=28, temperatura=36.7, spo2=91, glasgow=15),
        "exames": [],
        "prescricoes": [("ceftriaxona IV", "infecção de partes moles", "prescrição prévia", "suspensa")],
    },
]

_FILLER_COMPLAINTS = [
    ("pós-operatório de colecistectomia", "evolução pós-operatória", None),
    ("celulite em membro inferior", "infecção de pele e partes moles", None),
    ("hiperglicemia sem cetose", "diabetes descompensado", None),
    ("crise hipertensiva sem sintomas", "urgência hipertensiva", "PRT-009"),
    ("broncoespasmo leve", "crise asmática leve", "PRT-005"),
    ("hipoglicemia em uso de insulina", "hipoglicemia", "PRT-006"),
]


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _insert_exam(cur: sqlite3.Cursor, pid: str, now: datetime, spec: tuple) -> None:
    nome, prioridade, solicitado_h, realizado_h, valor, *texto = spec
    descricao, unidade = EXAM_UNITS[nome]
    realizado = _iso(now - timedelta(hours=realizado_h)) if realizado_h is not None else None
    cur.execute(
        "INSERT INTO exames (paciente_id, nome, descricao, prioridade, solicitado_em, realizado_em, valor, unidade,"
        " resultado_texto, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (pid, nome, descricao, prioridade, _iso(now - timedelta(hours=solicitado_h)), realizado,
         valor, unidade, texto[0] if texto else None, "realizado" if realizado else "solicitado"),
    )


def _insert_scenario(cur: sqlite3.Cursor, sc: dict[str, Any], now: datetime) -> None:
    p = sc["paciente"]
    cur.execute("INSERT INTO pacientes VALUES (:id,:idade,:sexo,:peso_kg,:comorbidades,:alergias,:setor,:leito)", p)
    a = sc["atendimento"]
    cur.execute(
        "INSERT INTO atendimentos (paciente_id, admitido_em, queixa, hipotese, protocolo_id, status) VALUES (?,?,?,?,?,?)",
        (p["id"], _iso(now - timedelta(hours=a["horas"])), a["queixa"], a["hipotese"], a["protocolo_id"], "ativo"),
    )
    v = sc["vitais"]
    cur.execute(
        "INSERT INTO sinais_vitais (paciente_id, registrado_em, pas, pad, fc, fr, temperatura, spo2, glasgow)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (p["id"], _iso(now - timedelta(minutes=20)), v["pas"], v["pad"], v["fc"], v["fr"], v["temperatura"], v["spo2"], v["glasgow"]),
    )
    for spec in sc["exames"]:
        _insert_exam(cur, p["id"], now, spec)
    for item, indicacao, fonte, status in sc["prescricoes"]:
        cur.execute(
            "INSERT INTO prescricoes (paciente_id, item, indicacao, fonte, status, criado_por, criado_em, validado_por,"
            " validado_em) VALUES (?,?,?,?,?,?,?,?,?)",
            (p["id"], item, indicacao, fonte, status, "MED-01", _iso(now - timedelta(hours=10)), "MED-01",
             _iso(now - timedelta(hours=10))),
        )


def _filler(rng: random.Random, index: int, now: datetime) -> dict[str, Any]:
    queixa, hipotese, protocolo = _FILLER_COMPLAINTS[index % len(_FILLER_COMPLAINTS)]
    pid = f"P{index + len(SCENARIOS) + 1:03d}"
    return {
        "paciente": dict(id=pid, idade=rng.randint(19, 90), sexo=rng.choice("FM"), peso_kg=rng.randint(50, 105),
                         comorbidades=rng.choice(["", "hipertensão", "diabetes tipo 2", "hipertensão;dislipidemia"]),
                         alergias=rng.choice(["", "", "", "sulfa", "látex"]),
                         setor="Enfermaria Clínica", leito=f"EC-{rng.randint(13, 40):02d}"),
        "atendimento": dict(horas=rng.randint(8, 96), queixa=queixa, hipotese=hipotese, protocolo_id=protocolo),
        "vitais": dict(pas=rng.randint(105, 145), pad=rng.randint(60, 90), fc=rng.randint(62, 98), fr=rng.randint(14, 20),
                       temperatura=round(rng.uniform(36.0, 37.4), 1), spo2=rng.randint(94, 99), glasgow=15),
        "exames": [("hemoglobina", "rotina", 20.0, 18.0, round(rng.uniform(10.5, 15.0), 1)),
                   ("creatinina", "rotina", 20.0, 18.0, round(rng.uniform(0.6, 1.3), 2)),
                   ("potassio", "rotina", 20.0, 18.0, round(rng.uniform(3.6, 5.0), 1))],
        "prescricoes": [],
    }


def build(db_path: Path = DB_PATH, now: datetime | None = None, n_filler: int = 22, seed: int = 42) -> Path:
    now = now or datetime.now()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    rng = random.Random(seed)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        cur = conn.cursor()
        for sc in SCENARIOS:
            _insert_scenario(cur, sc, now)
        for i in range(n_filler):
            _insert_scenario(cur, _filler(rng, i, now), now)
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o banco sintético de prontuários (Missão 4).")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    path = build(args.db)
    with sqlite3.connect(path) as conn:
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("pacientes", "atendimentos", "sinais_vitais", "exames", "prescricoes", "alertas")}
    print(f"Banco gerado em {path}: {counts}")

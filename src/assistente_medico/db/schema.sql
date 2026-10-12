-- Prontuário eletrônico sintético do Hospital Escola (fictício).
-- Pacientes são identificados apenas por código pseudonimizado: não há nome, CPF ou contato.

CREATE TABLE IF NOT EXISTS pacientes (
    id            TEXT PRIMARY KEY,           -- ex.: P001
    idade         INTEGER NOT NULL,
    sexo          TEXT NOT NULL CHECK (sexo IN ('F', 'M')),
    peso_kg       REAL,
    comorbidades  TEXT NOT NULL DEFAULT '',   -- lista separada por ';'
    alergias      TEXT NOT NULL DEFAULT '',   -- lista separada por ';' (vazio = nenhuma conhecida)
    setor         TEXT NOT NULL,
    leito         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS atendimentos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id     TEXT NOT NULL REFERENCES pacientes(id),
    admitido_em     TEXT NOT NULL,            -- ISO 8601
    queixa          TEXT NOT NULL,
    hipotese        TEXT NOT NULL,
    protocolo_id    TEXT,                     -- protocolo aberto, se houver
    status          TEXT NOT NULL CHECK (status IN ('ativo', 'alta'))
);

CREATE TABLE IF NOT EXISTS sinais_vitais (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id     TEXT NOT NULL REFERENCES pacientes(id),
    registrado_em   TEXT NOT NULL,
    pas INTEGER, pad INTEGER, fc INTEGER, fr INTEGER,
    temperatura REAL, spo2 INTEGER, glasgow INTEGER
);

CREATE TABLE IF NOT EXISTS exames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id     TEXT NOT NULL REFERENCES pacientes(id),
    nome            TEXT NOT NULL,            -- chave do exame, ex.: potassio, lactato, troponina
    descricao       TEXT NOT NULL,
    prioridade      TEXT NOT NULL CHECK (prioridade IN ('rotina', 'urgente')),
    solicitado_em   TEXT NOT NULL,
    realizado_em    TEXT,                     -- NULL = ainda não realizado
    valor           REAL,
    unidade         TEXT,
    resultado_texto TEXT,
    status          TEXT NOT NULL CHECK (status IN ('solicitado', 'realizado', 'cancelado'))
);

CREATE TABLE IF NOT EXISTS prescricoes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id     TEXT NOT NULL REFERENCES pacientes(id),
    item            TEXT NOT NULL,
    indicacao       TEXT NOT NULL,
    fonte           TEXT,
    status          TEXT NOT NULL CHECK (status IN ('pendente_validacao', 'aprovada', 'rejeitada', 'ativa', 'suspensa')),
    criado_por      TEXT NOT NULL,            -- 'assistente' ou identificador do médico
    criado_em       TEXT NOT NULL,
    validado_por    TEXT,
    validado_em     TEXT,
    observacao      TEXT
);

CREATE TABLE IF NOT EXISTS alertas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id     TEXT NOT NULL REFERENCES pacientes(id),
    gravidade       TEXT NOT NULL CHECK (gravidade IN ('CRITICO', 'ATENCAO')),
    titulo          TEXT NOT NULL,
    motivo          TEXT NOT NULL,
    acao            TEXT NOT NULL,
    fonte           TEXT NOT NULL,
    destino         TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('aberto', 'reconhecido')),
    criado_em       TEXT NOT NULL,
    trace_id        TEXT
);

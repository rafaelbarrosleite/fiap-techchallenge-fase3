"""Log de auditoria estruturado (JSONL) com trace_id por execução.

Cada evento vira uma linha JSON em ``<AUDIT_LOG_DIR>/audit.jsonl``. Campos
com dados de identificação direta são recusados: o log deve permitir
rastrear decisões sem expor o paciente.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORBIDDEN_FIELDS = frozenset({"nome", "name", "cpf", "rg", "cns", "telefone", "email", "endereco"})


class AuditLogger:
    def __init__(self, log_dir: str | Path | None = None, trace_id: str | None = None) -> None:
        self.log_dir = Path(log_dir or os.getenv("AUDIT_LOG_DIR", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "audit.jsonl"
        self.trace_id = trace_id or uuid.uuid4().hex

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        forbidden = FORBIDDEN_FIELDS.intersection(k.lower() for k in fields)
        if forbidden:
            raise ValueError(f"Campos de identificação proibidos no log: {sorted(forbidden)}")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": self.trace_id,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

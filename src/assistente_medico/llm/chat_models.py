"""Modelos de chat LangChain: Llama ajustado (MLX) e modelo fake determinístico.

``MLXChatModel`` roda o Llama 3.2 3B com o adaptador LoRA no mesmo processo (sem servidor).
``FakeClinicalChatModel`` responde a partir dos trechos de protocolo presentes no prompt,
com o mesmo formato de resposta do fine-tuning; serve para testes e demonstração offline.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from assistente_medico.llm.prompts import CLINICAL_DISCLAIMER

BASE_MODEL = os.getenv("BASE_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "artifacts/finetune/official/adapters")

_ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}


def to_chat_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    out = []
    for message in messages:
        role = next((r for cls, r in _ROLES.items() if isinstance(message, cls)), None)
        if role is None:
            raise ValueError(f"Tipo de mensagem não suportado: {type(message).__name__}")
        out.append({"role": role, "content": str(message.content)})
    return out


@lru_cache(maxsize=2)
def _load(model: str, adapter_path: str | None):
    from mlx_lm import load

    return load(model, adapter_path=adapter_path)


class MLXChatModel(BaseChatModel):
    """Llama 3.2 3B + LoRA via MLX-LM. Decodificação gulosa (temperatura 0) para reprodutibilidade.

    ``repetition_penalty`` evita o laço de repetição típico de modelos pequenos em respostas
    longas (ex.: repetir o aviso de validação). A avaliação da M3 usou decodificação gulosa pura.
    """

    model: str = BASE_MODEL
    adapter_path: str | None = ADAPTER_PATH
    max_tokens: int = 320
    repetition_penalty: float | None = 1.15

    @property
    def _llm_type(self) -> str:
        return "mlx-llama-lora"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "adapter_path": self.adapter_path, "max_tokens": self.max_tokens}

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors

        model, tokenizer = _load(self.model, self.adapter_path)
        prompt = tokenizer.apply_chat_template(to_chat_dicts(messages), add_generation_prompt=True, tokenize=False)
        processors = make_logits_processors(repetition_penalty=self.repetition_penalty) if self.repetition_penalty else None
        text = generate(model, tokenizer, prompt=prompt, max_tokens=kwargs.get("max_tokens", self.max_tokens),
                        logits_processors=processors, verbose=False)
        if stop:
            for token in stop:
                text = text.split(token)[0]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text.strip()))])


_SNIPPET = re.compile(r"^\[(PRT-\d{3}) §(\d+) — ([^\]]+)\] (.+)$", re.MULTILINE)


class FakeClinicalChatModel(BaseChatModel):
    """Responde com o primeiro trecho de protocolo do prompt, citando-o; sem trechos, declara falta de fonte."""

    @property
    def _llm_type(self) -> str:
        return "fake-clinical"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        prompt = str(messages[-1].content)
        match = _SNIPPET.search(prompt)
        if match is None:
            text = ("Não encontrei protocolo interno que cubra esse tema, então não posso responder com segurança. "
                    "Fonte: PRT-000 §4.")
        else:
            pid, sid, title, body = match.groups()
            text = f"{body.strip()}\n\nFonte: {pid} §{sid} ({title}).\n{CLINICAL_DISCLAIMER}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_chat_model(provider: str | None = None) -> BaseChatModel:
    provider = provider or os.getenv("LLM_PROVIDER", "mlx")
    if provider == "mlx":
        return MLXChatModel()
    if provider == "mlx-base":
        return MLXChatModel(adapter_path=None)
    if provider == "fake":
        return FakeClinicalChatModel()
    raise ValueError(f"Provider desconhecido: {provider} (use mlx, mlx-base ou fake)")

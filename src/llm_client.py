import os
from dataclasses import dataclass
from typing import Any

from src.config import get_mode


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    api_base: str
    timeout: int = 30


def generate_text(prompt: str, system_prompt: str | None = None) -> str:
    mode = get_mode()
    if mode in {"real", "hybrid"}:
        try:
            return generate_real_text(prompt, system_prompt=system_prompt)
        except Exception:
            if mode == "real":
                raise

    return generate_mock_text(prompt)


def generate_mock_text(prompt: str) -> str:
    normalized = prompt.lower()
    if "summary" in normalized or "总结" in prompt:
        return "模拟总结：该资料围绕课程项目要求、技术路线、评分标准和成员贡献展开。"
    if "quiz" in normalized or "出题" in prompt or "选择题" in prompt:
        return "模拟出题：请根据检索证据设计题目，并保留来源 chunk_id。"
    if "拒答" in prompt or "证据不足" in prompt:
        return "当前知识库未检索到足够相关的课程资料，无法可靠回答该问题。"
    return (
        "模拟回答：根据已检索到的课程证据，系统会优先基于原文片段回答，并给出引用来源。"
        "当前为 mock 模式，后续可在 real 模式下接入真实大语言模型 API。"
    )


def generate_real_text(prompt: str, system_prompt: str | None = None) -> str:
    config = get_llm_config()
    if config.provider not in {"openai", "openai_compatible"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.provider}")
    if not config.api_key:
        raise RuntimeError("Real LLM mode requires LLM_API_KEY.")

    return call_openai_compatible_chat(config, prompt, system_prompt=system_prompt)


def get_llm_config() -> LLMConfig:
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        api_base=os.getenv(
            "LLM_API_BASE", "https://api.openai.com/v1/chat/completions"
        ).strip(),
        timeout=int(os.getenv("LLM_TIMEOUT", "30")),
    )


def call_openai_compatible_chat(
    config: LLMConfig, prompt: str, system_prompt: str | None = None
) -> str:
    import requests

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        config.api_base,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    try:
        return str(payload["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected LLM API response format.") from exc


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
    temperature: float = 0.2
    max_tokens: int = 1800


def generate_text(prompt: str, system_prompt: str | None = None) -> str:
    mode = get_mode()
    if get_llm_config().provider == "mock":
        return generate_mock_text(prompt)

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
    if "评分标准" in prompt:
        return (
            "模拟回答：根据已检索到的课程资料，项目评分标准包含技术深度与正确性、"
            "演示效果与可交互性、创新性与工作量、PPT 结构与讲解表达、"
            "团队分工明确性和提交材料完整性等方面。具体分值请以引用来源中的评分表为准。"
        )
    if "项目要求" in prompt:
        return (
            "模拟回答：课程项目要求学生运用神经网络、强化学习等课堂所学技术，"
            "解决一个具体问题或完成一个有创意的任务，并完成可演示系统。"
        )
    return (
        "模拟回答：根据已检索到的课程证据，系统会优先基于原文片段回答，并给出引用来源。"
        "当前为 mock 模式，后续可在 real 模式下接入真实大语言模型 API。"
    )


def generate_real_text(prompt: str, system_prompt: str | None = None) -> str:
    config = get_llm_config()
    if config.provider not in {"openai", "openai_compatible", "deepseek"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.provider}")
    if not config.api_key:
        key_name = "DEEPSEEK_API_KEY or LLM_API_KEY" if config.provider == "deepseek" else "LLM_API_KEY"
        raise RuntimeError(f"Real LLM mode requires {key_name}.")

    return call_openai_compatible_chat(config, prompt, system_prompt=system_prompt)


def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    default_model = "deepseek-v4-flash" if provider == "deepseek" else "gpt-4o-mini"
    default_base = (
        "https://api.deepseek.com/chat/completions"
        if provider == "deepseek"
        else "https://api.openai.com/v1/chat/completions"
    )
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", api_key).strip()
    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=os.getenv("LLM_MODEL", default_model).strip() or default_model,
        api_base=normalize_chat_api_base(os.getenv("LLM_API_BASE", default_base).strip(), provider),
        timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1800")),
    )


def normalize_chat_api_base(api_base: str, provider: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if provider == "deepseek" and base == "https://api.deepseek.com":
        return f"{base}/chat/completions"
    return base


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
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    try:
        return str(payload["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected LLM API response format.") from exc

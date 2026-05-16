"""OpenAI SDK 调用封装。"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """大模型客户端封装。

    该类只负责与 OpenAI 兼容接口通信，不负责业务状态管理。
    """

    def __init__(self) -> None:
        """读取环境变量并初始化 OpenAI 客户端。"""
        load_dotenv()

        api_key = os.getenv("API_KEY", "")
        base_url = os.getenv("BASE_URL") or None

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek-v4-pro",
        temperature: float = 0.7,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发起一次非流式聊天补全请求。

        Args:
            messages: OpenAI Chat Completions 格式的消息列表。
            model: 模型名称，默认使用 deepseek-chat。
            temperature: 采样温度，数值越高回复越发散。
            response_format: OpenAI 兼容响应格式配置，例如 JSON Output。
            max_tokens: 限制最大输出长度，避免结构化 JSON 被中途截断。

        Returns:
            大模型返回的文本；发生异常时返回错误说明字符串。
        """
        try:
            request_payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format is not None:
                request_payload["response_format"] = response_format
            if max_tokens is not None:
                request_payload["max_tokens"] = max_tokens

            response: Any = self.client.chat.completions.create(
                **request_payload,
            )
            content = response.choices[0].message.content
            if content or response_format != {"type": "json_object"}:
                return content or ""

            # DeepSeek JSON Output 偶发返回空 content；同参数重试一次。
            response = self.client.chat.completions.create(
                **request_payload,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as exc:
            return f"模型调用失败：{exc}"

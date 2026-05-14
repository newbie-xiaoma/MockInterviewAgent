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
        model: str = "deepseek-chat",
        temperature: float = 0.7,
    ) -> str:
        """发起一次非流式聊天补全请求。

        Args:
            messages: OpenAI Chat Completions 格式的消息列表。
            model: 模型名称，默认使用 deepseek-chat。
            temperature: 采样温度，数值越高回复越发散。

        Returns:
            大模型返回的文本；发生异常时返回错误说明字符串。
        """
        try:
            response: Any = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as exc:
            return f"模型调用失败：{exc}"

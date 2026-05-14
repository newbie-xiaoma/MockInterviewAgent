"""面试评估智能体。"""

from __future__ import annotations

from core.llm_client import LLMClient
from prompts.evaluator import EVALUATOR_PROMPT


class EvaluatorAgent:
    """负责根据完整对话生成面试评估报告。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化评估智能体。

        Args:
            llm_client: 大模型客户端实例。
        """
        self.llm_client = llm_client

    def generate_report(
        self,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """生成 Markdown 格式的面试评估报告。

        Args:
            conversation_history: 完整面试对话历史。

        Returns:
            Markdown 格式评估报告。
        """
        messages = [
            {"role": "system", "content": EVALUATOR_PROMPT},
            {
                "role": "user",
                "content": (
                    "以下是完整面试对话历史，请生成评估报告：\n"
                    f"{conversation_history}"
                ),
            },
        ]
        return self.llm_client.chat(messages=messages, temperature=0.2)

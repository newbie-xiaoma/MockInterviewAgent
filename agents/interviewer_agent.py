"""面试官智能体。"""

from __future__ import annotations

from core.llm_client import LLMClient
from core.memory import ConversationMemory


class InterviewerAgent:
    """负责根据候选人回答生成下一轮面试问题。"""

    def __init__(
        self,
        llm_client: LLMClient,
        memory: ConversationMemory,
        persona_prompt: str,
    ) -> None:
        """初始化面试官智能体。

        Args:
            llm_client: 大模型客户端实例。
            memory: 对话记忆实例。
            persona_prompt: 面试官风格提示词。
        """
        self.llm_client = llm_client
        self.memory = memory
        self.persona_prompt = persona_prompt

    def generate_question(self, user_reply: str) -> str:
        """根据候选人回答生成下一个面试问题。

        Args:
            user_reply: 候选人的最新回答。

        Returns:
            面试官生成的下一条追问。
        """
        self.memory.add_message("user", user_reply)

        messages = [
            {"role": "system", "content": self.persona_prompt},
            *self.memory.get_history(),
        ]
        question = self.llm_client.chat(messages=messages)
        self.memory.add_message("assistant", question)
        return question

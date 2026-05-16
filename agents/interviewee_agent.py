"""候选人辅助回答智能体。"""

from __future__ import annotations

from core.llm_client import LLMClient
from prompts.interviewee import INTERVIEWEE_SYSTEM_PROMPT_TEMPLATE


class IntervieweeAgent:
    """面试 Copilot：根据 JD、简历和对话历史生成候选人参考回答。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化候选人辅助智能体。"""
        self.llm_client = llm_client

    def generate_draft_answer(
        self,
        conversation_memory: list[dict],
        jd_text: str,
        resume_text: str,
    ) -> str:
        """生成一段可供候选人参考的面试回答草稿。

        辅助智能体只读取对话历史作为上下文，不会修改 conversation_memory。
        """
        system_prompt = INTERVIEWEE_SYSTEM_PROMPT_TEMPLATE.format(
            jd_text=jd_text or "未提供岗位 JD。",
            resume_text=resume_text or "未提供简历内容。",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_memory,
        ]
        return self.llm_client.chat(
            messages=messages,
            temperature=0.6,
            max_tokens=600,
        ).strip()

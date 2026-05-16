"""回答指导智能体。

在面试评估报告生成后，对候选人每道题的作答进行逐题分析，
指出回答质量并提供真实面试中更优的作答建议。

设计原则：
- 与 EvaluatorAgent 独立，不依赖评分结果。
- 仅读取 conversation_memory 和 jd_text，不修改任何共享状态。
- 单次 LLM 调用生成全部逐题指导，保持效率。
"""

from __future__ import annotations

import json
from typing import Any

from core.llm_client import LLMClient
from prompts.answer_guidance import GUIDANCE_SYSTEM_PROMPT


class AnswerGuidanceAgent:
    """对候选人每道面试题的回答提供逐题指导。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化回答指导智能体。

        Args:
            llm_client: 大模型客户端实例。
        """
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def generate_guidance(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
        resume_text: str = "",
    ) -> str:
        """生成逐题回答指导 Markdown。

        Args:
            full_memory_list: 完整面试对话历史。
            jd_text: 岗位 JD 文本。
            resume_text: 候选人简历/背景文本，用于生成个性化示例回答。

        Returns:
            直接可渲染/下载的 Markdown 指导文本。
        """
        qa_pairs = self._extract_qa_pairs(full_memory_list)

        if not qa_pairs:
            return (
                "# 面试回答逐题指导\n\n"
                "当前没有足够的问答记录来生成指导，请先完成至少一轮面试问答。"
            )

        user_prompt = self._build_user_prompt(jd_text, resume_text, qa_pairs)

        try:
            guidance = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": GUIDANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
        except Exception as exc:
            self._debug_log(f"guidance generation failed: {exc}")
            return (
                "# 面试回答逐题指导\n\n"
                "很抱歉，指导生成过程中遇到问题，未能产出完整内容。\n\n"
                f"> 原因：{exc}\n\n"
                "> 建议：请稍后重试，或检查大模型服务是否可用。"
            )

        if not guidance.strip():
            return (
                "# 面试回答逐题指导\n\n"
                "模型返回内容为空，请稍后重试。"
            )

        return guidance

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_qa_pairs(
        full_memory_list: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """从对话历史中提取 (面试官提问, 候选人回答) 配对。

        规则：
        - 跳过第一条 assistant 消息（通常是开场白，没有对应的用户回答）。
        - 之后每条 assistant 消息与紧随其后的 user 消息配对。
        - assistant 消息的后面必须是 user 消息才构成有效配对。

        Args:
            full_memory_list: 完整面试对话历史。

        Returns:
            [{"question": "...", "answer": "..."}, ...] 列表。
        """
        qa_pairs: list[dict[str, str]] = []

        # 找到第一条 assistant 消息的索引，从之后开始配对
        first_question_idx = -1
        for i, msg in enumerate(full_memory_list):
            if msg.get("role") == "assistant":
                first_question_idx = i
                break

        if first_question_idx == -1:
            return qa_pairs

        # 从第一条 assistant 之后开始，配对 (assistant, user)
        i = first_question_idx
        while i < len(full_memory_list):
            msg = full_memory_list[i]
            if msg.get("role") == "assistant" and msg.get("content", "").strip():
                # 找紧随其后的 user 消息
                if i + 1 < len(full_memory_list):
                    next_msg = full_memory_list[i + 1]
                    if next_msg.get("role") == "user" and next_msg.get("content", "").strip():
                        qa_pairs.append({
                            "question": msg["content"].strip(),
                            "answer": next_msg["content"].strip(),
                        })
                        i += 2
                        continue
            i += 1

        return qa_pairs

    @staticmethod
    def _build_user_prompt(
        jd_text: str,
        resume_text: str,
        qa_pairs: list[dict[str, str]],
    ) -> str:
        """构建指导生成的 user prompt。

        Args:
            jd_text: 岗位 JD 文本。
            resume_text: 候选人简历/背景文本。
            qa_pairs: Q&A 配对列表。

        Returns:
            格式化后的 user prompt 字符串。
        """
        qa_text_parts: list[str] = []
        for idx, pair in enumerate(qa_pairs, start=1):
            qa_text_parts.append(
                f"第 {idx} 题：\n"
                f"面试官提问：{pair['question']}\n"
                f"候选人回答：{pair['answer']}\n"
            )

        resume_section = ""
        if resume_text.strip():
            resume_section = f"【候选人简历 / 背景】\n{resume_text}\n\n"

        return (
            "请对以下面试问答逐题进行指导分析。\n\n"
            f"【岗位 JD】\n{jd_text}\n\n"
            f"{resume_section}"
            "【面试问答记录】\n\n"
            + "\n".join(qa_text_parts)
        )

    @staticmethod
    def _debug_log(message: str) -> None:
        """统一的 debug 日志输出，与项目中其他智能体风格一致。"""
        print(f"[AnswerGuidanceAgent Debug] {message}")

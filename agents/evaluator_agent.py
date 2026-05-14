"""评估者智能体与报告生成者智能体。"""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from core.llm_client import LLMClient
from prompts.evaluator import (
    REPORT_SYSTEM_PROMPT_TEMPLATE,
    SCORING_SYSTEM_PROMPT,
    TRANSFER_ADVICE_NOT_REQUIRED,
    TRANSFER_ADVICE_REQUIRED,
)


SCORE_KEYS = ("专业深度", "表达逻辑", "应变能力", "岗位匹配度")


class ReportGeneratorAgent:
    """负责把结构化评分转换成可读 Markdown 报告。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化报告生成者。

        Args:
            llm_client: 大模型客户端实例。
        """
        self.llm_client = llm_client

    def generate_markdown_report(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
        score_payload: dict[str, int],
    ) -> str:
        """根据评分 JSON 生成最终 Markdown 报告。

        Args:
            full_memory_list: 完整面试对话。
            jd_text: 岗位 JD。
            score_payload: Step 1 得到的四项评分。

        Returns:
            可直接传给 st.markdown 渲染的 Markdown 文本。
        """
        job_fit_score = score_payload.get("岗位匹配度", 0)
        transfer_rule = (
            TRANSFER_ADVICE_REQUIRED
            if job_fit_score < 60
            else TRANSFER_ADVICE_NOT_REQUIRED
        )

        system_prompt = REPORT_SYSTEM_PROMPT_TEMPLATE.format(
            transfer_advice_rule=transfer_rule
        )
        user_prompt = (
            "请根据以下材料生成最终面试报告。\n\n"
            f"【岗位 JD】\n{jd_text}\n\n"
            "【评分 JSON】\n"
            f"{json.dumps(score_payload, ensure_ascii=False, indent=2)}\n\n"
            "【完整面试对话】\n"
            f"{json.dumps(full_memory_list, ensure_ascii=False, indent=2)}"
        )

        report = self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        if not report.strip():
            return "## 面试评估报告\n\n报告生成失败：模型返回内容为空，请稍后重试。"
        return report


class EvaluatorAgent:
    """负责先打分，再调用报告生成者输出最终报告。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化评估者智能体。

        Args:
            llm_client: 大模型客户端实例。
        """
        self.llm_client = llm_client
        self.report_generator = ReportGeneratorAgent(llm_client=llm_client)

    @staticmethod
    def _coerce_score(value: Any) -> int:
        """把模型返回的分数规整为 0-100 的整数。

        LLM 理论上会按提示词输出整数，但生产环境要防御字符串、
        浮点数或越界值，避免 Streamlit 页面因为类型问题崩溃。
        """
        score = int(float(value))
        return max(0, min(100, score))

    def _parse_score_json(self, raw_text: str) -> dict[str, int]:
        """解析 Step 1 返回的评分 JSON。

        这里使用 json.loads 保持结构化链路清晰：只有合法 JSON 才会进入
        Step 2 报告生成。解析失败时由上层捕获并返回友好错误报告。
        """
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("评分结果不是 JSON 对象")

        extra_keys = set(parsed) - set(SCORE_KEYS)
        if extra_keys:
            raise ValueError(f"评分 JSON 包含多余字段：{', '.join(extra_keys)}")

        normalized_scores: dict[str, int] = {}
        for key in SCORE_KEYS:
            if key not in parsed:
                raise ValueError(f"评分 JSON 缺少字段：{key}")
            normalized_scores[key] = self._coerce_score(parsed[key])

        return normalized_scores

    def _generate_score_json(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
    ) -> dict[str, int]:
        """Step 1：调用 LLM 生成四项结构化评分。"""
        user_prompt = (
            "请基于岗位 JD 和完整面试对话输出评分 JSON。\n\n"
            f"【岗位 JD】\n{jd_text}\n\n"
            "【完整面试对话】\n"
            f"{json.dumps(full_memory_list, ensure_ascii=False, indent=2)}"
        )

        raw_score_text = self.llm_client.chat(
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return self._parse_score_json(raw_score_text)

    def generate_evaluation_report(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
    ) -> str:
        """生成完整 Markdown 面试评估报告。

        这是评估流水线入口：
        - Step 1：评估者产出机器可解析的评分 JSON。
        - Step 2：报告生成者读取评分 JSON 和原始上下文，产出 Markdown。

        Args:
            full_memory_list: 完整面试对话历史。
            jd_text: 岗位 JD。

        Returns:
            Markdown 格式评估报告；异常时返回友好提示，不让页面白屏。
        """
        if not full_memory_list:
            return "## 面试评估报告\n\n当前没有可评估的面试对话，请先开始面试。"

        has_candidate_answer = any(
            message.get("role") == "user" and message.get("content", "").strip()
            for message in full_memory_list
        )
        if not has_candidate_answer:
            return "## 面试评估报告\n\n当前还没有候选人回答，无法形成有效评估。"

        try:
            score_payload = self._generate_score_json(
                full_memory_list=full_memory_list,
                jd_text=jd_text,
            )
            return self.report_generator.generate_markdown_report(
                full_memory_list=full_memory_list,
                jd_text=jd_text,
                score_payload=score_payload,
            )
        except (JSONDecodeError, ValueError, TypeError) as exc:
            print(f"[EvaluatorAgent Debug] score JSON parse failed: {exc}")
            return (
                "## 面试评估报告\n\n"
                "评分阶段没有返回合法 JSON，暂时无法生成完整报告。\n\n"
                f"> 友好提示：请检查模型是否遵守 JSON 输出规则。错误信息：{exc}"
            )
        except Exception as exc:
            print(f"[EvaluatorAgent Debug] unexpected error: {exc}")
            return (
                "## 面试评估报告\n\n"
                "报告生成过程中出现异常，但应用仍可继续使用。请稍后重试。\n\n"
                f"> 错误信息：{exc}"
            )

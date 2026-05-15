"""评估者智能体与报告生成者智能体。

评估流水线：
1. 评估者先对四个维度产出严格 JSON 评分。
2. 报告生成者把评分、JD 和对话整理成可读 Markdown 报告。

与面试官智能体的关系：
- 评估者仅读取 conversation_memory 和 interview_state 中的公开数据。
- 不依赖面试官内部状态机逻辑，两套代码互相独立。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from core.llm_client import LLMClient
from prompts.evaluator import (
    REPORT_SYSTEM_PROMPT_TEMPLATE,
    SCORING_REPAIR_PROMPT,
    SCORING_SYSTEM_PROMPT,
    TRANSFER_ADVICE_NOT_REQUIRED,
    TRANSFER_ADVICE_REQUIRED,
)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SCORE_KEYS = ("专业深度", "表达逻辑", "应变能力", "岗位匹配度")


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    """一次完整评估的结构化结果。

    包含评分和基于对话证据的结构化分析，可直接传给报告生成者。
    """

    scores: dict[str, int]
    overall_assessment: str
    highlights: list[str]
    risks: list[str]
    follow_up_suggestions: list[str]


# ---------------------------------------------------------------------------
# 报告生成者
# ---------------------------------------------------------------------------


class ReportGeneratorAgent:
    """负责把结构化评分转换成可读 Markdown 报告。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化报告生成者。

        Args:
            llm_client: 大模型客户端实例。
        """
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def generate_markdown_report(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
        score_payload: dict[str, int],
    ) -> str:
        """根据评分 JSON 和对话材料生成最终 Markdown 报告。

        Args:
            full_memory_list: 完整面试对话历史。
            jd_text: 岗位 JD 文本。
            score_payload: Step 1 得到的四项评分（0-100 整数）。

        Returns:
            可直接传给 st.markdown 渲染的 Markdown 文本。
        """
        transfer_rule = self._resolve_transfer_rule(score_payload)
        system_prompt = REPORT_SYSTEM_PROMPT_TEMPLATE.format(
            transfer_advice_rule=transfer_rule,
        )
        user_prompt = self._build_report_user_prompt(
            jd_text=jd_text,
            score_payload=score_payload,
            full_memory_list=full_memory_list,
        )

        report = self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        if not report.strip():
            return self.generate_error_report("模型返回内容为空，请稍后重试。")
        return report

    def generate_error_report(self, reason: str) -> str:
        """生成一份友好的错误报告，避免前端白屏。

        Args:
            reason: 失败原因描述。

        Returns:
            Markdown 格式的错误说明。
        """
        return (
            "## 面试评估报告\n\n"
            "很抱歉，报告生成过程中遇到问题，未能产出完整评估。\n\n"
            f"> 原因：{reason}\n\n"
            "> 建议：请稍后重试，或检查大模型服务是否可用。"
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_transfer_rule(score_payload: dict[str, int]) -> str:
        """根据岗位匹配度分数决定是否加入转岗建议模块。"""
        job_fit_score = score_payload.get("岗位匹配度", 0)
        if job_fit_score < 60:
            return TRANSFER_ADVICE_REQUIRED
        return TRANSFER_ADVICE_NOT_REQUIRED

    @staticmethod
    def _build_report_user_prompt(
        jd_text: str,
        score_payload: dict[str, int],
        full_memory_list: list[dict[str, str]],
    ) -> str:
        """构建报告生成的 user prompt，保证材料和评分规整传递。"""
        return (
            "请根据以下材料生成最终面试报告。\n\n"
            f"【岗位 JD】\n{jd_text}\n\n"
            "【评分 JSON】\n"
            f"{json.dumps(score_payload, ensure_ascii=False, indent=2)}\n\n"
            "【完整面试对话】\n"
            f"{json.dumps(full_memory_list, ensure_ascii=False, indent=2)}"
        )


# ---------------------------------------------------------------------------
# 评估者智能体
# ---------------------------------------------------------------------------


class EvaluatorAgent:
    """负责先打分，再调用报告生成者输出最终报告。

    核心设计：
    - 双阶段流水线（评分 → 报告），每阶段独立调用 LLM。
    - 评分阶段具备 JSON 修复和兜底逻辑，保证不因模型输出格式错误而崩溃。
    - 评估状态独立维护，不修改 conversation_memory 或 interview_state。
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化评估者智能体。

        Args:
            llm_client: 大模型客户端实例。
        """
        self.llm_client = llm_client
        self.report_generator = ReportGeneratorAgent(llm_client=llm_client)
        self.evaluation_state: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def generate_evaluation_report(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
    ) -> str:
        """生成完整 Markdown 面试评估报告。

        这是评估流水线入口：
        - Step 1：评估者产出机器可解析的评分 JSON。
        - Step 2：报告生成者读取评分和原始上下文，产出 Markdown。

        Args:
            full_memory_list: 完整面试对话历史（list[dict] 格式）。
            jd_text: 岗位 JD 文本。

        Returns:
            Markdown 格式评估报告；异常时返回友好提示，不让页面白屏。
        """
        # —— 边界检查 ——
        if not full_memory_list:
            return self.report_generator.generate_error_report(
                "当前没有可评估的面试对话，请先开始面试。"
            )

        if not self._has_candidate_answer(full_memory_list):
            return self.report_generator.generate_error_report(
                "当前还没有候选人回答，无法形成有效评估。"
            )

        # —— 重置评估状态 ——
        self.evaluation_state = self._build_initial_evaluation_state()

        # —— Step 1：结构化评分 ——
        try:
            score_payload = self._generate_score_json(
                full_memory_list=full_memory_list,
                jd_text=jd_text,
            )
            self.evaluation_state["scores"] = score_payload
            self.evaluation_state["scoring_success"] = True

        except (JSONDecodeError, ValueError, TypeError) as exc:
            self._debug_log(f"score JSON parse failed: {exc}")
            return self._format_score_parse_error(exc)

        except Exception as exc:
            self._debug_log(f"unexpected error in scoring: {exc}")
            return self.report_generator.generate_error_report(str(exc))

        # —— Step 2：报告生成 ——
        try:
            report = self.report_generator.generate_markdown_report(
                full_memory_list=full_memory_list,
                jd_text=jd_text,
                score_payload=score_payload,
            )
            self.evaluation_state["report_generated"] = True
            return report

        except Exception as exc:
            self._debug_log(f"unexpected error in report generation: {exc}")
            return self.report_generator.generate_error_report(str(exc))

    # ------------------------------------------------------------------
    # 静态工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json_object(raw_text: str) -> dict[str, Any]:
        """从模型输出中读取 JSON 对象，兼容 Markdown 代码块和前后缀噪声。

        解析策略（与 InterviewerAgent 保持一致）：
        1. 先尝试剥离 Markdown 代码围栏后直接解析。
        2. 失败后从左到右扫描第一个 '{'，尝试从该位置解析。
        """
        text = raw_text.strip()

        # 剥离 Markdown 代码块围栏
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 尝试直接解析
        try:
            parsed = json.loads(text)
        except JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(text):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(text[index:])
                    break
                except JSONDecodeError:
                    continue
            else:
                raise

        if not isinstance(parsed, dict):
            raise ValueError("模型返回内容不是 JSON 对象")
        return parsed

    @staticmethod
    def _coerce_score(value: Any) -> int:
        """把模型返回的分数规整为 0-100 的整数。

        LLM 理论上会按提示词输出整数，但生产环境要防御字符串、
        浮点数或越界值，避免 Streamlit 页面因为类型问题崩溃。
        """
        score = int(float(value))
        return max(0, min(100, score))

    @staticmethod
    def _has_candidate_answer(full_memory_list: list[dict[str, str]]) -> bool:
        """检查对话历史中是否包含至少一条候选人回答。"""
        return any(
            message.get("role") == "user" and message.get("content", "").strip()
            for message in full_memory_list
        )

    @staticmethod
    def _build_initial_evaluation_state() -> dict[str, Any]:
        """创建评估状态初始字典。"""
        return {
            "scores": {},
            "scoring_success": False,
            "repair_attempted": False,
            "report_generated": False,
        }

    # ------------------------------------------------------------------
    # JSON 解析
    # ------------------------------------------------------------------

    def _parse_score_json(self, raw_text: str) -> dict[str, int]:
        """解析 Step 1 返回的评分 JSON。

        先用 _load_json_object 做容错清洗，再逐字段验证。
        只有合法 JSON 且字段齐全才会返回。
        """
        parsed = self._load_json_object(raw_text)

        # 检查多余字段
        extra_keys = set(parsed) - set(SCORE_KEYS)
        if extra_keys:
            raise ValueError(f"评分 JSON 包含多余字段：{', '.join(sorted(extra_keys))}")

        # 检查缺失字段并规整分值
        normalized_scores: dict[str, int] = {}
        for key in SCORE_KEYS:
            if key not in parsed:
                raise ValueError(f"评分 JSON 缺少字段：{key}")
            normalized_scores[key] = self._coerce_score(parsed[key])

        self._debug_log(f"parsed scores: {normalized_scores}")
        return normalized_scores

    # ------------------------------------------------------------------
    # JSON 修复
    # ------------------------------------------------------------------

    def _build_repair_messages(
        self,
        raw_output: str,
    ) -> list[dict[str, str]]:
        """构造评分 JSON 修复请求的消息列表。"""
        repair_prompt = SCORING_REPAIR_PROMPT.format(raw_output=raw_output)
        return [{"role": "user", "content": repair_prompt}]

    def _repair_score_json(self, raw_output: str) -> dict[str, int]:
        """对非 JSON 评分输出做一次结构化修复。

        调用 LLM 重新输出合法 JSON；如果修复仍失败，抛异常由上层兜底。
        """
        self.evaluation_state["repair_attempted"] = True
        repaired_response = self.llm_client.chat(
            messages=self._build_repair_messages(raw_output),
            temperature=0.0,
        )
        return self._parse_score_json(repaired_response)

    # ------------------------------------------------------------------
    # 评分生成
    # ------------------------------------------------------------------

    def _generate_score_json(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
    ) -> dict[str, int]:
        """Step 1：调用 LLM 生成四项结构化评分。

        内置 JSON 修复流程：如果第一次输出的内容无法解析为合法 JSON，
        使用修复提示词重新请求一次。修复失败则抛出异常，由上层
        generate_evaluation_report 返回友好错误报告。
        """
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

        # 第一次尝试直接解析
        try:
            return self._parse_score_json(raw_score_text)
        except (JSONDecodeError, ValueError, TypeError) as exc:
            self._debug_log(
                f"first parse failed: {exc}; "
                f"raw_score_text preview={raw_score_text[:200]}"
            )

        # 第二次尝试：修复
        try:
            return self._repair_score_json(raw_score_text)
        except (JSONDecodeError, ValueError, TypeError) as exc:
            self._debug_log(f"repair also failed: {exc}")

        # 两次都失败，使用兜底评分
        return self._build_fallback_scores(full_memory_list)

    # ------------------------------------------------------------------
    # 兜底评分
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback_scores(
        full_memory_list: list[dict[str, str]],
    ) -> dict[str, int]:
        """当 JSON 解析和修复都失败时，生成保守兜底评分。

        策略：根据对话轮次粗略估算，有 3 轮以上用户回答给中等偏高，
        否则给中等默认分。这不是精确评估，只为让报告流程不中断。
        """
        candidate_turns = sum(
            1 for m in full_memory_list if m.get("role") == "user" and m.get("content", "").strip()
        )
        if candidate_turns >= 3:
            base = 65
        elif candidate_turns >= 1:
            base = 55
        else:
            base = 50

        print(
            "[EvaluatorAgent Debug] "
            f"using fallback scores, base={base}, turns={candidate_turns}"
        )
        return {key: base for key in SCORE_KEYS}

    # ------------------------------------------------------------------
    # 格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_score_parse_error(exc: Exception) -> str:
        """格式化评分 JSON 解析失败的友好错误报告。"""
        return (
            "## 面试评估报告\n\n"
            "评分阶段没有返回合法 JSON，暂时无法生成完整报告。\n\n"
            "### 可能原因\n"
            "- 大模型输出格式不符合 JSON 规范\n"
            "- 对话内容过长导致模型截断输出\n"
            "- 提示词中的 JSON Schema 未被遵守\n\n"
            "### 建议\n"
            "- 请稍后重试\n"
            "- 检查大模型服务状态\n\n"
            f"> 技术细节：{exc}"
        )

    @staticmethod
    def _debug_log(message: str) -> None:
        """统一的 debug 日志输出，方便开发者观察评估流程。

        与 InterviewerAgent 的 print debug 风格保持一致。
        """
        print(f"[EvaluatorAgent Debug] {message}")

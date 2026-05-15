"""面试官智能体。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from core.llm_client import LLMClient
from prompts.interviewer import INTERVIEWER_SYSTEM_PROMPT_TEMPLATE


ACTION_DECISIONS = {
    "next_question",
    "drill_down",
    "broaden",
    "simplify",
    "finish_interview",
}
ANSWER_QUALITIES = {
    "清晰完整",
    "缺乏细节",
    "自相矛盾",
    "偏离问题",
    "信息不足",
}
STATE_LIST_KEYS = ("covered_dimensions", "pending_dimensions")


@dataclass(frozen=True)
class InterviewTurnResult:
    """面试官一轮状态机执行结果。"""

    reply_text: str
    action_decision: str
    answer_quality: str
    should_finish: bool
    finish_reason: str = ""


class InterviewerAgent:
    """负责完成面试中的“感知 -> 决策 -> 行动 -> 反馈”闭环。"""

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_memory: list[dict[str, str]],
        interview_state: dict[str, Any],
        jd_text: str,
        persona_style: str,
        resume_text: str = "",
    ) -> None:
        """初始化面试官智能体。

        Args:
            llm_client: 大模型客户端实例。
            conversation_memory: Streamlit session_state 中的共享对话列表。
            interview_state: 本场面试的结构化状态机状态。
            jd_text: 当前岗位 JD。
            persona_style: 面试风格描述。
            resume_text: 候选人简历或背景信息，可为空。
        """
        self.llm_client = llm_client
        self.conversation_memory = conversation_memory
        self.interview_state = interview_state
        self.jd_text = jd_text
        self.persona_style = persona_style
        self.resume_text = resume_text
        self._ensure_state_defaults()

    @staticmethod
    def build_initial_state() -> dict[str, Any]:
        """创建一场新面试的状态机初始状态。"""
        return {
            "turn_count": 0,
            "covered_dimensions": [],
            "pending_dimensions": [],
            "current_focus": "",
            "coverage_summary": "",
            "last_answer_quality": "",
            "last_action_decision": "",
            "finish_reason": "",
        }

    def _ensure_state_defaults(self) -> None:
        """补齐旧 session 中缺失的状态字段。"""
        defaults = self.build_initial_state()
        for key, value in defaults.items():
            self.interview_state.setdefault(key, value.copy() if isinstance(value, list) else value)

    def _build_messages(self) -> list[dict[str, str]]:
        """组装发送给 LLM 的消息列表。

        System Prompt 负责规定 JSON 输出格式；历史消息负责提供上下文。
        这里不把 JSON 规则散落在业务代码里，便于后续维护提示词。
        """
        system_prompt = INTERVIEWER_SYSTEM_PROMPT_TEMPLATE.format(
            jd_text=self.jd_text,
            resume_text=self.resume_text or "未提供简历内容。",
            persona_style=self.persona_style,
            interview_state=json.dumps(
                self.interview_state,
                ensure_ascii=False,
                indent=2,
            ),
        )
        return [
            {"role": "system", "content": system_prompt},
            *self.conversation_memory,
        ]

    @staticmethod
    def _load_json_object(raw_text: str) -> dict[str, Any]:
        """从模型输出中读取 JSON 对象，兼容少量代码块或前后缀噪声。"""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

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

    def _parse_reply_json(self, raw_text: str) -> dict[str, Any]:
        """解析面试官模型返回的 JSON 字符串。

        状态机要求每轮输出决策、可见回复和最新覆盖状态。解析阶段会做
        字段校验和轻量归一化，避免无效动作进入前端状态切换。
        """
        parsed = self._load_json_object(raw_text)
        required_keys = {
            "answer_quality",
            "action_decision",
            "interview_state",
            "reply_text",
        }

        missing_keys = required_keys - set(parsed)
        if missing_keys:
            raise ValueError(f"模型返回 JSON 缺少字段：{', '.join(missing_keys)}")

        extra_keys = set(parsed) - required_keys
        if extra_keys:
            raise ValueError(f"模型返回 JSON 包含多余字段：{', '.join(extra_keys)}")

        answer_quality = str(parsed["answer_quality"])
        action_decision = str(parsed["action_decision"])
        if answer_quality not in ANSWER_QUALITIES:
            raise ValueError(f"无效 answer_quality：{answer_quality}")
        if action_decision not in ACTION_DECISIONS:
            raise ValueError(f"无效 action_decision：{action_decision}")

        state_payload = parsed["interview_state"]
        if not isinstance(state_payload, dict):
            raise ValueError("interview_state 必须是 JSON 对象")

        normalized_state = self._normalize_state_payload(state_payload)

        return {
            "answer_quality": answer_quality,
            "action_decision": action_decision,
            "interview_state": normalized_state,
            "reply_text": str(parsed["reply_text"]),
        }

    @staticmethod
    def _normalize_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
        """规整模型返回的状态字段，只保留状态机需要的内容。"""
        normalized: dict[str, Any] = {}
        for key in STATE_LIST_KEYS:
            value = state_payload.get(key, [])
            if isinstance(value, list):
                normalized[key] = [str(item).strip() for item in value if str(item).strip()]
            elif str(value).strip():
                normalized[key] = [str(value).strip()]
            else:
                normalized[key] = []

        for key in ("current_focus", "coverage_summary", "finish_reason"):
            normalized[key] = str(state_payload.get(key, "")).strip()

        return normalized

    @staticmethod
    def _extract_plain_reply(raw_text: str) -> str:
        """从非 JSON 模型输出中提取可直接展示的追问。"""
        reply_text = raw_text.strip().strip('"')
        if not reply_text:
            return ""

        if reply_text.startswith(("模型调用失败", "{", "[", "```")):
            return ""

        internal_markers = ("answer_quality", "action_decision", "reply_text")
        if any(marker in reply_text for marker in internal_markers):
            return ""

        if len(reply_text) > 600:
            return ""

        question_end_positions = [
            reply_text.find(mark)
            for mark in ("？", "?")
            if reply_text.find(mark) != -1
        ]
        if question_end_positions:
            first_question_end = min(question_end_positions) + 1
            reply_text = reply_text[:first_question_end].strip()

        return reply_text

    @staticmethod
    def _normalize_reply_text(reply_text: str, action_decision: str) -> str:
        """限制普通追问为单个候选人可回答的问题。"""
        normalized = reply_text.strip().strip('"')
        if action_decision == "finish_interview":
            return normalized

        question_end_positions = [
            normalized.find(mark)
            for mark in ("？", "?")
            if normalized.find(mark) != -1
        ]
        if question_end_positions:
            first_question_end = min(question_end_positions) + 1
            normalized = normalized[:first_question_end].strip()
        return normalized

    @staticmethod
    def _build_fallback_reply(user_text: str) -> str:
        """构造兜底追问，避免把模型解析失败暴露给候选人。"""
        if len(user_text) >= 40:
            return (
                "你刚才介绍的这个项目里，最关键的一次技术取舍是什么？"
            )
        return (
            "结合你刚才的回答，你认为最能体现你岗位匹配度的一个细节是什么？"
        )

    def _build_repair_messages(self, raw_response: str) -> list[dict[str, str]]:
        """构造一次 JSON 修复请求，尽量保留模型已经生成的追问。"""
        plain_reply = self._extract_plain_reply(raw_response)
        repair_prompt = (
            "请把上一条模型输出修复成合法 JSON。只能输出 JSON，禁止输出解释文字。\n"
            "如果原始输出已经是面向候选人的追问，把它放入 reply_text。\n"
            "action_decision 必须是 next_question、drill_down、broaden、simplify、"
            "finish_interview 之一。除非面试已经充分覆盖 JD、简历和对话中的主要能力点，"
            "否则不要使用 finish_interview。\n"
            "JSON Schema：\n"
            "{\n"
            '  "answer_quality": "清晰完整 / 缺乏细节 / 自相矛盾 / 偏离问题 / 信息不足",\n'
            '  "action_decision": "next_question / drill_down / broaden / simplify / finish_interview",\n'
            '  "interview_state": {\n'
            '    "covered_dimensions": ["已覆盖能力点"],\n'
            '    "pending_dimensions": ["仍需验证能力点"],\n'
            '    "current_focus": "当前关注点",\n'
            '    "coverage_summary": "覆盖情况摘要",\n'
            '    "finish_reason": "结束原因；未结束时为空字符串"\n'
            "  },\n"
            '  "reply_text": "对候选人说的具体话术"\n'
            "}\n\n"
            f"【当前状态】\n{json.dumps(self.interview_state, ensure_ascii=False, indent=2)}\n\n"
            f"【原始输出】\n{raw_response}\n\n"
            f"【可用追问文本】\n{plain_reply}"
        )
        return [{"role": "user", "content": repair_prompt}]

    def _repair_reply_json(self, raw_response: str) -> dict[str, Any]:
        """对非 JSON 输出做一次结构化修复。"""
        repaired_response = self.llm_client.chat(
            messages=self._build_repair_messages(raw_response),
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        return self._parse_reply_json(repaired_response)

    def _apply_state_payload(
        self,
        state_payload: dict[str, Any],
        answer_quality: str,
        action_decision: str,
    ) -> None:
        """把本轮模型状态写回持久化状态机。"""
        self.interview_state["turn_count"] = int(self.interview_state.get("turn_count", 0)) + 1
        for key in STATE_LIST_KEYS:
            self.interview_state[key] = state_payload.get(key, [])
        for key in ("current_focus", "coverage_summary", "finish_reason"):
            self.interview_state[key] = state_payload.get(key, "")
        self.interview_state["last_answer_quality"] = answer_quality
        self.interview_state["last_action_decision"] = action_decision

    def _can_finish(self) -> bool:
        """避免模型过早结束面试。"""
        turn_count = int(self.interview_state.get("turn_count", 0))
        covered_count = len(self.interview_state.get("covered_dimensions", []))
        pending_count = len(self.interview_state.get("pending_dimensions", []))
        return turn_count >= 6 and (covered_count >= 4 or pending_count == 0)

    def _coerce_premature_finish(self, reply_text: str) -> tuple[str, str]:
        """过早 finish 时退回到扩展追问，保持面试继续。"""
        self.interview_state["finish_reason"] = ""
        fallback_reply = self._extract_plain_reply(reply_text)
        if not fallback_reply or "结束" in fallback_reply:
            fallback_reply = "我们再补充验证一个关键能力点：结合这个岗位要求，你认为你最需要进一步证明的工程能力是什么？"
        return "broaden", fallback_reply

    def generate_turn(self, user_text: str) -> InterviewTurnResult:
        """根据候选人最新回答执行一轮面试状态机。

        数据流转说明：
        1. 先把候选人回答写入 conversation_memory，让 LLM 能看到最新输入。
        2. 解析 LLM 输出的状态机 JSON，必要时做一次 JSON 修复。
        3. 将状态写回 interview_state，前端据此判断是否结束面试。
        4. 将候选人可见 reply_text 写回 conversation_memory。
        """
        cleaned_user_text = user_text.strip()
        if not cleaned_user_text:
            return InterviewTurnResult(
                reply_text="我没有收到有效回答，请补充你的想法后再继续。",
                action_decision="simplify",
                answer_quality="信息不足",
                should_finish=False,
            )

        self.conversation_memory.append(
            {"role": "user", "content": cleaned_user_text}
        )

        raw_response = self.llm_client.chat(
            messages=self._build_messages(),
            temperature=0.4,
            response_format={"type": "json_object"},
            max_tokens=1600,
        )

        try:
            decision_payload = self._parse_reply_json(raw_response)

        except (JSONDecodeError, ValueError, TypeError) as exc:
            print(
                "[InterviewerAgent Debug] JSON parse failed: "
                f"{exc}; raw_response={raw_response}"
            )
            try:
                decision_payload = self._repair_reply_json(raw_response)
                print("[InterviewerAgent Debug] repaired non-JSON reply into state JSON")
            except (JSONDecodeError, ValueError, TypeError) as repair_exc:
                print(
                    "[InterviewerAgent Debug] JSON repair failed: "
                    f"{repair_exc}"
                )
                plain_reply_text = self._extract_plain_reply(raw_response)
                if plain_reply_text:
                    reply_text = plain_reply_text
                else:
                    reply_text = self._build_fallback_reply(cleaned_user_text)
                decision_payload = {
                    "answer_quality": "信息不足",
                    "action_decision": "drill_down",
                    "interview_state": self._normalize_state_payload(self.interview_state),
                    "reply_text": reply_text,
                }

        answer_quality = decision_payload["answer_quality"]
        action_decision = decision_payload["action_decision"]
        reply_text = self._normalize_reply_text(
            decision_payload["reply_text"],
            action_decision,
        )

        if not reply_text:
            reply_text = self._build_fallback_reply(cleaned_user_text)
            action_decision = "drill_down"

        self._apply_state_payload(
            state_payload=decision_payload["interview_state"],
            answer_quality=answer_quality,
            action_decision=action_decision,
        )

        if action_decision == "finish_interview" and not self._can_finish():
            action_decision, reply_text = self._coerce_premature_finish(reply_text)
            self.interview_state["last_action_decision"] = action_decision

        should_finish = action_decision == "finish_interview"

        # action_decision 是面试官内部决策结果，只打印到后台供开发者观察，
        # 不展示给候选人，避免影响真实面试体验。
        print(
            "[InterviewerAgent Debug] "
            f"answer_quality={answer_quality} | "
            f"action_decision={action_decision} | "
            f"turn_count={self.interview_state['turn_count']}"
        )

        self.conversation_memory.append(
            {"role": "assistant", "content": reply_text}
        )
        return InterviewTurnResult(
            reply_text=reply_text,
            action_decision=action_decision,
            answer_quality=answer_quality,
            should_finish=should_finish,
            finish_reason=str(self.interview_state.get("finish_reason", "")),
        )

    def generate_reply(self, user_text: str) -> str:
        """兼容旧调用方：只返回候选人可见回复。"""
        return self.generate_turn(user_text=user_text).reply_text

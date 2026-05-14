"""MockInterviewAgent 的 Streamlit 网页入口。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agents.evaluator_agent import EvaluatorAgent
from agents.interviewer_agent import InterviewerAgent
from core.llm_client import LLMClient
from core.memory import ConversationMemory
from prompts.interviewer import INTERVIEWER_PERSONAS
from utils.mock_data import DEFAULT_JOB_DESCRIPTION, DEFAULT_RESUME


def initialize_session_state() -> None:
    """初始化 Streamlit 会话状态，避免页面刷新导致上下文丢失。"""
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = LLMClient()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "selected_persona" not in st.session_state:
        st.session_state.selected_persona = next(iter(INTERVIEWER_PERSONAS))

    if "interviewer_agent" not in st.session_state:
        st.session_state.interviewer_agent = build_interviewer_agent(
            st.session_state.selected_persona
        )

    if "evaluator_agent" not in st.session_state:
        st.session_state.evaluator_agent = EvaluatorAgent(
            llm_client=st.session_state.llm_client
        )

    if "chat_records" not in st.session_state:
        st.session_state.chat_records = []

    if "report" not in st.session_state:
        st.session_state.report = ""


def build_interviewer_agent(persona_name: str) -> InterviewerAgent:
    """根据面试风格创建面试官智能体。

    Args:
        persona_name: 面试风格名称。

    Returns:
        配置好提示词的面试官智能体实例。
    """
    persona_prompt = INTERVIEWER_PERSONAS[persona_name]
    return InterviewerAgent(
        llm_client=st.session_state.llm_client,
        memory=st.session_state.memory,
        persona_prompt=persona_prompt,
    )


def reset_interviewer_when_persona_changed(persona_name: str) -> None:
    """当用户切换面试风格时，刷新面试官配置但保留历史记忆。"""
    if persona_name != st.session_state.selected_persona:
        st.session_state.selected_persona = persona_name
        st.session_state.interviewer_agent = build_interviewer_agent(persona_name)


def render_sidebar() -> bool:
    """渲染侧边栏配置区。

    Returns:
        是否点击了生成报告按钮。
    """
    with st.sidebar:
        st.header("面试设置")

        persona_names = list(INTERVIEWER_PERSONAS.keys())
        selected_persona = st.selectbox(
            "面试风格",
            options=persona_names,
            index=persona_names.index(st.session_state.selected_persona),
        )
        reset_interviewer_when_persona_changed(selected_persona)

        st.divider()
        st.subheader("模拟材料")
        st.text_area("候选人简历", DEFAULT_RESUME, height=180, disabled=True)
        st.text_area("岗位 JD", DEFAULT_JOB_DESCRIPTION, height=180, disabled=True)

        st.divider()
        generate_report = st.button(
            "结束面试并生成报告",
            type="primary",
            use_container_width=True,
        )

        if st.button("清空当前会话", use_container_width=True):
            clear_session()
            st.rerun()

    return generate_report


def clear_session() -> None:
    """清空面试上下文和页面展示记录。"""
    st.session_state.memory.clear()
    st.session_state.chat_records = []
    st.session_state.report = ""
    st.session_state.interviewer_agent = build_interviewer_agent(
        st.session_state.selected_persona
    )


def render_chat_history(chat_records: list[dict[str, str]]) -> None:
    """渲染历史聊天记录。

    Args:
        chat_records: 用于前端展示的聊天记录列表。
    """
    for record in chat_records:
        with st.chat_message(record["role"]):
            st.markdown(record["content"])


def stream_text(text: str) -> Any:
    """将完整文本包装成 Streamlit 可消费的生成器。

    Args:
        text: 需要展示的文本内容。

    Yields:
        分片文本。当前 LLMClient 为非流式封装，因此这里按词做轻量模拟。
    """
    for chunk in text.split(" "):
        yield chunk + " "


def handle_user_reply(user_reply: str) -> None:
    """处理候选人的输入，并生成面试官回复。

    Args:
        user_reply: 候选人在聊天输入框提交的回答。
    """
    st.session_state.chat_records.append(
        {"role": "user", "content": user_reply}
    )

    with st.chat_message("user"):
        st.markdown(user_reply)

    with st.chat_message("assistant"):
        answer = st.session_state.interviewer_agent.generate_question(user_reply)
        if hasattr(st, "write_stream"):
            displayed_answer = st.write_stream(stream_text(answer))
        else:
            st.markdown(answer)
            displayed_answer = answer

    st.session_state.chat_records.append(
        {"role": "assistant", "content": str(displayed_answer)}
    )


def handle_report_generation() -> None:
    """调用评估智能体生成并展示 Markdown 面试报告。"""
    history = st.session_state.memory.get_history()
    st.session_state.report = st.session_state.evaluator_agent.generate_report(
        history
    )


def main() -> None:
    """Streamlit 应用主函数。"""
    st.set_page_config(
        page_title="MockInterviewAgent",
        layout="wide",
    )
    initialize_session_state()

    st.title("MockInterviewAgent")
    st.caption("原生 Python + OpenAI SDK 驱动的 AI 模拟面试智能体")

    should_generate_report = render_sidebar()
    if should_generate_report:
        handle_report_generation()

    render_chat_history(st.session_state.chat_records)

    if not st.session_state.chat_records:
        with st.chat_message("assistant"):
            welcome_text = (
                "你好，我是本轮模拟面试官。请先用 1 分钟介绍一下你的项目经历，"
                "我会根据你的回答继续追问。"
            )
            st.markdown(welcome_text)

    user_reply = st.chat_input("请输入你的面试回答...")
    if user_reply:
        handle_user_reply(user_reply)

    if st.session_state.report:
        st.divider()
        st.subheader("面试评估报告")
        st.markdown(st.session_state.report)


if __name__ == "__main__":
    main()

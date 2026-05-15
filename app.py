"""MockInterviewAgent 的 Streamlit 网页入口。"""

from __future__ import annotations

import streamlit as st

from agents.evaluator_agent import EvaluatorAgent
from agents.interviewer_agent import InterviewerAgent
from core.llm_client import LLMClient
from prompts.interviewer import INTERVIEWER_PERSONAS
from utils.mock_data import DEFAULT_JOB_DESCRIPTION


STATUS_NOT_STARTED = "未开始"
STATUS_IN_PROGRESS = "进行中"
STATUS_FINISHED = "已结束"


def initialize_session_state() -> None:
    """初始化 Streamlit 会话状态。

    Streamlit 每次交互都会从头执行脚本，因此跨轮对话必须放在
    st.session_state 中。这里显式持久化三个核心状态：
    - interviewer_agent：面试官智能体实例。
    - conversation_memory：对话历史列表，供前端渲染和 LLM 上下文复用。
    - interview_state：结构化面试状态机，记录覆盖能力点和结束决策。
    - jd_text / resume_text：用于生成全局面试地图的岗位和候选人背景。
    - interview_status：控制页面在“进行中/已结束”等状态之间切换。
    """
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = LLMClient()

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []

    if "interview_state" not in st.session_state:
        st.session_state.interview_state = InterviewerAgent.build_initial_state()

    if "interview_status" not in st.session_state:
        st.session_state.interview_status = STATUS_NOT_STARTED

    if "jd_text" not in st.session_state:
        st.session_state.jd_text = DEFAULT_JOB_DESCRIPTION

    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""

    if "persona_name" not in st.session_state:
        st.session_state.persona_name = "温柔引导型"
    elif st.session_state.persona_name not in INTERVIEWER_PERSONAS:
        st.session_state.persona_name = "温柔引导型"

    if "final_report" not in st.session_state:
        st.session_state.final_report = ""

    if "final_pdf" not in st.session_state:
        st.session_state.final_pdf = None

    if "interviewer_agent" not in st.session_state or not hasattr(
        st.session_state.interviewer_agent,
        "generate_turn",
    ):
        st.session_state.interviewer_agent = build_interviewer_agent()

    if "evaluator_agent" not in st.session_state or not hasattr(
        st.session_state.evaluator_agent,
        "generate_evaluation_report",
    ):
        st.session_state.evaluator_agent = EvaluatorAgent(
            llm_client=st.session_state.llm_client
        )


def build_interviewer_agent() -> InterviewerAgent:
    """根据当前 JD、风格和共享记忆创建面试官智能体。"""
    return InterviewerAgent(
        llm_client=st.session_state.llm_client,
        conversation_memory=st.session_state.conversation_memory,
        interview_state=st.session_state.interview_state,
        jd_text=st.session_state.jd_text,
        resume_text=st.session_state.resume_text,
        persona_style=INTERVIEWER_PERSONAS[st.session_state.persona_name],
    )


def start_interview(jd_text: str, resume_text: str, persona_name: str) -> None:
    """开始一场新的面试。

    开始面试时会清空旧历史和旧报告，并重新创建面试官智能体。
    这样可以保证新的 JD 和面试风格不会被上一场会话污染。
    """
    st.session_state.jd_text = jd_text.strip() or DEFAULT_JOB_DESCRIPTION
    st.session_state.resume_text = resume_text.strip()
    st.session_state.persona_name = persona_name
    st.session_state.conversation_memory = []
    st.session_state.interview_state = InterviewerAgent.build_initial_state()
    st.session_state.final_report = ""
    st.session_state.final_pdf = None
    st.session_state.interview_status = STATUS_IN_PROGRESS
    st.session_state.interviewer_agent = build_interviewer_agent()

    # 首轮开场问题直接写入记忆。后续所有问答都由 interviewer_agent
    # 继续追加到同一份 conversation_memory，实现前端和智能体共享上下文。
    st.session_state.conversation_memory.append(
        {
            "role": "assistant",
            "content": (
                "你好，我们开始本轮模拟面试。请先用 1 分钟介绍一个"
                "与你申请岗位最相关的项目。"
            ),
        }
    )


def finish_interview() -> None:
    """结束当前面试，并切换到报告生成状态。"""
    st.session_state.interview_status = STATUS_FINISHED
    st.session_state.final_report = ""
    st.session_state.final_pdf = None


def finish_interview_from_agent(finish_reason: str) -> None:
    """根据面试官状态机决策结束面试。"""
    st.session_state.interview_status = STATUS_FINISHED
    st.session_state.final_report = ""
    st.session_state.final_pdf = None
    if finish_reason:
        st.session_state.interview_state["finish_reason"] = finish_reason


def render_sidebar() -> None:
    """渲染侧边栏设置区，并处理开始/结束按钮事件。"""
    with st.sidebar:
        st.header("面试设置")

        jd_text = st.text_area(
            "岗位 JD",
            value=st.session_state.jd_text,
            height=240,
            placeholder="请输入本次模拟面试的岗位 JD...",
        )

        resume_text = st.text_area(
            "简历 / 候选人背景",
            value=st.session_state.resume_text,
            height=180,
            placeholder="可粘贴简历、项目经历或候选人背景信息...",
        )

        persona_names = list(INTERVIEWER_PERSONAS.keys())
        persona_name = st.selectbox(
            "面试风格",
            options=persona_names,
            index=persona_names.index(st.session_state.persona_name),
        )

        st.divider()
        if st.button("开始面试", type="primary", use_container_width=True):
            start_interview(
                jd_text=jd_text,
                resume_text=resume_text,
                persona_name=persona_name,
            )
            st.rerun()

        if st.button(
            "结束面试并生成报告",
            use_container_width=True,
            disabled=st.session_state.interview_status != STATUS_IN_PROGRESS,
        ):
            finish_interview()
            st.rerun()

        st.caption(f"当前状态：{st.session_state.interview_status}")


def render_conversation_memory() -> None:
    """把 conversation_memory 渲染为 Streamlit 聊天消息。"""
    for message in st.session_state.conversation_memory:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_answer(user_text: str) -> None:
    """处理候选人输入，并触发面试官智能体追问。

    app.py 不直接拼 Prompt、不解析 JSON，只负责把用户输入交给智能体。
    JSON 解析、action_decision 调试打印、reply_text 写回记忆都在
    InterviewerAgent.generate_turn() 内完成。
    """
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("面试官正在分析你的回答..."):
            turn_result = st.session_state.interviewer_agent.generate_turn(
                user_text=user_text
            )
            reply_text = turn_result.reply_text
        st.markdown(reply_text)

    if turn_result.should_finish:
        finish_interview_from_agent(turn_result.finish_reason)
        st.rerun()


def render_in_progress_view() -> None:
    """渲染进行中的面试聊天区。"""
    render_conversation_memory()

    user_text = st.chat_input("请输入你的面试回答...")
    if user_text:
        handle_user_answer(user_text)


def render_finished_view() -> None:
    """渲染已结束状态。

    面试结束后主聊天区不渲染 conversation_memory，
    而是调用 evaluator_agent 生成 Markdown 报告和 PDF 报告。
    final_report 和 final_pdf 会被缓存到 session_state，避免重复调用模型。
    """
    st.subheader("面试评估报告")

    if not st.session_state.final_report:
        with st.spinner("正在完成评分并生成报告..."):
            st.session_state.final_report = (
                st.session_state.evaluator_agent.generate_evaluation_report(
                    full_memory_list=st.session_state.conversation_memory,
                    jd_text=st.session_state.jd_text,
                    interview_state=st.session_state.interview_state,
                )
            )
            # 从评估者状态中取出 PDF
            pdf_bytes = st.session_state.evaluator_agent.get_pdf_bytes()
            if pdf_bytes:
                st.session_state.final_pdf = pdf_bytes

    # 下载按钮行
    col_md, col_pdf = st.columns(2)
    with col_md:
        st.download_button(
            label="下载 Markdown 报告",
            data=st.session_state.final_report,
            file_name="面试评估报告.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_pdf:
        if st.session_state.final_pdf:
            st.download_button(
                label="下载 PDF 报告",
                data=st.session_state.final_pdf,
                file_name="面试评估报告.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            pdf_error = st.session_state.evaluator_agent.get_pdf_error()
            st.warning(f"PDF 生成失败：{pdf_error}" if pdf_error else "PDF 生成失败，请重试")

    st.divider()
    st.markdown(st.session_state.final_report)


def render_not_started_view() -> None:
    """渲染未开始状态。"""
    st.info("请在左侧填写岗位 JD、选择面试风格，然后点击“开始面试”。")


def main() -> None:
    """Streamlit 应用主函数。"""
    st.set_page_config(page_title="MockInterviewAgent", layout="wide")
    initialize_session_state()

    st.title("MockInterviewAgent")
    st.caption("原生 Python + OpenAI SDK + Streamlit 的 AI 模拟面试系统")

    render_sidebar()

    if st.session_state.interview_status == STATUS_IN_PROGRESS:
        render_in_progress_view()
    elif st.session_state.interview_status == STATUS_FINISHED:
        render_finished_view()
    else:
        render_not_started_view()


if __name__ == "__main__":
    main()

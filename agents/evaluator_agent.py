"""评估者智能体与报告生成者智能体。

评估流水线：
1. 评估者先对四个维度产出严格 JSON 评分。
2. 报告生成者把评分、JD 和对话整理成可读 Markdown 报告。
3. 报告生成者同时支持标准 PDF 导出，PDF 不含 Markdown 格式符号。

与面试官智能体的关系：
- 评估者仅读取 conversation_memory 和 interview_state 中的公开数据。
- 不依赖面试官内部状态机逻辑，两套代码互相独立。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from fpdf import FPDF

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

# 系统中文字体路径（Windows）
_FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")

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
# PDF 报告构建器
# ---------------------------------------------------------------------------


class PDFReportBuilder:
    """使用 fpdf2 将评估结果渲染为标准 PDF 文档。

    PDF 不含 Markdown 格式符号，而是按照印刷排版规范生成：
    - 分级标题（不同字号）
    - 标准表格（含边框、对齐）
    - 项目符号列表
    - 段落文本自动换行
    """

    def __init__(self) -> None:
        """初始化 PDF 构建器，注册中文字体。"""
        self._font_loaded = False
        self._page_w = 210 - 40  # A4 宽减去左右边距（各 20mm）
        self._page_h = 297 - 30  # A4 高减去上下边距（各 15mm）

    #: 报告模块渲染顺序 —— 不在该列表中的模块会追加到末尾
    _SECTION_ORDER: list[str] = [
        "总体结论",
        "能力评分",
        "关键亮点",
        "主要风险",
        "改进建议",
        "学习路径",
        "能力图谱",
        "后续追问建议",
        "转岗建议",
    ]

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def build(
        self,
        md_report: str,
        score_payload: dict[str, int],
    ) -> bytes:
        """将 Markdown 报告和评分结构渲染为标准 PDF。

        Args:
            md_report: Step 2 生成的 Markdown 报告全文。
            score_payload: Step 1 评出的四项分数。

        Returns:
            PDF 文件二进制内容。
        """
        sections = self._parse_md_sections(md_report)
        pdf = self._create_pdf()

        # — 封面标题 —
        self._render_title(pdf)

        # — 按优先级渲染已知模块 —
        rendered: set[str] = set()
        for section_name in self._SECTION_ORDER:
            if section_name in sections:
                if section_name == "能力评分":
                    self._render_score_table(pdf, score_payload, sections)
                else:
                    self._render_section(pdf, section_name, sections)
                rendered.add(section_name)

        # — 兜底：渲染 LLM 生成的其他模块（防未来新增模块被遗漏） —
        for section_name in sections:
            if section_name not in rendered:
                self._render_section(pdf, section_name, sections)

        return pdf.output()

    # ------------------------------------------------------------------
    # PDF 创建与字体
    # ------------------------------------------------------------------

    def _create_pdf(self) -> FPDF:
        """创建并配置 FPDF 实例。"""
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        if _FONT_PATH.exists():
            pdf.add_font("CJK", "", str(_FONT_PATH))
            pdf.add_font("CJK", "B", str(_FONT_PATH))
            self._font_loaded = True
        else:
            # 回退：如果找不到中文字体，使用内置字体（中文将显示为问号）
            self._font_loaded = False
            print("[PDFReportBuilder] 未找到中文字体，PDF 中文可能无法正常显示")

        return pdf

    def _use_font(
        self,
        pdf: FPDF,
        style: str = "",
        size: float = 10,
    ) -> None:
        """统一字体切换。"""
        if self._font_loaded:
            pdf.set_font("CJK", style, size)
        else:
            pdf.set_font("Helvetica", style, size)

    # ------------------------------------------------------------------
    # 渲染方法
    # ------------------------------------------------------------------

    def _render_title(self, pdf: FPDF) -> None:
        """渲染报告封面标题。"""
        self._use_font(pdf, "B", 22)
        pdf.ln(10)
        pdf.cell(self._page_w, 12, "面试评估报告", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        # 装饰线
        y = pdf.get_y()
        pdf.line(30, y, 180, y)
        pdf.ln(6)

    def _render_section(
        self,
        pdf: FPDF,
        section_name: str,
        sections: dict[str, str],
    ) -> None:
        """渲染一个报告模块。

        Args:
            pdf: FPDF 实例。
            section_name: 模块标题（如 "关键亮点"）。
            sections: 从 MD 解析出的模块字典。
        """
        content = sections.get(section_name, "")
        if not content.strip():
            return

        # 检查是否需要分页（至少需要 20mm 剩余空间）
        if pdf.get_y() > self._page_h - 20:
            pdf.add_page()

        # 模块标题
        self._use_font(pdf, "B", 14)
        pdf.cell(self._page_w, 8, section_name, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # 模块内容（剥离 Markdown 格式符号）
        content_clean = self._strip_markdown(content)
        self._render_content_lines(pdf, content_clean)

    def _render_score_table(
        self,
        pdf: FPDF,
        score_payload: dict[str, int],
        sections: dict[str, str],
    ) -> None:
        """渲染能力评分表格。

        使用 fpdf 原生表格 API，含表头和边框。
        """
        # 模块标题
        self._use_font(pdf, "B", 14)
        pdf.cell(self._page_w, 8, "能力评分", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # 解析 MD 表格中的"简要说明"列
        score_descriptions = self._parse_score_table_descriptions(
            sections.get("能力评分", "")
        )

        # 构建表格
        col_widths = [40, 18, 102]  # 维度 / 分数 / 说明
        headers = ["能力维度", "评分", "说明"]

        # 表头
        self._use_font(pdf, "B", 10)
        pdf.set_fill_color(240, 240, 240)
        for i, (header, w) in enumerate(zip(headers, col_widths)):
            pdf.cell(w, 8, header, border=1, fill=True, align="C")
        pdf.ln()

        # 数据行
        self._use_font(pdf, "", 10)
        for key in SCORE_KEYS:
            score = score_payload.get(key, 0)
            desc = self._strip_markdown(score_descriptions.get(key, ""))

            # 计算行高（根据说明文字长度自动调整）
            desc_wrapped = self._wrap_text(pdf, desc, col_widths[2])
            line_count = max(len(desc_wrapped), 1)
            row_h = max(line_count * 5.5, 7)

            # 检查是否需要分页
            if pdf.get_y() + row_h > self._page_h:
                pdf.add_page()
                # 翻页后重绘表头
                self._use_font(pdf, "B", 10)
                pdf.set_fill_color(240, 240, 240)
                for i, (header, w) in enumerate(zip(headers, col_widths)):
                    pdf.cell(w, 8, header, border=1, fill=True, align="C")
                pdf.ln()
                self._use_font(pdf, "", 10)

            y_before = pdf.get_y()

            # 维度列
            pdf.cell(col_widths[0], row_h, key, border=1, align="C")
            # 分数列
            pdf.cell(col_widths[1], row_h, str(score), border=1, align="C")
            # 说明列 —— 用 rect 先画框再填充文字，保证四边边框完整
            x0 = pdf.get_x()
            pdf.rect(x0, y_before, col_widths[2], row_h)
            for line in desc_wrapped[:3]:
                pdf.cell(col_widths[2], 5.5, line, border=0, align="L")
                pdf.set_x(x0)
                pdf.ln(5.5)
            pdf.set_y(y_before + row_h)

        pdf.ln(4)

    @staticmethod
    def _wrap_text(pdf: FPDF, text: str, width: float) -> list[str]:
        """按实际渲染宽度逐字精确换行（针对 CJK 字符）。"""
        if not text:
            return [""]
        if width <= 0:
            return [text]
        lines: list[str] = []
        current = ""
        for ch in text:
            test = current + ch
            if pdf.get_string_width(test) > width:
                if current:
                    lines.append(current)
                current = ch
            else:
                current = test
        if current:
            lines.append(current)
        return lines or [""]

    def _render_content_lines(self, pdf: FPDF, content: str) -> None:
        """渲染模块正文内容，处理列表和普通段落。"""
        self._use_font(pdf, "", 10)
        lines = content.strip().split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                pdf.ln(2)
                continue

            if pdf.get_y() > self._page_h - 10:
                pdf.add_page()

            # 数字列表项：1. xxx  —— 按实际前缀宽度精确换行
            num_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
            if num_match:
                num, text = num_match.groups()
                indent = 10
                prefix = f"{num}. "
                prefix_w = pdf.get_string_width(prefix)
                wrap_w = max(self._page_w - indent - prefix_w, 30)
                wrapped = self._wrap_text(pdf, text, wrap_w)
                for i, wl in enumerate(wrapped):
                    pdf.set_x(pdf.l_margin + indent)
                    if i == 0:
                        pdf.cell(
                            self._page_w - indent,
                            5.5,
                            prefix + wl,
                            new_x="LMARGIN",
                            new_y="NEXT",
                        )
                    else:
                        pdf.cell(
                            self._page_w - indent,
                            5.5,
                            "   " + wl,
                            new_x="LMARGIN",
                            new_y="NEXT",
                        )
                continue

            # 无序列表项：- xxx 或 * xxx  —— 按实际前缀宽度精确换行
            bullet_match = re.match(r"^[-*]\s+(.+)", stripped)
            if bullet_match:
                text = bullet_match.group(1)
                indent = 10
                prefix = "- "
                prefix_w = pdf.get_string_width(prefix)
                wrap_w = max(self._page_w - indent - prefix_w, 30)
                wrapped = self._wrap_text(pdf, text, wrap_w)
                for i, wl in enumerate(wrapped):
                    pdf.set_x(pdf.l_margin + indent)
                    if i == 0:
                        pdf.cell(
                            self._page_w - indent,
                            5.5,
                            prefix + wl,
                            new_x="LMARGIN",
                            new_y="NEXT",
                        )
                    else:
                        pdf.cell(
                            self._page_w - indent,
                            5.5,
                            "  " + wl,
                            new_x="LMARGIN",
                            new_y="NEXT",
                        )
                continue

            # 优先标签：如 「优先」
            if stripped.startswith("「") or stripped.startswith("**"):
                self._use_font(pdf, "B", 10)

            # 普通段落：自动换行
            wrapped = self._wrap_text(pdf, stripped, self._page_w)
            for wl in wrapped:
                pdf.cell(self._page_w, 5.5, wl, new_x="LMARGIN", new_y="NEXT")
            self._use_font(pdf, "", 10)

        pdf.ln(3)

    # ------------------------------------------------------------------
    # Markdown 解析
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """去除行内 Markdown 格式符号，PDF 使用字体控制排版。

        处理：**粗体**、*斜体*、`代码`、~~删除线~~、[链接](url)。
        列表前缀 (- /*) 和标题 (##) 由 _parse_md_sections 在上游处理。
        """
        # **粗体** → 粗体文本
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        # *斜体*（不匹配列表前缀的 -）
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
        # ~~删除线~~
        text = re.sub(r"~~(.+?)~~", r"\1", text)
        # `行内代码`
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # [链接文字](url) → 链接文字
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        # 多余空格
        return text.strip()

    #: 文档标题关键词 —— 匹配到这些前缀的标题不会作为报告模块渲染
    _TITLE_KEYWORDS = ("面试报告", "评估报告")

    @classmethod
    def _parse_md_sections(cls, md_text: str) -> dict[str, str]:
        """将 Markdown 报告按标题拆分为模块字典。

        同时兼容 H2 (## ) 和 H3 (### )，以应对 LLM 输出层级不
        固定的问题。以「面试报告」「评估报告」开头的标题被视为封面
        标题层，不会进入模块字典。

        Args:
            md_text: 完整的 Markdown 报告文本。

        Returns:
            {"模块名": "模块内容", ...} 的字典。
        """
        sections: dict[str, str] = {}
        current_key: str | None = None
        current_lines: list[str] = []

        for line in md_text.split("\n"):
            header_match = re.match(r"^#{2,3}\s+(.+)", line)
            if header_match:
                if current_key is not None:
                    sections[current_key] = "\n".join(current_lines).strip()
                title = header_match.group(1).strip()
                # 过滤掉封面标题行
                if title.startswith(cls._TITLE_KEYWORDS):
                    current_key = None
                    current_lines = []
                    continue
                current_key = title
                current_lines = []
            elif current_key is not None:
                current_lines.append(line)

        if current_key is not None:
            sections[current_key] = "\n".join(current_lines).strip()

        return sections

    @staticmethod
    def _parse_score_table_descriptions(
        table_content: str,
    ) -> dict[str, str]:
        """从能力评分的 Markdown 表格中提取每行的说明文字。

        Args:
            table_content: "能力评分" 模块的 Markdown 文本。

        Returns:
            {"专业深度": "一句话说明", ...} 的字典。
        """
        descriptions: dict[str, str] = {}
        for line in table_content.split("\n"):
            # 匹配表格行：| 专业深度 | 85 | 说明文字 |
            match = re.match(
                r"\|\s*(专业深度|表达逻辑|应变能力|岗位匹配度)\s*\|\s*\d+\s*\|\s*(.+?)\s*\|",
                line,
            )
            if match:
                key = match.group(1)
                desc = match.group(2).strip()
                descriptions[key] = desc
        return descriptions


# ---------------------------------------------------------------------------
# 报告生成者
# ---------------------------------------------------------------------------


class ReportGeneratorAgent:
    """负责把结构化评分转换成可读 Markdown 报告和标准 PDF 报告。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化报告生成者。

        Args:
            llm_client: 大模型客户端实例。
        """
        self.llm_client = llm_client
        self.pdf_builder = PDFReportBuilder()

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def generate_markdown_report(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
        score_payload: dict[str, int],
        interview_state: dict[str, Any],
    ) -> str:
        """根据评分 JSON 和对话材料生成 Markdown 报告。

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
            interview_state=interview_state,
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

    def generate_pdf_report(
        self,
        md_report: str,
        score_payload: dict[str, int],
    ) -> bytes:
        """将 Markdown 报告和评分渲染为标准 PDF。

        PDF 非 Markdown 转 PDF，而是基于 fpdf2 的排版渲染：
        - 标题、正文使用不同字号
        - 表格使用原生 PDF 表格
        - 列表使用项目符号

        Args:
            md_report: 已生成的 Markdown 报告全文。
            score_payload: Step 1 评出的四项分数。

        Returns:
            PDF 文件二进制内容。
        """
        return self.pdf_builder.build(
            md_report=md_report,
            score_payload=score_payload,
        )

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
        interview_state: dict[str, Any],
    ) -> str:
        """构建报告生成的 user prompt，保证材料和评分规整传递。"""
        return (
            "请根据以下材料生成最终面试报告。\n\n"
            f"【岗位 JD】\n{jd_text}\n\n"
            "【评分 JSON】\n"
            f"{json.dumps(score_payload, ensure_ascii=False, indent=2)}\n\n"
            "【全局面试地图】\n"
            f"{json.dumps(interview_state, ensure_ascii=False, indent=2)}\n\n"
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
    - 报告生成阶段同时产出 Markdown 和 PDF 两种格式。
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
        interview_state: dict[str, Any],
    ) -> str:
        """生成完整 Markdown 面试评估报告。

        这是评估流水线入口：
        - Step 1：评估者产出机器可解析的评分 JSON。
        - Step 2：报告生成者读取评分和原始上下文，产出 Markdown。

        评分和 PDF 同时缓存到 evaluation_state，供后续 get 方法取用。

        Args:
            full_memory_list: 完整面试对话历史（list[dict] 格式）。
            jd_text: 岗位 JD 文本。
            interview_state: 面试官维护的全局面试状态（含覆盖维度与总结）。

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
                interview_state=interview_state,
            )
            self.evaluation_state["scores"] = score_payload
            self.evaluation_state["scoring_success"] = True

        except (JSONDecodeError, ValueError, TypeError) as exc:
            self._debug_log(f"score JSON parse failed: {exc}")
            return self._format_score_parse_error(exc)

        except Exception as exc:
            self._debug_log(f"unexpected error in scoring: {exc}")
            return self.report_generator.generate_error_report(str(exc))

        # —— Step 2：Markdown 报告生成 ——
        try:
            md_report = self.report_generator.generate_markdown_report(
                full_memory_list=full_memory_list,
                jd_text=jd_text,
                score_payload=score_payload,
                interview_state=interview_state,
            )
            self.evaluation_state["md_report"] = md_report
            self.evaluation_state["report_generated"] = True

        except Exception as exc:
            self._debug_log(f"unexpected error in report generation: {exc}")
            return self.report_generator.generate_error_report(str(exc))

        # —— Step 3：PDF 报告生成（基于已生成的 MD） ——
        try:
            pdf_bytes = self.report_generator.generate_pdf_report(
                md_report=md_report,
                score_payload=score_payload,
            )
            self.evaluation_state["pdf_bytes"] = bytes(pdf_bytes)
            self.evaluation_state["pdf_generated"] = True
            self._debug_log(
                f"PDF generated, size={len(pdf_bytes)} bytes"
            )

        except Exception as exc:
            self._debug_log(f"PDF generation failed: {exc}")
            import traceback
            self.evaluation_state["pdf_error"] = f"{type(exc).__name__}: {exc}"
            self.evaluation_state["pdf_generated"] = False

        return md_report

    def get_pdf_bytes(self) -> bytes | None:
        """获取最近一次评估生成的 PDF 二进制数据。

        Returns:
            PDF 字节串；如果 PDF 生成失败或尚未评估，返回 None。
        """
        pdf_bytes = self.evaluation_state.get("pdf_bytes")
        # fpdf2 output() 可能返回 bytearray，统一转为 bytes
        if isinstance(pdf_bytes, (bytes, bytearray)):
            return bytes(pdf_bytes)
        return None

    def get_pdf_error(self) -> str:
        """获取 PDF 生成失败的错误信息。

        Returns:
            错误描述字符串；如果没有错误则返回空字符串。
        """
        return str(self.evaluation_state.get("pdf_error", ""))

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
            "pdf_generated": False,
            "md_report": "",
            "pdf_bytes": None,
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
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        return self._parse_score_json(repaired_response)

    # ------------------------------------------------------------------
    # 评分生成
    # ------------------------------------------------------------------

    def _generate_score_json(
        self,
        full_memory_list: list[dict[str, str]],
        jd_text: str,
        interview_state: dict[str, Any],
    ) -> dict[str, int]:
        """Step 1：调用 LLM 生成四项结构化评分。

        内置 JSON 修复流程：如果第一次输出的内容无法解析为合法 JSON，
        使用修复提示词重新请求一次。修复失败则抛出异常，由上层
        generate_evaluation_report 返回友好错误报告。
        """
        user_prompt = (
            "请基于岗位 JD 和完整面试对话输出评分 JSON。\n\n"
            f"【岗位 JD】\n{jd_text}\n\n"
            "【全局面试地图】\n"
            f"{json.dumps(interview_state, ensure_ascii=False, indent=2)}\n\n"
            "【完整面试对话】\n"
            f"{json.dumps(full_memory_list, ensure_ascii=False, indent=2)}"
        )

        raw_score_text = self.llm_client.chat(
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=800,
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

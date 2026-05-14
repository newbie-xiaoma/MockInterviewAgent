"""对话历史管理模块。"""

from __future__ import annotations


class ConversationMemory:
    """维护一场面试中的对话历史。"""

    def __init__(self) -> None:
        """初始化空消息列表。"""
        self.messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """追加一条对话消息。

        Args:
            role: 消息角色，通常为 system、user 或 assistant。
            content: 消息正文。
        """
        self.messages.append({"role": role, "content": content})

    def get_history(self) -> list[dict[str, str]]:
        """获取当前对话历史副本。

        Returns:
            对话消息列表的浅拷贝，避免外部直接修改内部列表对象。
        """
        return self.messages.copy()

    def clear(self) -> None:
        """清空全部对话历史。"""
        self.messages.clear()

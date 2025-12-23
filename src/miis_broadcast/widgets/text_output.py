from PySide6 import QtCore, QtGui, QtWidgets


class TextOutputWidget(QtWidgets.QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.initializeUI()

    def initializeUI(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 用 QPlainTextEdit：天生支援捲動、效能也比 QTextEdit 好（適合一直 append）
        self.text_output = QtWidgets.QPlainTextEdit(self)
        self.text_output.setReadOnly(True)
        self.text_output.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.text_output.setUndoRedoEnabled(False)

        # 讓使用者可以選取/複製文字
        self.text_output.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )

        # 視覺（可留可不留）
        self.text_output.setStyleSheet("""
            QPlainTextEdit {
                background: #252525;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 10px;
                padding: 10px;
                line-height: 150%;
            }
        """)

        layout.addWidget(self.text_output)

    def appendText(self, new_text: str) -> None:
        self.text_output.appendPlainText(new_text)
        self.scrollToBottom()

    def clearText(self) -> None:
        self.text_output.clear()

    def setText(self, new_text: str) -> None:
        self.text_output.setPlainText(new_text)
        self.scrollToBottom()

    def getText(self) -> str:
        return self.text_output.toPlainText()

    def scrollToBottom(self) -> None:
        sb = self.text_output.verticalScrollBar()
        sb.setValue(sb.maximum())

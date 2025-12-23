from PySide6 import QtCore, QtGui, QtWidgets

class FileExplorerWidget(QtWidgets.QTreeView):
    keyUpDown = QtCore.Signal(QtCore.QModelIndex)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        model = QtWidgets.QFileSystemModel()
        model.setRootPath(QtCore.QDir.currentPath())
        model.setNameFilters(['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.mp4', '*.avi', '*.mkv'])
        model.setNameFilterDisables(False)
        model.setReadOnly(True)
        self.setModel(model)

        self.hideColumn(1)
        self.hideColumn(2)
        self.hideColumn(3)

    def setRootPath(self, path: str) -> None:
        self.setRootIndex(self.model().index(path))

    def getModelPath(self, index: QtCore.QModelIndex) -> str:
        return self.model().filePath(index)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() in [QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down]:
            self.keyUpDown.emit(self.selectedIndexes()[0])
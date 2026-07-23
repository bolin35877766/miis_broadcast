from PySide6 import QtCore

from miis_broadcast.workers.gemini import GeminiBackgroundWorker


class _ContextEmitter(QtCore.QObject):
    context = QtCore.Signal(str)


def test_background_loop_yields_to_queued_context(monkeypatch):
    """The polling scheduler must not starve update_context queued signals."""
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    worker = GeminiBackgroundWorker(get_remaining_sec_fn=lambda: 0.0)
    worker.POLL_INTERVAL_MS = 10
    worker.MIN_FIRE_INTERVAL_SEC = 0.0
    worker._build_context = lambda: {"event": "fresh context"}

    class _Event:
        broadcast_text = "Test commentary"

        @staticmethod
        def to_dict():
            return {
                "priority": 3,
                "broadcast_text": "Test commentary",
                "action_label": "test",
                "should_speak": True,
            }

    monkeypatch.setattr(
        "miis_broadcast.core.models.gemini_broadcaster.stream_gemini",
        lambda _context: iter([_Event()]),
    )

    thread = QtCore.QThread()
    worker.moveToThread(thread)
    emitter = _ContextEmitter()
    emitter.context.connect(worker.update_context, QtCore.Qt.QueuedConnection)

    received = []
    loop = QtCore.QEventLoop()
    worker.signal_broadcast.connect(lambda *_args: (received.append(_args), loop.quit()))
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)

    thread.started.connect(worker.run_background_loop)
    thread.start()
    emitter.context.emit("A player drives toward the basket")
    timeout.start(1500)
    loop.exec()

    QtCore.QMetaObject.invokeMethod(worker, "requestStop", QtCore.Qt.BlockingQueuedConnection)
    thread.quit()
    thread.wait(1000)

    assert received, "queued LiveCC context never reached background Gemini"
    assert worker._context_version == 1

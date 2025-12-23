import mimetypes
import time

import cv2
import numpy as np
from ..core.io.input import CameraInput, ImageInput, VideoInput
from PySide6 import QtCore, QtWidgets

class BaseWorkerThread(QtCore.QThread):
    frame_data_updated = QtCore.Signal(np.ndarray, int)
    invalid_video_file = QtCore.Signal()
    video_ended = QtCore.Signal()

    def __init__(self, parent, video_source='') -> None:
        super().__init__(parent=parent)

        self.parent = parent
        self.video_source = video_source

        self.io = None
        self.delay = 1.0
        self.fps = 1.0
        self.frame_count = 1

    def run(self) -> None:
        raise NotImplementedError('Has to be implemented in subclasses')

    def spin(self, seconds=None):
        if seconds is None:
            seconds = self.delay
        elif seconds <= 0:
            return

        time_end = time.perf_counter() + seconds
        while time.perf_counter() < time_end:
            QtWidgets.QApplication.processEvents()

    def stop_thread(self):
        self.wait()
        QtWidgets.QApplication.processEvents()
        if self.io is not None:
            self.io.release()

class VideoWorkerThread(BaseWorkerThread):
    def __init__(self, parent, video_source: str=None) -> None:
        super().__init__(parent, video_source)

        try:
            self.io = CameraInput(self.video_source)

            self.video_status = True
            self.delay = self.io.delay
            self.fps = self.io.fps
        except FileNotFoundError:
            self.video_status = False
            self.invalid_video_file.emit()

    def run(self):
        while True:
            try:
                if self.io is not None and self.parent.thread_is_running:
                    try:
                        frame = self.io.get_frame()
                        self.parent.counter_read.update()
                        self.frame_data_updated.emit(frame, 0)
                    except EOFError:
                        self.video_ended.emit()
                        break
                else:
                    self.video_ended.emit()
                    break
            except AttributeError:
                self.video_ended.emit()
                break

class OfflineVideoWorkerThread(BaseWorkerThread):
    def __init__(self, parent, video_source: str=None) -> None:
        super().__init__(parent, video_source)

        input_type = mimetypes.guess_type(video_source)[0]
        if input_type:
            mime_type, mime_subtype = input_type.split("/")
        else:
            mime_type = ""

        self.run_function = self._run_none
        try:
            if mime_type == "image":
                self.io = ImageInput(self.video_source)
                self.run_function = self._run_image
            elif mime_type == "video":
                self.io = VideoInput(self.video_source)
                self.run_function = self._run_video
            else:
                raise FileNotFoundError()

            self.video_status = True
            self.delay = self.io.delay
            self.fps = self.io.fps
            self.frame_count = self.io.frame_count
            self.seek_frame = 0
            self.current_frame = 0
        except FileNotFoundError:
            self.video_status = False
            self.invalid_video_file.emit()

    def run(self) -> None:
        self.run_function()

    def _run_image(self) -> None:
        img = self.io.get_frame()
        self.parent.counter_read.update()
        self.frame_data_updated.emit(img, 0)
        self.video_ended.emit()

    def _run_video(self) -> None:
        while self.parent.thread_is_running:
            try:
                t0 = time.perf_counter()
                try:
                    if not self.parent.thread_is_paused:
                        img = self.io.get_frame()
                        self.parent.counter_read.update()
                        self.frame_data_updated.emit(img, self.io.current_frame)
                except EOFError:
                    self.video_ended.emit()
                    break
                self.spin_remaining(t0)
            except AttributeError:
                self.video_ended.emit()
                break
    
    def _run_none(self) -> None:
        pass

    def run_once(self) -> None:
        try:
            img = self.io.get_frame()
            self.parent.counter_read.update()
            self.frame_data_updated.emit(img, self.io.current_frame)
        except EOFError:
            self.video_ended.emit()

    def spin_remaining(self, start_time: float=time.perf_counter()) -> None:
        end_time = start_time + self.delay
        remaining_time = end_time - time.perf_counter()

        self.spin(remaining_time)

    def goto_frame(self, position: int) -> None:
        self.io.goto_frame(position)

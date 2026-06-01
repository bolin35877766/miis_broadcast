import os
import cv2
import numpy as np
from PIL import Image

class BaseInput:
    def __init__(self, input_path) -> None:
        self.input_path = input_path
        
        self.capture = None
        self.frame_count = 0
        self.fps = 1.0
        self.delay = 1.0

    def verify_input_source(self) -> bool:
        return True

    def calculate_frame_delay(self):
        self.delay = 1 / (self.fps + 1e-8)
        self.delay = round(self.delay, 3)

    def get_frame(self) -> np.ndarray:
        raise NotImplementedError('Has to be implemented in subclasses')

    def goto_frame(self, frame_position) -> None:
        raise NotImplementedError('Has to be implemented in subclasses')

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()

class CameraInput(BaseInput):
    def __init__(self, input_path) -> None:
        super().__init__(input_path)

        if self.verify_input_source():
            self.capture = cv2.VideoCapture(self.input_path)

            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.calculate_frame_delay()
        else:
            raise FileNotFoundError()

    def verify_input_source(self):
        capture = cv2.VideoCapture(self.input_path)
        if not capture.isOpened():
            return False
        capture.release()
        return True

    def get_frame(self) -> np.ndarray:
        if self.capture is not None and self.capture.isOpened():
            status, img = self.capture.read()
            if status:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise EOFError()
        else:
            raise EOFError()

        return img

    def goto_frame(self, frame_position) -> None:
        pass # do nothing

class ImageInput(BaseInput):
    def __init__(self, input_path) -> None:
        super().__init__(input_path)
        
        self.frame_count = 1

        if not self.verify_input_source():
            raise FileNotFoundError()

    def verify_input_source(self) -> bool:
        return os.path.isfile(self.input_path)

    def get_frame(self) -> np.ndarray:
        img = np.array(Image.open(self.input_path), dtype=np.uint8)

        return img

    def goto_frame(self, frame_position) -> None:
        pass # do nothing

class VideoInput(BaseInput):
    def __init__(self, input_path) -> None:
        super().__init__(input_path)

        self.seeking = False
        self.seek_frame = 0
        self.current_frame = 0

        if self.verify_input_source():
            self.capture = cv2.VideoCapture(self.input_path)

            self.frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.calculate_frame_delay()
        else:
            raise FileNotFoundError()

    def verify_input_source(self):
        capture = cv2.VideoCapture(self.input_path)
        if not capture.isOpened():
            return False
        capture.release()
        return True

    def get_frame(self) -> np.ndarray:
        if self.seeking:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
            self.seeking = False
            self.current_frame = self.seek_frame
            self.seek_frame = 0

        if self.capture is not None and self.capture.isOpened():
            status, img = self.capture.read()
            if status:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.current_frame += 1
            else:
                raise EOFError()
        else:
            raise EOFError()

        return img

    def goto_frame(self, frame_position) -> None:
        if frame_position >= 0 and frame_position < self.frame_count:
            self.seeking = True
            self.seek_frame = frame_position

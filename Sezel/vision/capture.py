from __future__ import annotations

import mss
import mss.tools
from PIL import Image
from io import BytesIO
import base64

from jedi.inference.utils import to_tuple
from zmq.utils import monitor


class ScreenCapture:
    """
        Usage:
        capture = ScreenCapture()
        img = capture.screen()        # PIL Image of the full monitor
        b64  = capture.base64(img)    # base64 string for Ollama/LLM APIs
    """
    def  __init__(self, monitor: int = 0):
        """
        Args:
            monitor: Monitor index (0 = all monitors combined, 1 = primary, etc.)
        """
        self.monitor = monitor

    def screen(self) -> Image.Image:
        with mss.mss() as sct:
            sct_img = sct.grab(sct.monitors[self.monitor])
            return Image.frombytes("RGB", to_tuple(sct_img.size), sct_img.rgb)

    def region(self, left: int, top: int, width: int, height: int) -> Image.Image:
        # capture specific region of the screen
        with mss.mss() as sct:
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            return Image.frombytes("RGB", to_tuple(sct_img.size), sct_img.rgb)

    @staticmethod
    def base64(img: Image.Image, format: str = "PNG") -> str:
        buf = BytesIO()
        img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
        buf = BytesIO()
        img.save(buf, format=format)
        return buf.getvalue()

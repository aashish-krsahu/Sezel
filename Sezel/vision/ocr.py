"""OCR: extract text from screen images using Tesseract."""

from __future__ import annotations

from dataclasses import dataclass, field
from PIL import Image
import pytesseract

@dataclass
class WordBox:
    """A single word with its bounding box and confidence."""
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 0.0

@dataclass
class OCRResult:
    """full OCR result: raw text + per-word detail"""
    text: str
    words: list[WordBox] = field(default_factory=list)

class OCREngine:
    """
    Extract text from images using Tesseract.

    Usage:
        ocr = OCREngine()
        result = ocr.read(screenshot_pil)
    """
    def __init__(self, tesseract_cmd: str | None = None):
        """
        Args:
            tesseract_cmd: Path to tesseract executable.
                           If None, uses system PATH.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def read(self, image: Image.Image) -> OCRResult:
        """
        Pre-processing: convert to grayscale + slight contrast boost
        for better accuracy.
        """

        gray = image.convert("L")

        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        words = []
        text_parts = []

        for i in range(len(data["text"])):
            word_text = data["text"][i].strip()
            conf = int(data["conf"][i])

            if word_text and conf > 0:
                words.append(WordBox(
                    text = word_text,
                    left = data["left"][i],
                    top = data["top"][i],
                    width = data["width"][i],
                    height = data["height"][i],
                    confidence = conf/100.0,
                ))
                text_parts.append(word_text)


        return OCRResult(
            text = " ".join(text_parts),
            words= words
        )


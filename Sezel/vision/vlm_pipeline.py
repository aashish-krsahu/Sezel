"""Vision pipeline: orchestrates capture → OCR → a11y → VLM → visual memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from PIL import Image
from typing import Optional

from .capture import ScreenCapture
from .ocr import OCRResult, OCREngine
from .ally import UIElement, UIAccessibility
from ..cognition.vlm import OllamaVLM
from ..cognition.model_manager import ModelManager
from ..memory.visual import VisualStore

@dataclass
class VLMContext:
    text_on_screen: str
    elements: list[UIElement] = field(default_factory=list)
    caption: Optional[str] = None
    image_ref: Optional[str] = None
    ocr_detail: Optional[str] = None

class VLMPipeline:
    """
        The main vision coordinator.

    Usage:
        pipeline = VisionPipeline()
        vc = await pipeline.look(need_semantics=True)
        print(vc.text_on_screen)
        print(vc.caption)
    """

    def __init__(
            self,
            capture: ScreenCapture | None = None,
            ocr: OCRResult | None = None,
            ally: UIAccessibility | None = None,
            vlm: OllamaVLM | None = None,
            model_manager: ModelManager | None = None,
            visual_store: VisualStore | None = None,
    ):
        self.capture = capture or ScreenCapture()
        self.ocr = ocr or OCRResult()
        self.ally = ally or UIAccessibility()
        self.vlm = vlm
        self.model_manager = model_manager
        self.visual_store = visual_store

    async def look(
            self,
            need_semantics: bool = True,
    ) -> VLMContext:
        """
        Args:
            need_semantics: If True, also runs the VLM to generate a caption.
                            Set to False when only OCR + UI tree is needed
                            (saves VRAM/time).

        Returns:
            A VisualContext with all available information.
        """
        screenshot = self.capture.screen()

        ocr_result = self.ocr.read(screenshot)

        ui_elements = self.ally.tree()

        caption = None
        if need_semantics and self.vlm is not None:
            if self.model_manager is not None:
                async with self.model_manager.using("vlm", vram_gb= 4.0) as acquired:
                    if acquired:
                        caption = await self.vlm.caption(screenshot)
                    else:
                        caption = "VLM unavailable - not enough VRAM"

            else:
                caption = await self.vlm.caption(screenshot)

        image_ref = None
        if self.visual_store is not None:
            image_ref = await self.visual_store.save(
                image = screenshot,
                caption = caption or ocr_result.text,
                ocr = ocr_result.text,
            )

        return VLMContext(
            text_on_screen = ocr_result.text,
            elements = ui_elements,
            caption = caption,
            image_ref = image_ref,
            ocr_detail = ocr_result,
        )


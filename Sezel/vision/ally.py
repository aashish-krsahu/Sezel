from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sympy import false


@dataclass
class UIElement:
    role: str
    name: str
    bbox: tuple[int,int,int,int]
    clickable: bool = False
    enabled: bool = True
    children: list["UIElement"] = field(default_factory=list)

class UIAccessibility:
    """
    Extract the Windows UI Automation tree.

    Uses pywinauto's Desktop object to enumerate all top-level windows
    and their children.

    This is Windows-only. On other platforms, this returns an empty list.
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def tree(self) -> list[UIElement]:
        """Get the full UI Automation tree as a flat list of UIElement."""

        elements : list[UIElement] = []
        try:
            import pywinauto
            from pywinauto import Desktop

            desktop = Desktop(backend="uia")
            for window in desktop.windows():
                win_el = self._element_from_wrapper(window, depth = 0)
                elements.append(win_el)
                self._enumerate_children(window, win_el, depth = 1, elements = elements)

        except ImportError:
            pass
        except Exception as e:
            print(f"  [a11y: UI tree error — {e}]")
            pass

        return elements

    def _enumerate_children(
            self,
            wrapper,
            parent: UIElement,
            depth: int,
            elements: list[UIElement],
    ) -> None:
        """Recursively enumerate children of a UI element"""
        if depth > self.max_depth:
            return

        try:
            for child in wrapper.children():
                try:
                    el = self._element_from_wrapper(child, depth)
                    parent.children.append(el)
                    elements.append(el)
                    self._enumerate_children(child, el, depth + 1, elements)

                except Exception:
                    continue

        except Exception:
            pass

    @staticmethod
    def _element_from_wrapper(wrapper, depth: int)-> UIElement:
        """Convert a pywinauto wrapper to our UIElement dataclass."""
        try:
            rect = wrapper.rectangle()
            bbox = (rect.left, rect.top, rect.right, rect.bottom)
        except Exception:
            bbox = (0, 0, 0, 0)

        try:
            ctrl = wrapper.element_info.control_type or "pane"
        except Exception:
            ctrl = "pane"

        try:
            name = wrapper.window_text() or ""
        except Exception:
            name = ""

        return UIElement(
            role= ctrl.lower(),
            name=name,
            bbox=bbox,
            clickable=ctrl in ("button", "check box", "menu item", "tab item"),
            enabled= True
        )


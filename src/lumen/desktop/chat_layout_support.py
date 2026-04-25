from __future__ import annotations


def bubble_side_padding(canvas_width: int) -> int:
    width = max(int(canvas_width or 0), 640)
    return max(48, min(140, width // 6))


def bubble_wraplength(canvas_width: int) -> int:
    width = max(int(canvas_width or 0), 640)
    side_padding = bubble_side_padding(width)
    return max(280, min(620, width - (side_padding * 2) - 40))

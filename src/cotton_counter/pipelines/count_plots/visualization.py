"""
Tools for visualizing results.
"""


from PIL import ImageDraw, ImageFont, Image
from typing import List
import numpy as np
from typing import Tuple
import random


_TAG_FONT = ImageFont.truetype("fonts/VeraBd.ttf", 24)
"""
Font to use for bounding box tags.
"""


def _color_for_box() -> Tuple[int, int, int]:
    """
    Generates a unique color for a particular box.

    Returns:
        The generated color.

    """
    # Create a random color. We want it to be not very green (because the
    # background is pretty green), and relatively dark, so the label shows up
    # well.
    rgb = np.array(
        [
            random.randint(0, 255),
            random.randint(0, 128),
            random.randint(0, 255),
        ],
        dtype=np.float32,
    )

    brightness = np.sum(rgb)
    scale = brightness / 300
    # Keep a constant brightness.
    rgb *= scale

    return tuple(rgb.astype(int))


def _draw_text(
    artist: ImageDraw.ImageDraw,
    *,
    text: str,
    coordinates: Tuple[int, int],
    anchor: str = "la",
    color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Draws text on an image, over a colored box.

    Args:
        artist: The `ImageDraw` object to draw with.
        text: The text to draw.
        coordinates: The coordinates to place the text at.
        anchor: The anchor type to use for the text.
        color: The background color to use. (The text itself will be white.)

    """
    # Find and draw the bounding box.
    text_bbox = artist.textbbox(
        coordinates, text, anchor=anchor, font=_TAG_FONT
    )
    artist.rectangle(text_bbox, fill=color)

    # Draw the text itself.
    artist.text(
        coordinates, text, fill=(255, 255, 255), anchor=anchor, font=_TAG_FONT
    )


def _draw_bounding_box(
    artist: ImageDraw.ImageDraw,
    *,
    box: np.ndarray,
    confidence: float,
    color: Tuple[int, int, int],
) -> None:
    """
    Draws a bounding box.

    Args:
        artist: The `ImageDraw` object to draw with.
        box: The box to be drawn, in the form
            `[x1, y1, x2, y2]`.
        confidence: The confidence of the detection.
        color: The color to use for the box.
    """
    # Convert the box to a form we can draw with.
    min_point = tuple(box[:2])
    max_point = tuple(box[2:])

    artist.rectangle((min_point, max_point), outline=color, width=5)

    # Draw a tag with the track ID.
    tag_pos = min_point
    _draw_text(
        artist,
        text=f"Flower ({int(confidence * 100)}%)",
        anchor="lb",
        color=color,
        coordinates=tag_pos,
    )


def _draw_mask(
    artist: ImageDraw.ImageDraw,
    *,
    mask: Image.Image,
    color: Tuple[int, int, int],
) -> None:
    """
    Overlays a mask on top of an image.

    Args:
        artist: The `ImageDraw` object to draw with.
        mask: The mask to be drawn, as a boolean array the same size as the
            image.
        color: The color to use for the box.
    """
    # Find a color for the mask, and add transparency.
    mask_color = color + (127,)
    artist.bitmap((0, 0), mask, fill=mask_color)


def draw_detections(
    image: Image.Image, *, detections: np.array, masks: List[Image.Image]
) -> Image.Image:
    """
    Draws all of the detections for a particular image.

    Args:
        image: The image to draw  on.
        detections: The detection bounding boxes. Should have the form
            `[x1, y1, x2, y2, confidence]`, in pixels.
        masks: The segmentation masks.

    Returns:
        The same image, with the detections drawn.

    """
    # Draw on a transparency, and later merge this with the image.
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)

    # Draw everything.
    for box, mask in zip(detections, masks):
        color = _color_for_box()
        _draw_bounding_box(draw, box=box[:4], confidence=box[4], color=color)
        _draw_mask(draw, mask=mask, color=color)

    # Apply the overlay.
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    return image.convert("RGB")

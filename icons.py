"""PIL icon processing for podcast artwork."""

import customtkinter as ctk
from PIL import Image, ImageOps

from constants import SIDEBAR_ICON_SIZE


def make_square_icon(pil_image, size=SIDEBAR_ICON_SIZE):
    """Resize a PIL image to a uniform square CTkImage using center-crop."""
    img = pil_image.convert("RGBA")
    img = ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.5))
    return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))

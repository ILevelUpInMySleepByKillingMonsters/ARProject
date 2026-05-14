import cv2
from ARMask import ImageMaskProcessor


def process_background(image, background):
    image_mask_processor = ImageMaskProcessor()
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    white_mask = (
        (rgba[:, :, 0] == 255) & (rgba[:, :, 1] == 255) & (rgba[:, :, 2] == 255)
    )

    rgba[white_mask, 3] = 0

    return image_mask_processor._overlay_transparent(background, rgba, 0, 0)

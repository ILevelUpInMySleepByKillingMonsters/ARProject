import cv2
import numpy as np


def overlay_transparent(background, overlay, x=0, y=0, size=None):
    if size:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)

    bg_h, bg_w = background.shape[:2]
    o_h, o_w = overlay.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + o_w, bg_w), min(y + o_h, bg_h)

    if x2 <= x1 or y2 <= y1:
        return background

    o_x1, o_y1 = x1 - x, y1 - y
    o_x2, o_y2 = o_x1 + (x2 - x1), o_y1 + (y2 - y1)

    overlay_crop = overlay[o_y1:o_y2, o_x1:o_x2]
    overlay_img = overlay_crop[:, :, :3]
    mask = (overlay_crop[:, :, 3] / 255.0)[:, :, np.newaxis].astype(np.float32)

    bg_roi = background[y1:y2, x1:x2]
    background[y1:y2, x1:x2] = (mask * overlay_img + (1.0 - mask) * bg_roi).astype(
        np.uint8
    )

    return background


def remove_green_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask_inv = cv2.bitwise_not(mask)
    mask_inv = cv2.medianBlur(mask_inv, 9)
    result = cv2.bitwise_and(image, image, mask=mask_inv)

    return result


def replace_black_to_white(image):
    img = image.copy()

    black_pixels = np.where(
        (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    )

    img[black_pixels] = [255, 255, 255]

    return img


def overlay_background(image, background, mask):
    mask_inv = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)
    bg_cut = cv2.bitwise_and(background, background, mask=mask)
    return cv2.add(foreground, bg_cut)


def replace_no_white_background(image, background):
    white_mask = cv2.inRange(
        image, np.array([255, 255, 255]), np.array([255, 255, 255])
    )
    mask_inv = cv2.bitwise_not(white_mask)
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)
    bg_cut = cv2.bitwise_and(background, background, mask=white_mask)
    return cv2.add(foreground, bg_cut)


def replace_color_to_transparent(image, color=[255, 255, 255]):
    img_copy = image.copy()

    if img_copy.shape[2] == 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)

    white_mask = np.all(img_copy[:, :, :3] == color, axis=-1)
    img_copy[white_mask, 3] = 0
    return img_copy


def replace_color_to_color(image, color_from=[255, 255, 255], color_out=[0, 255, 0]):
    img_copy = image.copy()

    white_mask = np.all(img_copy[:, :, :3] == color_from, axis=-1)
    img_copy[white_mask] = color_out
    return img_copy


def match_background_size(original, mathed):
    h, w = original.shape[:2]
    bg_h, bg_w = mathed.shape[:2]

    scale = max(w / bg_w, h / bg_h)

    if w / bg_w >= h / bg_h:
        new_w = w
        new_h = int(bg_h * scale)
    else:
        new_w = int(bg_w * scale)
        new_h = h

    bg_frame_resized = cv2.resize(mathed, (new_w, new_h))

    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2

    return bg_frame_resized[start_y : start_y + h, start_x : start_x + w]


def resize_to(original, mathed):
    h, w = original.shape[:2]
    m_h, m_w = mathed.shape[:2]

    scale = min(w / m_w, h / m_h)

    new_h = int(m_h * scale)
    new_w = int(m_w * scale)

    return cv2.resize(mathed, (new_w, new_h))

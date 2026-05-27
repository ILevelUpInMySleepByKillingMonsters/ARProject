import cv2
from .mask_config import *
from ..paths import MASKS_DIR
from ..utils import image_utils


class ImageMasker:
    def __init__(self, mask_name="test"):
        mask = MASK_CONFIGS[mask_name]

        loaded_mask: list[MaskFrame] = []

        for mask_data in mask:
            src = str(MASKS_DIR / mask_data.file)

            if mask_data.animated:
                cap = cv2.VideoCapture(src)
                mask_frame = MaskFrame(index=0, frames=[])
                while cap.isOpened():
                    success, frame = cap.read()

                    if not success:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    frame = image_utils.remove_green_color(frame)
                    frame = image_utils.replace_color_to_transparent(frame)
                    mask_frame.frames.append(frame)

                loaded_mask.append(mask_frame)
            else:
                img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                mask_frame = MaskFrame(index=0, frames=[img])
                loaded_mask.append(mask_frame)

        self.loaded_mask = loaded_mask
        self.mask = mask

    def mask_image(self, image):
        h, w = image.shape[:2]

        output_image = image.copy()

        mask_index = 0
        for mask_data in self.mask:
            frame_index = self.loaded_mask[mask_index].index
            mask_img = self.loaded_mask[mask_index].frames[frame_index]
            self.loaded_mask[mask_index].index += 1

            if self.loaded_mask[mask_index].index >= len(
                self.loaded_mask[mask_index].frames
            ):
                self.loaded_mask[mask_index].index = 0

            mask_img = image_utils.resize_to(image, mask_img)
            m_h, m_w = mask_img.shape[:2]

            new_h = int(m_h * mask_data.scale_h)
            new_w = int(m_w * mask_data.scale_w)
            mask_img = cv2.resize(mask_img, (new_w, new_h))
            m_h, m_w = mask_img.shape[:2]

            if mask_data.place_pivot == PlacePivot.Center:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(w / 2 - m_w / 2), int(h / 2 - m_h / 2)
                )
            elif mask_data.place_pivot == PlacePivot.CenterLeft:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(0), int(h / 2 - m_h / 2)
                )
            elif mask_data.place_pivot == PlacePivot.CenterRight:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(w - m_w), int(h / 2 - m_h / 2)
                )
            elif mask_data.place_pivot == PlacePivot.Bottom:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(w / 2 - m_w / 2), h - m_h
                )
            elif mask_data.place_pivot == PlacePivot.BottomLeft:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(0), h - m_h
                )
            elif mask_data.place_pivot == PlacePivot.BottomRight:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(w - m_w), h - m_h
                )
            elif mask_data.place_pivot == PlacePivot.Top:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(w / 2 - m_w / 2), 0
                )
            elif mask_data.place_pivot == PlacePivot.TopLeft:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(0), 0
                )
            elif mask_data.place_pivot == PlacePivot.TopRight:
                output_image = image_utils.overlay_transparent(
                    image, mask_img, int(w - m_w), 0
                )
            else:
                output_image = image_utils.overlay_transparent(image, mask_img)

            mask_index += 1
            continue

        return output_image

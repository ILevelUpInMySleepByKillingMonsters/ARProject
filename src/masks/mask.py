import cv2
from .mask_config import MASK_CONFIGS, MaskConfigData, PlacePivot
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ..paths import MASKS_DIR
from ..utils import image_utils


class ImageMaskProcessor:
    def __init__(self, mask_name="test", show_face_points=False):
        mask = MASK_CONFIGS[mask_name]

        loaded_mask = []

        for mask_data in mask:
            img = cv2.imread(str(MASKS_DIR / mask_data.file), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            loaded_mask.append(img)

        self.loaded_mask = loaded_mask
        self.mask = mask
        self.show_face_points = show_face_points

        self.detector = self._get_detector()

    def _get_detector(self):
        asset_path = str(MASKS_DIR / "face_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options)

        return vision.FaceLandmarker.create_from_options(options)

    def _rotate_z_axis(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    def _get_landmark_coords(self, face_landmarks, w, h):
        return np.array([face_landmarks.x * w, face_landmarks.y * h])

    def _calculate_angle(self, p_left, p_right):
        d_x = p_right[0] - p_left[0]
        d_y = p_right[1] - p_left[1]

        angle_deg = np.degrees(np.arctan2(d_y, d_x))

        return angle_deg

    def _show_face_points(self, mask_data: MaskConfigData, face_landmarks, image, w, h):
        index = 0
        selected_point = [
            mask_data.left,
            mask_data.right,
            mask_data.bottom,
            mask_data.top,
            mask_data.center,
        ]
        for landmark in face_landmarks:
            if index not in selected_point:
                index += 1
                continue

            coords = self._get_landmark_coords(landmark, w, h)
            point = (int(coords[0]), int(coords[1]))

            cv2.circle(image, point, 2, (255, 0, 0), -1)
            cv2.putText(
                image,
                str(index),
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )
            index += 1

    def _process_image(
        self,
        image,
        mask_data: MaskConfigData,
        mask_index,
        w,
        h,
        m_center,
        m_left,
        m_right,
        m_top,
        m_bottom,
        new_mask_width,
        new_mask_height,
    ):
        angle_deg = self._calculate_angle(m_left, m_right)
        scale = 1
        img = self.loaded_mask[mask_index]
        if mask_data.native_size:
            r_h, r_w, _ = img.shape
            scale = r_w / new_mask_width
            resized_mask = cv2.resize(
                img,
                (int(new_mask_width), int(r_h / scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized_mask = cv2.resize(
                img,
                (new_mask_width, new_mask_height),
                interpolation=cv2.INTER_AREA,
            )

        if mask_data.rotate:
            rotated_mask = self._rotate_z_axis(resized_mask, -angle_deg)
        else:
            rotated_mask = resized_mask

        m_h, m_w, _ = rotated_mask.shape

        if mask_data.place_pivot == PlacePivot.Center:
            mask_x = int(m_center[0] - m_w / 2 + (int)(mask_data.start_point_x / scale))
            mask_y = int(m_center[1] - m_h / 2 + (int)(mask_data.start_point_y / scale))
        elif mask_data.place_pivot == PlacePivot.Bottom:
            mask_x = int(m_center[0] - m_w / 2 + (int)(mask_data.start_point_x / scale))
            mask_y = int(m_center[1] + (int)(mask_data.start_point_y / scale))
        else:
            mask_x = int(m_center[0] - m_w / 2 + (int)(mask_data.start_point_x / scale))
            mask_y = int(m_center[1] - m_h + (int)(mask_data.start_point_y / scale))

        image = image_utils.overlay_transparent(image, rotated_mask, mask_x, mask_y)

    def mask_image(self, image):
        h, w, _ = image.shape

        output_image = image

        rgb_frame = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                mask_index = 0
                for mask_data in self.mask:
                    m_left = self._get_landmark_coords(
                        face_landmarks[mask_data.left], w, h
                    )
                    m_right = self._get_landmark_coords(
                        face_landmarks[mask_data.right], w, h
                    )
                    m_top = self._get_landmark_coords(
                        face_landmarks[mask_data.top], w, h
                    )
                    m_bottom = self._get_landmark_coords(
                        face_landmarks[mask_data.bottom], w, h
                    )
                    m_center = self._get_landmark_coords(
                        face_landmarks[mask_data.center], w, h
                    )

                    new_mask_width = int(
                        np.linalg.norm(m_left - m_right) * mask_data.scale_w
                    )
                    new_mask_height = int(
                        np.linalg.norm(m_bottom - m_top) * mask_data.scale_h
                    )

                    if new_mask_width < 10 or new_mask_height < 10:
                        continue

                    self._process_image(
                        output_image,
                        mask_data,
                        mask_index,
                        w,
                        h,
                        m_center,
                        m_left,
                        m_right,
                        m_bottom,
                        m_top,
                        new_mask_width,
                        new_mask_height,
                    )

                    mask_index += 1

                    if self.show_face_points:
                        self._show_face_points(
                            mask_data, face_landmarks, output_image, w, h
                        )

        return output_image

import cv2
from .mask_config import MASK_CONFIGS
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path


class ImageMaskProcessor:
    def __init__(self, mask_name="glasses", show_face_points=False):
        mask = MASK_CONFIGS[mask_name]

        BASE_DIR = Path(__file__).resolve().parent
        
        image_path = str(BASE_DIR / mask["file"])
        mask_src = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask_src = cv2.cvtColor(mask_src, cv2.COLOR_RGB2RGBA)

        self.m_image = mask_src
        self.m_left = mask["left"]
        self.m_right = mask["right"]
        self.m_top = mask["top"]
        self.m_bottom = mask["bottom"]
        self.m_center = mask["center"]
        self.m_scale_w = mask["scale_w"]
        self.m_scale_h = mask["scale_h"]
        self.m_start_point_x = mask["start_point_x"]
        self.m_start_point_y = mask["start_point_y"]
        self.m_rotate = mask["rotate"]
        self.m_place_on_point = mask["place_on_point"]
        self.m_native_size = mask["native_size"]

        self.selected_points = [
            self.m_left,
            self.m_right,
            self.m_top,
            self.m_bottom,
            self.m_center,
        ]

        self.show_face_points = show_face_points

        self.detector = self._get_detector()

    def _get_detector(self):
        BASE_DIR = Path(__file__).resolve().parent

        asset_path = str(BASE_DIR / "face_landmarker.task")

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

    def _show_face_points(self, face_landmarks, image, w, h):
        index = 0
        for landmark in face_landmarks:
            if index not in self.selected_points:
                index += 1
                continue

            coords = self.get_landmark_coords(landmark, w, h)
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

    def _overlay_transparent(self, background, overlay, x, y, size=None):
        bg_h, bg_w, _ = background.shape
        if size:
            overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)

        o_h, o_w, _ = overlay.shape
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + o_w, bg_w), min(y + o_h, bg_h)

        o_x1, o_y1 = x1 - x, y1 - y
        o_x2, o_y2 = o_x1 + (x2 - x1), o_y1 + (y2 - y1)

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return background

        overlay_crop = overlay[o_y1:o_y2, o_x1:o_x2]
        overlay_img = overlay_crop[:, :, :3]
        mask = overlay_crop[:, :, 3] / 255.0
        mask_inv = 1.0 - mask

        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (
                mask * overlay_img[:, :, c] + mask_inv * background[y1:y2, x1:x2, c]
            )
        return background

    def _process_image(
        self,
        image,
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
        angle_deg = 0
        scale = 1
        if self.m_native_size:
            r_h, r_w, _ = self.m_image.shape
            scale = r_w / new_mask_width
            resized_mask = cv2.resize(
                self.m_image,
                (int(new_mask_width), int(r_h / scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized_mask = cv2.resize(
                self.m_image,
                (new_mask_width, new_mask_height),
                interpolation=cv2.INTER_AREA,
            )

        if self.m_rotate:
            rotated_mask = self._rotate_z_axis(resized_mask, -angle_deg)
        else:
            rotated_mask = resized_mask

        m_h, m_w, _ = rotated_mask.shape

        if self.m_place_on_point:
            mask_x = int(m_center[0] - m_w + (int)(self.m_start_point_x / scale))
            mask_y = int(m_center[1] - m_h + (int)(self.m_start_point_y / scale))
        else:
            mask_x = int(m_center[0] - m_w / 2 + (int)(self.m_start_point_x / scale))
            mask_y = int(m_center[1] - m_h / 2 + (int)(self.m_start_point_y / scale))

        image = self._overlay_transparent(image, rotated_mask, mask_x, mask_y)

    def mask_image(self, image):
        h, w, _ = image.shape

        output_image = image

        rgb_frame = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                m_left = self._get_landmark_coords(face_landmarks[self.m_left], w, h)
                m_right = self._get_landmark_coords(face_landmarks[self.m_right], w, h)
                m_top = self._get_landmark_coords(face_landmarks[self.m_top], w, h)
                m_bottom = self._get_landmark_coords(face_landmarks[self.m_bottom], w, h)
                m_center = self._get_landmark_coords(face_landmarks[self.m_center], w, h)

                new_mask_width = int(np.linalg.norm(m_left - m_right) * self.m_scale_w)
                new_mask_height = int(np.linalg.norm(m_bottom - m_top) * self.m_scale_h)

                if new_mask_width < 10 or new_mask_height < 10:
                    continue

                self._process_image(
                    output_image,
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

                if self.show_face_points:
                    self._show_face_points(face_landmarks, output_image, w, h)

        return output_image
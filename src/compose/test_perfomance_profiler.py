import cv2
import time
import numpy as np
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import *
from ..face_detection import ImageFaceDetectorProcessor
import cProfile
import pstats

def print_result(time_list, name):
    time_list.sort()
    avg_result = sum(time_list) / len(time_list)
    max_result = max(time_list)
    median_result = np.median(time_list)
    print("-----")
    print(f"avg {name} - {avg_result:.3f}, frame_rate: {1 / avg_result:.3f}")
    print(f"max {name} - {max_result:.3f}, frame_rate: {1 / max_result:.3f}")
    print(f"median {name} - {median_result:.3f}, frame_rate: {1 / median_result:.3f}")
    print("-----")

def main():
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        new_bg = image_utils.match_background_size(frame, bg)

        background, human, segment_mask = image_segment_processor.get_segmentation(frame)

        background = image_utils.overlay_background(human, new_bg, segment_mask)

        face_data = face_detector.detect(human)

        mask = image_mask_processor.mask_image(background, face_data)

if __name__ == "__main__":
    image_mask_processor = ImageMaskProcessor("3d_text")
    image_segment_processor = ImageSegmenterProcessor()
    face_detector = ImageFaceDetectorProcessor()

    cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")
    bg = cv2.imread(f"{DATA_DIR}/vitebsk.jpg")

    with cProfile.Profile() as profile:
        main()

    results = pstats.Stats(profile)
    results.sort_stats('cumulative').print_stats()

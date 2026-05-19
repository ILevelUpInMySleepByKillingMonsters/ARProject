import cv2
import time
import numpy as np
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import *
from ..face_detection import ImageFaceDetectorProcessor


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


image_mask_processor = ImageMaskProcessor("3d_text")
image_segment_processor = ImageSegmenterProcessor()
face_detector = ImageFaceDetectorProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")
bg = cv2.imread(f"{DATA_DIR}/vitebsk.jpg")

one_frame_process_list = []
get_segmentation_list = []
process_background_list = []
mask_image_list = []
face_detector_list = []

index = 0
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    print(f"Start process - {index}")
    index += 1
    start_time = time.perf_counter()

    # start_resize_time = time.perf_counter()
    new_bg = image_utils.match_background_size(frame, bg)
    # time_utils.show_elapsed_time(start_resize_time, "resize")

    start_segment_time = time.perf_counter()
    background, human, segment_mask = image_segment_processor.get_segmentation(frame)
    elapsed_time = time_utils.show_elapsed_time(start_segment_time, "get_segmentation")
    get_segmentation_list.append(elapsed_time)

    start_process_background_time = time.perf_counter()
    background = image_utils.overlay_background(human, new_bg, segment_mask)
    elapsed_time = time_utils.show_elapsed_time(
        start_process_background_time, "process_background"
    )
    process_background_list.append(elapsed_time)

    start_face_detector_time = time.perf_counter()
    face_data = face_detector.detect(human)
    elapsed_time = time_utils.show_elapsed_time(
        start_face_detector_time, "face_detector"
    )
    face_detector_list.append(elapsed_time)

    start_mask_image_time = time.perf_counter()
    mask = image_mask_processor.mask_image(background, face_data)
    elapsed_time = time_utils.show_elapsed_time(start_mask_image_time, "mask_image")
    mask_image_list.append(elapsed_time)

    elapsed_time = time_utils.show_elapsed_time(start_time, "one_frame_process")
    one_frame_process_list.append(elapsed_time)

    print()

    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

print_result(get_segmentation_list, "segment")
print_result(process_background_list, "background")
print_result(face_detector_list, "face_detector")
print_result(mask_image_list, "mask")
print_result(one_frame_process_list, "one frame process")

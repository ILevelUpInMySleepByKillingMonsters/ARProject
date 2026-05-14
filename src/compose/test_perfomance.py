import cv2
import time
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import *

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/test-2.mp4")
bg = cv2.imread(f"{DATA_DIR}/vitebsk.jpg")

one_frame_process_list = []
get_segmentation_list = []
process_background_list = []
mask_image_list = []

index = 0
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    print(f"Start process - {index}")
    index += 1
    start_time = time.perf_counter()

    # start_resize_time = time.perf_counter()
    new_bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
    # time_utils.show_elapsed_time(start_resize_time, "resize")

    start_segment_time = time.perf_counter()
    background, human = image_segment_processor.get_segmentation(frame)
    elapsed_time = time_utils.show_elapsed_time(start_segment_time, "get_segmentation")
    get_segmentation_list.append(elapsed_time)

    start_process_background_time = time.perf_counter()
    
    background = image_utils.replace_no_white_background(human, new_bg)
    elapsed_time = time_utils.show_elapsed_time(start_process_background_time, "process_background")
    process_background_list.append(elapsed_time)

    start_mask_image_time = time.perf_counter()
    mask = image_mask_processor.mask_image(background)
    elapsed_time = time_utils.show_elapsed_time(start_mask_image_time, "mask_image")
    mask_image_list.append(elapsed_time)

    cv2.imshow("Mask", mask)

    elapsed_time = time_utils.show_elapsed_time(start_time, "one_frame_process")

    print()

    one_frame_process_list.append(elapsed_time)

    key = cv2.waitKey(1)
    if key == 27:
        break

average_one_frame_process_list = sum(one_frame_process_list) / len(one_frame_process_list)
average_get_segmentation_list = sum(get_segmentation_list) / len(get_segmentation_list)
average_process_background_list = sum(process_background_list) / len(process_background_list)
average_mask_image_list = sum(mask_image_list) / len(mask_image_list)

print(f"avg segment - {average_get_segmentation_list:.3f}, frame_rate: {1 / average_get_segmentation_list:.3f}")
print(f"avg background - {average_process_background_list:.3f}, frame_rate: {1 / average_process_background_list:.3f}")
print(f"avg mask - {average_mask_image_list:.3f}, frame_rate: {1 / average_mask_image_list:.3f}")
print(f"avg one frame process - {average_one_frame_process_list:.3f}, frame_rate: {1 / average_one_frame_process_list:.3f}")
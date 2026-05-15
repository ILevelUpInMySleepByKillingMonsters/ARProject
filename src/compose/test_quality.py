import cv2
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import image_utils
from ..face_detection import ImageFaceDetectorProcessor

image_mask_processor = ImageMaskProcessor("3d_text")
image_segment_processor = ImageSegmenterProcessor()
face_detector = ImageFaceDetectorProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")
bg_video = cv2.VideoCapture(f"{DATA_DIR}/slav_fest.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
f = True
while cap.isOpened():
    success, frame = cap.read()
    _, bg_frame = bg_video.read()

    if not success:
        break

    if f:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))
        f = False

    bg_frame = image_utils.match_background_size(frame, bg_frame)

    background, human = image_segment_processor.get_segmentation(frame)
    background = image_utils.replace_no_white_background(human, bg_frame)
    
    face_data = face_detector.detect(human)
    mask = image_mask_processor.mask_image_face_detector(background, face_data)

    # cv2.imshow("Mask", mask)
    out.write(mask)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break

out.release()
print("done")
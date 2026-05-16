import cv2
from rembg import remove, new_session


class ImageQualitySegmenterProcessor:
    def __init__(self):
        model_name = "u2net_human_seg"
        self.session = new_session(model_name=model_name)

    def _generate_mask(self, input):
        input = input.copy()

        mask = remove(input, self.session)
        is_transparent = mask[:, :, 3] <= 127
        input[is_transparent] = [255, 255, 255]
        input[~is_transparent] = [0, 0, 0]

        return input

    def get_segmentation(self, image):
        thresh = self._generate_mask(image)

        return cv2.bitwise_and(thresh, image), cv2.bitwise_or(thresh, image)

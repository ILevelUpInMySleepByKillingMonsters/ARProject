from dataclasses import dataclass

@dataclass
class FaceDetectorData:
    source_image: any
    bounding_box: BoundingBox = None

@dataclass(frozen=True)
class BoundingBox:
    origin_x: int
    origin_y: int
    width: int
    height: int
    end_x: int
    end_y: int
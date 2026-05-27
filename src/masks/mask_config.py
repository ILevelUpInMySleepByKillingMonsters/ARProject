from dataclasses import dataclass
from enum import StrEnum, auto


class PlacePivot(StrEnum):
    Center = auto()
    CenterLeft = auto()
    CenterRight = auto()
    Bottom = auto()
    BottomLeft = auto()
    BottomRight = auto()
    Top = auto()
    TopLeft = auto()
    TopRight = auto()


@dataclass(frozen=True)
class MaskConfigData:
    file: str
    left: int | None = None
    right: int | None = None
    top: int | None = None
    bottom: int | None = None
    center: int | None = None
    scale_w: int = 1
    scale_h: int = 1
    start_point_x: int | None = None
    start_point_y: int | None = None
    rotate: bool = False
    place_pivot: PlacePivot | None = None
    native_size: bool = False
    animated: bool = False
    is_fixed: bool = False


@dataclass()
class MaskFrame:
    index: int
    frames: list


MASK_CONFIGS: dict[str, list[MaskConfigData]] = {
    "test": [
        MaskConfigData(
            file="masks/bel.png",
            left=411,
            right=435,
            top=356,
            bottom=152,
            center=411,
            scale_h=1,
            scale_w=1,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Center,
            native_size=True,
            animated=False,
        ),
        MaskConfigData(
            file="masks/dog-nose.png",
            left=49,
            right=279,
            top=195,
            bottom=1,
            center=5,
            scale_h=2,
            scale_w=1,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Center,
            native_size=True,
            animated=False,
        ),
    ],
    "dog": [
        MaskConfigData(
            file="masks/dog-nose.png",
            left=49,
            right=279,
            top=195,
            bottom=1,
            center=1,
            scale_h=1,
            scale_w=1,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Center,
            native_size=True,
            animated=False,
        ),
        MaskConfigData(
            file="masks/dog-tongue.png",
            left=49,
            right=279,
            top=195,
            bottom=1,
            center=0,
            scale_h=1,
            scale_w=1,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Bottom,
            native_size=True,
            animated=False,
        ),
    ],
    "glasses": [
        MaskConfigData(
            file="masks/glasses.png",
            left=127,
            right=356,
            top=9,
            bottom=197,
            center=168,
            scale_h=1,
            scale_w=1,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Center,
            native_size=True,
            animated=False,
        ),
    ],
    "dog-tongue": [
        MaskConfigData(
            file="masks/dog-tongue.gif",
            left=127,
            right=356,
            top=9,
            bottom=197,
            center=168,
            scale_h=1,
            scale_w=1,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Center,
            native_size=True,
            animated=True,
        ),
    ],
    "3d_text": [
        MaskConfigData(
            file="masks/3d_text.mp4",
            left=127,
            right=356,
            top=9,
            bottom=197,
            center=10,
            scale_h=1,
            scale_w=3,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Top,
            native_size=True,
            animated=True,
        ),
        MaskConfigData(
            file="masks/bel-flag.gif",
            left=127,
            right=356,
            top=9,
            bottom=197,
            center=0,
            scale_h=1,
            scale_w=3,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Bottom,
            native_size=True,
            animated=True,
            is_fixed=True,
        ),
    ],
    "bear-dance": [
        MaskConfigData(
            file="masks/bear.mp4",
            left=127,
            right=356,
            top=9,
            bottom=197,
            center=0,
            scale_h=1,
            scale_w=0.5,
            start_point_x=0,
            start_point_y=0,
            rotate=True,
            place_pivot=PlacePivot.Bottom,
            native_size=True,
            animated=True,
            is_fixed=True,
        ),
    ],
    "bear-1": [
        MaskConfigData(
            file="masks/bear-1.mp4",
            scale_h=0.5,
            scale_w=0.5,
            place_pivot=PlacePivot.BottomLeft,
            animated=True,
        ),
    ],
    "bear-2": [
        MaskConfigData(
            file="masks/bear-2.mp4",
            scale_h=1,
            scale_w=1,
            place_pivot=PlacePivot.BottomLeft,
            animated=True,
        ),
    ],
}

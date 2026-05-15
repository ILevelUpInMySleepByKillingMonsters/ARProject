from dataclasses import dataclass
from enum import StrEnum, auto


class PlacePivot(StrEnum):
    Center = auto()
    Bottom = auto()
    Top = auto()


@dataclass(frozen=True)
class MaskConfigData:
    file: str
    left: int
    right: int
    top: int
    bottom: int
    center: int
    scale_w: int
    scale_h: int
    start_point_x: int
    start_point_y: int
    rotate: bool
    place_pivot: PlacePivot
    native_size: bool
    animated: bool


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
}

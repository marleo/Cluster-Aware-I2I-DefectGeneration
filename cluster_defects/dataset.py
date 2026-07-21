from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def expand(self, image_width: int, image_height: int, fraction: float) -> "YoloBox":
        return YoloBox(
            class_id=self.class_id,
            x1=max(0.0, self.x1 - self.width * fraction),
            y1=max(0.0, self.y1 - self.height * fraction),
            x2=min(float(image_width), self.x2 + self.width * fraction),
            y2=min(float(image_height), self.y2 + self.height * fraction),
        )

    def scale(self, x_scale: float, y_scale: float) -> "YoloBox":
        return YoloBox(
            class_id=self.class_id,
            x1=self.x1 * x_scale,
            y1=self.y1 * y_scale,
            x2=self.x2 * x_scale,
            y2=self.y2 * y_scale,
        )


def list_images(directory: Path, extensions: Iterable[str]) -> list[Path]:
    normalized = {extension.lower() for extension in extensions}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )


def matching_label_path(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    return (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")


def read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id = int(float(values[0]))
        xc, yc, width, height = map(float, values[1:5])
        boxes.append(
            YoloBox(
                class_id=class_id,
                x1=max(0.0, (xc - width / 2.0) * image_width),
                y1=max(0.0, (yc - height / 2.0) * image_height),
                x2=min(float(image_width), (xc + width / 2.0) * image_width),
                y2=min(float(image_height), (yc + height / 2.0) * image_height),
            )
        )
    return boxes


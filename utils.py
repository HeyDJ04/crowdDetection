import json
import cv2
import numpy as np


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_source(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def normalized_polygon_to_pixels(points, width, height):
    converted = []
    for x, y in points:
        converted.append([int(x * width), int(y * height)])
    return np.array(converted, dtype=np.int32)


def point_in_polygon(point, polygon):
    if polygon is None:
        return True
    x, y = point
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def resize_keep_aspect(frame, max_width=1280):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

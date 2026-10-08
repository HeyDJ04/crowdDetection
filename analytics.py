from collections import defaultdict, deque
from .utils import center_of_box, point_in_polygon


class CrowdAnalytics:
    def __init__(self, config):
        self.crowd_cfg = config["crowd"]
        self.line_cfg = config["line_counting"]
        self.display_cfg = config["display"]

        self.unique_ids = set()
        self.previous_side = {}
        self.entered = 0
        self.exited = 0

        trail_length = int(self.display_cfg.get("trail_length", 20))
        self.trails = defaultdict(lambda: deque(maxlen=trail_length))

    def reset(self):
        self.unique_ids.clear()
        self.previous_side.clear()
        self.entered = 0
        self.exited = 0
        self.trails.clear()

    def filter_roi(self, detections, polygon):
        filtered = []
        for det in detections:
            center = center_of_box(det["box"])
            det["center"] = center
            if point_in_polygon(center, polygon):
                filtered.append(det)
        return filtered

    def update_tracks(self, detections, frame_width, frame_height):
        for det in detections:
            track_id = det.get("track_id")
            center = det.get("center") or center_of_box(det["box"])

            if track_id is not None:
                self.unique_ids.add(track_id)
                self.trails[track_id].append(center)

                if self.line_cfg.get("enabled", True):
                    self._update_line_crossing(track_id, center, frame_width, frame_height)

    def _update_line_crossing(self, track_id, center, frame_width, frame_height):
        orientation = self.line_cfg.get("orientation", "horizontal")
        ratio = float(self.line_cfg.get("position_ratio", 0.55))
        dead_zone = int(self.line_cfg.get("dead_zone_pixels", 8))

        if orientation == "vertical":
            line_pos = int(frame_width * ratio)
            delta = center[0] - line_pos
        else:
            line_pos = int(frame_height * ratio)
            delta = center[1] - line_pos

        if abs(delta) <= dead_zone:
            side = 0
        elif delta < 0:
            side = -1
        else:
            side = 1

        previous = self.previous_side.get(track_id)

        if previous in (-1, 1) and side in (-1, 1) and previous != side:
            if previous == -1 and side == 1:
                self.entered += 1
            elif previous == 1 and side == -1:
                self.exited += 1

        if side != 0:
            self.previous_side[track_id] = side

    def crowd_status(self, current_count):
        capacity = max(1, int(self.crowd_cfg["capacity"]))
        warning_ratio = float(self.crowd_cfg["warning_ratio"])
        critical_ratio = float(self.crowd_cfg["critical_ratio"])
        ratio = current_count / capacity

        if ratio >= critical_ratio:
            return "CRITICAL", ratio
        if ratio >= warning_ratio:
            return "WARNING", ratio
        return "NORMAL", ratio

    def should_alert(self, current_count, status):
        minimum = int(self.crowd_cfg.get("minimum_people_for_alert", 1))
        return current_count >= minimum and status in ("WARNING", "CRITICAL")

    def stats(self, current_count):
        status, ratio = self.crowd_status(current_count)
        return {
            "current_count": current_count,
            "unique_count": len(self.unique_ids),
            "entered": self.entered,
            "exited": self.exited,
            "status": status,
            "occupancy_ratio": ratio
        }

import cv2
import numpy as np


STATUS_COLORS = {
    "NORMAL": (60, 180, 75),
    "WARNING": (0, 190, 255),
    "CRITICAL": (50, 50, 230)
}


def draw_roi(frame, polygon):
    if polygon is None:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (60, 140, 255))
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
    cv2.polylines(frame, [polygon], True, (60, 140, 255), 2, cv2.LINE_AA)


def draw_counting_line(frame, config):
    if not config.get("enabled", True):
        return

    h, w = frame.shape[:2]
    ratio = float(config.get("position_ratio", 0.55))
    orientation = config.get("orientation", "horizontal")

    if orientation == "vertical":
        x = int(w * ratio)
        cv2.line(frame, (x, 0), (x, h), (220, 220, 220), 2)
    else:
        y = int(h * ratio)
        cv2.line(frame, (0, y), (w, y), (220, 220, 220), 2)


def draw_detections(frame, detections, trails, show_ids=True, show_trails=True):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["confidence"]
        track_id = det.get("track_id")
        center = det.get("center")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)

        label = f"Person {conf:.2f}"
        if show_ids and track_id is not None:
            label += f" | ID {track_id}"

        cv2.putText(
            frame, label, (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2, cv2.LINE_AA
        )

        if center:
            cv2.circle(frame, center, 3, (255, 255, 255), -1)

        if show_trails and track_id is not None and track_id in trails:
            pts = list(trails[track_id])
            if len(pts) > 1:
                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_np], False, (255, 255, 255), 2, cv2.LINE_AA)


def draw_dashboard(frame, stats, fps=None):
    status = stats["status"]
    color = STATUS_COLORS.get(status, (255, 255, 255))

    panel = frame.copy()
    cv2.rectangle(panel, (18, 18), (410, 194), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)

    rows = [
        ("CROWD STATUS", status),
        ("CURRENT PEOPLE", str(stats["current_count"])),
        ("UNIQUE TRACKED", str(stats["unique_count"])),
        ("ENTERED", str(stats["entered"])),
        ("EXITED", str(stats["exited"]))
    ]

    y = 48
    for i, (label, value) in enumerate(rows):
        value_color = color if i == 0 else (255, 255, 255)
        cv2.putText(frame, label, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (195, 195, 195), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (220, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, value_color, 2, cv2.LINE_AA)
        y += 29

    if fps is not None:
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (35, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (210, 210, 210), 1, cv2.LINE_AA
        )


def draw_alert(frame, status):
    if status not in ("WARNING", "CRITICAL"):
        return

    h, w = frame.shape[:2]
    color = STATUS_COLORS[status]
    text = "CROWD WARNING" if status == "WARNING" else "OVERCROWDING ALERT"

    cv2.rectangle(frame, (0, h - 72), (w, h), color, -1)
    cv2.putText(
        frame, text, (24, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA
    )

import argparse
import time
from pathlib import Path

import cv2

from crowd_detection.detector import CrowdDetector
from crowd_detection.analytics import CrowdAnalytics
from crowd_detection.logger import EventLogger
from crowd_detection.drawing import (
    draw_roi,
    draw_counting_line,
    draw_detections,
    draw_dashboard,
    draw_alert,
)
from crowd_detection.utils import (
    load_config,
    parse_source,
    normalized_polygon_to_pixels,
    resize_keep_aspect,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Advanced Crowd Detection System")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    parser.add_argument("--config", default="config/config.json", help="Path to JSON configuration")
    parser.add_argument("--record", action="store_true", help="Save annotated output video")
    return parser


def create_writer(frame, capture, cfg, path):
    h, w = frame.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = float(cfg["recording"].get("fps_fallback", 25.0))

    codec_name = cfg["recording"].get("codec", "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*codec_name)
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)

    detector = CrowdDetector(cfg["model"])
    analytics = CrowdAnalytics(cfg)
    logger = EventLogger(
        "output/logs/events.csv",
        "output/snapshots"
    )

    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    writer = None
    paused = False
    last_alert_time = 0.0
    previous_time = time.perf_counter()
    fps = 0.0
    polygon = None

    window_name = cfg["display"].get("window_name", "Advanced Crowd Detection")

    while True:
        if not paused:
            ok, frame = capture.read()
            if not ok:
                break

            frame = resize_keep_aspect(frame, 1280)
            h, w = frame.shape[:2]

            if cfg["roi"].get("enabled", True):
                polygon = normalized_polygon_to_pixels(
                    cfg["roi"]["normalized_points"], w, h
                )
            else:
                polygon = None

            detections = detector.track_people(frame)
            inside = analytics.filter_roi(detections, polygon)
            analytics.update_tracks(inside, w, h)
            stats = analytics.stats(len(inside))

            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-6)
            instant_fps = 1.0 / elapsed
            fps = (fps * 0.9) + (instant_fps * 0.1) if fps else instant_fps
            previous_time = now

            if cfg["display"].get("show_roi", True):
                draw_roi(frame, polygon)

            draw_counting_line(frame, cfg["line_counting"])

            draw_detections(
                frame,
                inside,
                analytics.trails,
                show_ids=cfg["display"].get("show_ids", True),
                show_trails=cfg["display"].get("show_trails", True)
            )

            draw_dashboard(
                frame,
                stats,
                fps if cfg["display"].get("show_fps", True) else None
            )

            if analytics.should_alert(stats["current_count"], stats["status"]):
                draw_alert(frame, stats["status"])

                cooldown = float(cfg["crowd"].get("alert_cooldown_seconds", 15))
                if time.time() - last_alert_time >= cooldown:
                    snapshot = logger.save_snapshot(frame, prefix=stats["status"].lower())
                    logger.log("automatic_alert", stats, snapshot)
                    last_alert_time = time.time()

            if args.record:
                if writer is None:
                    Path("output").mkdir(parents=True, exist_ok=True)
                    writer = create_writer(
                        frame,
                        capture,
                        cfg,
                        Path("output/annotated_output.mp4")
                    )
                writer.write(frame)

            current_frame = frame.copy()

        if "current_frame" in locals():
            cv2.imshow(window_name, current_frame)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("p"):
            paused = not paused

        elif key == ord("s") and "current_frame" in locals():
            if "stats" not in locals():
                stats = analytics.stats(0)
            snapshot = logger.save_snapshot(current_frame, prefix="manual")
            logger.log("manual_snapshot", stats, snapshot)
            print(f"Snapshot saved: {snapshot}")

        elif key == ord("r"):
            analytics.reset()
            print("Counters reset.")

    capture.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

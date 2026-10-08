import csv
from pathlib import Path
from datetime import datetime
import cv2


class EventLogger:
    def __init__(self, log_path, snapshot_dir):
        self.log_path = Path(log_path)
        self.snapshot_dir = Path(snapshot_dir)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "status",
                    "current_count",
                    "unique_count",
                    "entered",
                    "exited",
                    "snapshot"
                ])

    def save_snapshot(self, frame, prefix="snapshot"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.snapshot_dir / f"{prefix}_{stamp}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def log(self, event_type, stats, snapshot=""):
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                event_type,
                stats["status"],
                stats["current_count"],
                stats["unique_count"],
                stats["entered"],
                stats["exited"],
                snapshot
            ])

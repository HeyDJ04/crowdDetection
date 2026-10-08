from ultralytics import YOLO


class CrowdDetector:
    def __init__(self, model_config):
        self.model_name = model_config["name"]
        self.confidence = float(model_config.get("confidence", 0.35))
        self.iou = float(model_config.get("iou", 0.45))
        self.device = model_config.get("device", "auto")
        self.tracker = model_config.get("tracker", "bytetrack.yaml")

        self.model = YOLO(self.model_name)

    def track_people(self, frame):
        kwargs = {
            "source": frame,
            "persist": True,
            "classes": [0],
            "conf": self.confidence,
            "iou": self.iou,
            "tracker": self.tracker,
            "verbose": False
        }

        if self.device and self.device != "auto":
            kwargs["device"] = self.device

        results = self.model.track(**kwargs)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes

        detections = []
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

        for box, conf, track_id in zip(xyxy, confs, ids):
            detections.append({
                "box": [int(v) for v in box.tolist()],
                "confidence": float(conf),
                "track_id": None if track_id is None else int(track_id)
            })

        return detections

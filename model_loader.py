"""Load local YOLO weights through the prediction interface used by the reader."""

from types import SimpleNamespace
import hashlib


class LocalGaugeModel:
    def __init__(self, path):
        if not path.is_file():
            raise FileNotFoundError(f"Local model not found: {path}. Copy Colab best.pt here.")
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        names = set(self.model.names.values())
        if not ({"gauge", "pointer"}.issubset(names) or {"base", "circle_plate", "tip"}.issubset(names)):
            raise ValueError(f"Unsupported model classes: {sorted(names)}")
        self._frame = None
        self._confidence = None
        self._result = None

    def infer(self, frame, confidence=0.05):
        # Needle and gauge consumers share one prediction for the same frame.
        if frame is self._frame and confidence == self._confidence:
            return self._result
        result = self.model.predict(source=frame, conf=confidence, verbose=False)[0]
        predictions = []
        boxes = result.obb if self.model.task == "obb" else result.boxes
        if boxes is not None:
            for box in boxes:
                name = result.names[int(box.cls.item())]
                if name == "pointer_centre":
                    name = "base"
                elif name == "pointer_end":
                    name = "tip"
                # Other classes are not reliable enough to drive the existing reader.
                if name not in {"gauge", "pointer", "base", "circle_plate", "tip"}:
                    continue
                if self.model.task == "obb":
                    corners = box.xyxyxyxy[0]
                    low, high = corners.amin(dim=0), corners.amax(dim=0)
                    x, y = ((low + high) / 2).tolist()
                    width, height = (high - low).tolist()
                else:
                    x, y, width, height = box.xywh[0].tolist()
                predictions.append(SimpleNamespace(
                    class_name=name, confidence=float(box.conf.item()),
                    x=x, y=y, width=width, height=height,
                ))
        self._frame, self._confidence = frame, confidence
        self._result = [SimpleNamespace(predictions=predictions)]
        return self._result


class LocalModelEnsemble:
    """Share cached predictions and avoid running identical checkpoints twice."""
    def __init__(self, paths):
        self.models = []
        seen = set()
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                print(f"Duplicate checkpoint shares inference: {path.name}")
                continue
            seen.add(digest)
            self.models.append(LocalGaugeModel(path))

    def infer(self, frame, confidence=0.05):
        predictions = []
        for model in self.models:
            predictions.extend(model.infer(frame, confidence)[0].predictions)
        # Associate endpoints only inside the same detected dial. Endpoint centers
        # are landmarks, not full pointer boxes.
        dials = [p for p in predictions if p.class_name == "circle_plate" and p.confidence >= 0.25]
        for dial in dials:
            inside = lambda p: abs(p.x-dial.x) <= dial.width/2 and abs(p.y-dial.y) <= dial.height/2
            bases = [p for p in predictions if p.class_name == "base" and p.confidence >= 0.25 and inside(p)]
            tips = [p for p in predictions if p.class_name == "tip" and p.confidence >= 0.25 and inside(p)]
            if not bases or not tips:
                continue
            base, tip = max(bases, key=lambda p:p.confidence), max(tips, key=lambda p:p.confidence)
            import math
            if not 15 <= math.hypot(tip.x-base.x, tip.y-base.y) <= math.hypot(dial.width, dial.height)/2:
                continue
            predictions.append(SimpleNamespace(
                class_name="pointer", confidence=min(base.confidence, tip.confidence),
                x=(base.x+tip.x)/2, y=(base.y+tip.y)/2,
                width=max(24, abs(tip.x-base.x)+18), height=max(24, abs(tip.y-base.y)+18),
                landmark_center=(base.x,base.y), landmark_tip=(tip.x,tip.y),
            ))
        return [SimpleNamespace(predictions=predictions)]



def load_models():
    from config import (
        MODEL_BACKEND, LOCAL_MODEL_PATH, LOCAL_OBB_PATHS, ROBOFLOW_API_KEY,
        NEEDLE_MODEL_ID, GAUGE_MODEL_ID,
    )
    if MODEL_BACKEND == "local":
        print(f"Loading local YOLO model: {str(LOCAL_MODEL_PATH)!a}")
        model = LocalModelEnsemble([LOCAL_MODEL_PATH, *LOCAL_OBB_PATHS])
        return model, model

    from inference import get_model
    needle = get_model(model_id=NEEDLE_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    try:
        gauge = get_model(model_id=GAUGE_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    except Exception as error:
        gauge = None
        print(f"Gauge model unavailable; using computer-vision fallback: {error}")
    return needle, gauge

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from autocam_tracker.detection.detection_models import VehicleDetection


class RuntimeReIDFeatureExtractor:
    def __init__(
        self,
        model_path: Path | None,
        imgsz: int = 224,
        batch_size: int = 32,
        device: int | str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.requested_device = device
        self._model = None
        self._torch = None
        self._functional = None
        self._device = None
        self._disabled = self.model_path is None or not self.model_path.exists()
        self._warned = False

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def encode_detections(self, detections: list[VehicleDetection]) -> None:
        if self._disabled:
            return

        indexed_images = [
            (index, detection.thumbnail)
            for index, detection in enumerate(detections)
            if detection.thumbnail is not None
        ]
        if not indexed_images:
            return

        try:
            model = self._get_model()
            features = self._encode_images([image for _, image in indexed_images], model)
        except Exception as exc:
            self._disabled = True
            if not self._warned:
                print(f"Warning: Runtime GID ReID feature extraction disabled: {exc}")
                self._warned = True
            return

        for (index, _), feature in zip(indexed_images, features):
            detections[index].reid_feature = feature

    def _get_model(self):
        if self._model is not None:
            return self._model

        import torch
        import torch.nn.functional as F

        self._torch = torch
        self._functional = F
        self._device = self._resolve_device(torch)
        print(f"Loading {self.model_path} for runtime GID ReID feature memory...")
        self._model = torch.jit.load(str(self.model_path), map_location=self._device).eval().to(self._device)
        return self._model

    def _resolve_device(self, torch):
        if self.requested_device in (None, "", "auto"):
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if isinstance(self.requested_device, int):
            return torch.device(f"cuda:{self.requested_device}" if torch.cuda.is_available() else "cpu")

        text = str(self.requested_device).strip().lower()
        if text.isdigit():
            return torch.device(f"cuda:{text}" if torch.cuda.is_available() else "cpu")
        if text == "cuda":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if text.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(text)

    def _encode_images(self, images: list[np.ndarray], model) -> np.ndarray:
        assert self._torch is not None
        assert self._functional is not None
        assert self._device is not None

        outputs = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            batch = self._torch.stack([self._image_tensor(image) for image in batch_images]).to(self._device)
            with self._torch.no_grad():
                features = model(batch)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = self._functional.normalize(features, p=2, dim=1)
                outputs.append(features.detach().cpu().numpy().astype("float32"))
        if not outputs:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _image_tensor(self, image: np.ndarray):
        assert self._torch is not None

        resized = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        data = np.ascontiguousarray(rgb)
        tensor = self._torch.from_numpy(data).permute(2, 0, 1).contiguous()
        return tensor.float().div(255.0)

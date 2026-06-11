from __future__ import annotations

import cv2
import numpy as np

from autocam_tracker.detection.detection_models import VehicleDetection


def draw_detections(frame: np.ndarray, detections: list[VehicleDetection], selected_global_vehicle_id: int) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        x, y, w, h = detection.bbox
        is_selected = detection.selected or (
            selected_global_vehicle_id >= 0 and detection.global_vehicle_id == selected_global_vehicle_id
        )
        color = (0, 220, 80) if is_selected else (0, 180, 255)
        thickness = 3 if is_selected else 2
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        label = f"{detection.label} {detection.confidence:.2f}"
        if detection.local_track_id >= 0:
            label += f" L{detection.local_track_id}"
        if detection.global_vehicle_id >= 0:
            label += f" G{detection.global_vehicle_id}"
        cv2.putText(output, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def bgr_to_tk_image(frame: np.ndarray, max_size: tuple[int, int]):
    from PIL import Image, ImageTk

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


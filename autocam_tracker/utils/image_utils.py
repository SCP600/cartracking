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
        if detection.global_vehicle_id >= 0:
            label = f"G{detection.global_vehicle_id}"
            font_scale = 1.5 if is_selected else 1.2
            font_thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_y = max(text_height + 5, y - 8)
            
            # Draw background rectangle
            bg_color = (0, 100, 0) if is_selected else (0, 0, 0)
            cv2.rectangle(output, (x, text_y - text_height - 5), (x + text_width, text_y + baseline - 5), bg_color, -1)
            
            # Draw text
            text_color = (0, 255, 255) if is_selected else (255, 255, 255)
            cv2.putText(output, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return output


def bgr_to_tk_image(frame: np.ndarray, max_size: tuple[int, int]):
    from PIL import Image, ImageTk

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


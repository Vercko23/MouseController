import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Dict, Any, Tuple, List

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

# Standard 21 Hand Landmark Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

class HandTracker:
    """
    Encapsulates MediaPipe HandLandmarker for real-time video stream detection,
    coordinate extraction, distance measurement, and HUD rendering.
    """
    def __init__(self, model_path: str = MODEL_FILENAME, min_confidence: float = 0.5):
        self.model_path = model_path
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"[HandTracker] Downloading MediaPipe model to {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("[HandTracker] Model downloaded successfully.")

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        """
        Processes a BGR video frame and returns extracted keypoints and gestures if hand is found.
        """
        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not detection_result.hand_landmarks:
            return None

        landmarks = detection_result.hand_landmarks[0]
        h, w, _ = frame_bgr.shape

        # Key Landmarks:
        # 0: Wrist, 4: Thumb Tip, 8: Index Tip, 9: Middle MCP, 12: Middle Tip, 20: Pinky Tip
        lm_wrist = landmarks[0]
        lm_thumb_tip = landmarks[4]
        lm_index_tip = landmarks[8]
        lm_middle_mcp = landmarks[9]
        lm_middle_tip = landmarks[12]
        lm_pinky_tip = landmarks[20]

        # Normalized coordinates
        index_x, index_y = lm_index_tip.x, lm_index_tip.y
        thumb_x, thumb_y = lm_thumb_tip.x, lm_thumb_tip.y
        middle_x, middle_y = lm_middle_tip.x, lm_middle_tip.y
        pinky_x, pinky_y = lm_pinky_tip.x, lm_pinky_tip.y

        # Pixel coordinates
        px_index = (int(index_x * w), int(index_y * h))
        px_thumb = (int(thumb_x * w), int(thumb_y * h))
        px_middle = (int(middle_x * w), int(middle_y * h))
        px_pinky = (int(pinky_x * w), int(pinky_y * h))

        # Hand reference scale (distance between Wrist and Middle MCP)
        hand_scale_norm = np.hypot(lm_wrist.x - lm_middle_mcp.x, lm_wrist.y - lm_middle_mcp.y)
        if hand_scale_norm < 1e-4:
            hand_scale_norm = 0.2

        # 1. Left Click Distance: Thumb Tip <-> Middle Finger Tip
        dist_thumb_middle_norm = np.hypot(thumb_x - middle_x, thumb_y - middle_y)
        dist_thumb_middle_px = np.hypot(px_thumb[0] - px_middle[0], px_thumb[1] - px_middle[1])
        left_pinch_ratio = dist_thumb_middle_norm / hand_scale_norm
        is_left_pinched = left_pinch_ratio < 0.28

        # 2. Right Click Distance: Thumb Tip <-> Pinky Finger Tip
        dist_thumb_pinky_norm = np.hypot(thumb_x - pinky_x, thumb_y - pinky_y)
        dist_thumb_pinky_px = np.hypot(px_thumb[0] - px_pinky[0], px_thumb[1] - px_pinky[1])
        right_pinch_ratio = dist_thumb_pinky_norm / hand_scale_norm
        is_right_pinched = right_pinch_ratio < 0.28

        return {
            "raw_landmarks": landmarks,
            "index_norm": (index_x, index_y),
            "thumb_norm": (thumb_x, thumb_y),
            "middle_norm": (middle_x, middle_y),
            "pinky_norm": (pinky_x, pinky_y),
            "index_px": px_index,
            "thumb_px": px_thumb,
            "middle_px": px_middle,
            "pinky_px": px_pinky,
            "left_pinch_ratio": left_pinch_ratio,
            "right_pinch_ratio": right_pinch_ratio,
            "dist_left_px": dist_thumb_middle_px,
            "dist_right_px": dist_thumb_pinky_px,
            "is_left_pinched": is_left_pinched,
            "is_right_pinched": is_right_pinched,
            # Backwards compatibility:
            "is_pinched": is_left_pinched
        }

    def draw_hud(
        self,
        frame: np.ndarray,
        hand_data: Optional[Dict[str, Any]],
        margin_x: float,
        margin_y: float,
        fps: float,
        left_clicks: int = 0,
        right_clicks: int = 0,
        is_left_pinched: bool = False,
        is_right_pinched: bool = False,
        # Backwards compatible signature:
        click_count: Optional[int] = None,
        is_pinched: Optional[bool] = None
    ) -> np.ndarray:
        """
        Renders HUD overlays: active zone, skeleton, landmark highlights, and state labels.
        """
        if click_count is not None and left_clicks == 0:
            left_clicks = click_count
        if is_pinched is not None:
            is_left_pinched = is_pinched

        h, w, _ = frame.shape

        # 1. Draw Active Zone Box (interaction region)
        box_x1 = int(margin_x * w)
        box_y1 = int(margin_y * h)
        box_x2 = int((1.0 - margin_x) * w)
        box_y2 = int((1.0 - margin_y) * h)

        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 200, 0), 2)
        cv2.putText(frame, "Active Area", (box_x1 + 6, box_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

        if hand_data is not None:
            landmarks = hand_data["raw_landmarks"]

            # Convert all landmarks to pixel coordinates
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # 2. Draw Skeleton Connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = pts[start_idx]
                p2 = pts[end_idx]
                cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)

            # Draw small joint circles
            for p in pts:
                cv2.circle(frame, p, 3, (120, 220, 120), -1, cv2.LINE_AA)

            # 3. Highlight Index Finger Tip (Cursor Guide)
            idx_pt = hand_data["index_px"]
            cv2.circle(frame, idx_pt, 12, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, idx_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "Cursor (Index)", (idx_pt[0] + 12, idx_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # 4. Highlight Thumb, Middle (Left Click) and Pinky (Right Click)
            th_pt = hand_data["thumb_px"]
            mid_pt = hand_data["middle_px"]
            pin_pt = hand_data["pinky_px"]

            # --- Left Click Visuals (Thumb + Middle) ---
            if is_left_pinched:
                cv2.circle(frame, th_pt, 10, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, mid_pt, 10, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(frame, th_pt, mid_pt, (0, 0, 255), 3, cv2.LINE_AA)
                center_pt = ((th_pt[0] + mid_pt[0]) // 2, (th_pt[1] + mid_pt[1]) // 2)
                cv2.putText(frame, "LEFT CLICK!", (center_pt[0] - 35, center_pt[1] - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, mid_pt, 7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(frame, th_pt, mid_pt, (0, 255, 0), 1, cv2.LINE_AA)

            # --- Right Click Visuals (Thumb + Pinky) ---
            if is_right_pinched:
                cv2.circle(frame, th_pt, 10, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pin_pt, 10, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.line(frame, th_pt, pin_pt, (255, 0, 255), 3, cv2.LINE_AA)
                center_r_pt = ((th_pt[0] + pin_pt[0]) // 2, (th_pt[1] + pin_pt[1]) // 2)
                cv2.putText(frame, "RIGHT CLICK!", (center_r_pt[0] - 40, center_r_pt[1] - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, pin_pt, 7, (255, 150, 0), 2, cv2.LINE_AA)
                cv2.line(frame, th_pt, pin_pt, (255, 150, 0), 1, cv2.LINE_AA)

            if not is_left_pinched and not is_right_pinched:
                cv2.circle(frame, th_pt, 8, (200, 200, 200), 2, cv2.LINE_AA)

        # 5. Top Bar HUD Info
        cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        l_text = f"L: {left_clicks}"
        cv2.putText(frame, l_text, (105, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2, cv2.LINE_AA)

        r_text = f"R: {right_clicks}"
        cv2.putText(frame, r_text, (175, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 255), 2, cv2.LINE_AA)

        if is_left_pinched:
            status_text = "LEFT CLICK"
            status_color = (0, 0, 255)
        elif is_right_pinched:
            status_text = "RIGHT CLICK"
            status_color = (255, 0, 255)
        else:
            status_text = "MOVING"
            status_color = (200, 200, 200)

        cv2.putText(frame, f"Mode: {status_text}", (250, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        exit_hint = "Exit: 'q' / ESC"
        cv2.putText(frame, exit_hint, (w - 140, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    def close(self):
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()

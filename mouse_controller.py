import ctypes
import pyautogui
from typing import Tuple

# Disable PyAutoGUI delay and fail-safe crash
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

class MouseController:
    """
    Handles mouse cursor positioning and clicking with smoothing and active zone mapping.
    """
    def __init__(self, smooth_factor: float = 0.25, margin_x: float = 0.15, margin_y: float = 0.15):
        """
        :param smooth_factor: Smoothing interpolation factor alpha in (0, 1].
                              Lower = smoother (less jitter), Higher = faster response.
        :param margin_x: Horizontal margin of interaction zone in normalized coordinates [0, 0.5).
        :param margin_y: Vertical margin of interaction zone in normalized coordinates [0, 0.5).
        """
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_factor = smooth_factor
        self.margin_x = margin_x
        self.margin_y = margin_y

        self.curr_x = float(self.screen_w / 2)
        self.curr_y = float(self.screen_h / 2)

        # Click state machine with hysteresis for Left and Right clicks
        self.is_left_pinched = False
        self.is_right_pinched = False
        self.left_click_count = 0
        self.right_click_count = 0

    @property
    def click_count(self) -> int:
        return self.left_click_count + self.right_click_count

    def map_to_screen(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """
        Maps normalized camera coordinates (within the active interaction box)
        to full screen pixel coordinates.
        """
        # Active box bounds
        min_x = self.margin_x
        max_x = 1.0 - self.margin_x
        min_y = self.margin_y
        max_y = 1.0 - self.margin_y

        # Clamp inside active box
        clamped_x = max(min_x, min(norm_x, max_x))
        clamped_y = max(min_y, min(norm_y, max_y))

        # Normalize to [0, 1] relative to the active box
        rel_x = (clamped_x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
        rel_y = (clamped_y - min_y) / (max_y - min_y) if max_y > min_y else 0.5

        target_x = rel_x * self.screen_w
        target_y = rel_y * self.screen_h

        return target_x, target_y

    def move_cursor(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Applies exponential moving average smoothing and moves the mouse cursor.
        """
        target_x, target_y = self.map_to_screen(norm_x, norm_y)

        # Dynamic smoothing: if sudden large movement, respond faster
        dx = target_x - self.curr_x
        dy = target_y - self.curr_y
        dist = (dx * dx + dy * dy) ** 0.5

        # Adapt alpha: higher alpha for larger movements, lower alpha for tiny jitters
        alpha = self.smooth_factor
        if dist > 100:
            alpha = min(1.0, self.smooth_factor * 2.0)

        self.curr_x += alpha * dx
        self.curr_y += alpha * dy

        screen_x = int(max(0, min(self.curr_x, self.screen_w - 1)))
        screen_y = int(max(0, min(self.curr_y, self.screen_h - 1)))

        # Use Windows fast cursor API for zero-latency moving
        try:
            ctypes.windll.user32.SetCursorPos(screen_x, screen_y)
        except Exception:
            pyautogui.moveTo(screen_x, screen_y)

        return screen_x, screen_y

    def update_click_states(self, is_left_pinch: bool, is_right_pinch: bool) -> Tuple[bool, bool]:
        """
        State machine for left and right mouse clicks with real hold/drag support:
        - When pinch begins: sends MOUSEEVENTF_*DOWN (button is held down in the OS).
        - While pinched: button stays held down in the OS (allows dragging/selecting).
        - When pinch is released: sends MOUSEEVENTF_*UP (button is released).
        Returns (left_just_pressed, right_just_pressed).
        """
        left_triggered = False
        right_triggered = False

        # --- Left Click / Hold (Thumb + Middle) ---
        if is_left_pinch and not self.is_left_pinched:
            self.is_left_pinched = True
            self.left_click_count += 1
            left_triggered = True
            try:
                # Fast Win32 mouse button press down: MOUSEEVENTF_LEFTDOWN = 0x0002
                ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            except Exception:
                pyautogui.mouseDown(button='left')
        elif not is_left_pinch and self.is_left_pinched:
            self.is_left_pinched = False
            try:
                # Fast Win32 mouse button release: MOUSEEVENTF_LEFTUP = 0x0004
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            except Exception:
                pyautogui.mouseUp(button='left')

        # --- Right Click / Hold (Thumb + Pinky) ---
        if is_right_pinch and not self.is_right_pinched:
            self.is_right_pinched = True
            self.right_click_count += 1
            right_triggered = True
            try:
                # Fast Win32 mouse button press down: MOUSEEVENTF_RIGHTDOWN = 0x0008
                ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)
            except Exception:
                pyautogui.mouseDown(button='right')
        elif not is_right_pinch and self.is_right_pinched:
            self.is_right_pinched = False
            try:
                # Fast Win32 mouse button release: MOUSEEVENTF_RIGHTUP = 0x0010
                ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)
            except Exception:
                pyautogui.mouseUp(button='right')

        return left_triggered, right_triggered

    def release_all(self):
        """
        Safety method to release all held buttons (e.g. when hand is lost or program exits).
        """
        if self.is_left_pinched:
            self.is_left_pinched = False
            try:
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0) # LEFTUP
            except Exception:
                pyautogui.mouseUp(button='left')

        if self.is_right_pinched:
            self.is_right_pinched = False
            try:
                ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0) # RIGHTUP
            except Exception:
                pyautogui.mouseUp(button='right')

    def update_click_state(self, is_pinch_detected: bool) -> bool:
        """
        Backwards-compatible method for left click only.
        """
        left_clicked, _ = self.update_click_states(is_pinch_detected, False)
        return left_clicked

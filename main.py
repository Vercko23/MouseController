import cv2
import time
import argparse
from hand_tracker import HandTracker
from mouse_controller import MouseController

def parse_args():
    parser = argparse.ArgumentParser(description="CV Virtual Mouse - Hand Tracking Cursor & Pinch Click")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height (default: 480)")
    parser.add_argument("--smooth", type=float, default=0.25, help="Smoothing alpha [0.05 - 1.0] (default: 0.25)")
    parser.add_argument("--margin-x", type=float, default=0.15, help="Horizontal active margin ratio (default: 0.15)")
    parser.add_argument("--margin-y", type=float, default=0.15, help="Vertical active margin ratio (default: 0.15)")
    parser.add_argument("--no-mouse", action="store_true", help="Run in test mode without actually moving the OS mouse cursor")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("🖐️  CV Virtual Mouse Starting...")
    print(f"📷  Camera index: {args.camera} ({args.width}x{args.height})")
    print(f"⚙️   Smoothing factor: {args.smooth}")
    print(f"📐  Active margins: X={args.margin_x}, Y={args.margin_y}")
    print(f"🖱️   Mouse control: {'DISABLED (Preview Only)' if args.no_mouse else 'ENABLED'}")
    print("👉  Указательный палец: перемещение курсора")
    print("🤏  Большой + Средний пальцы: ЛЕВЫЙ клик (ЛКМ)")
    print("🤙  Большой + Мизинец: ПРАВЫЙ клик (ПКМ)")
    print("⌨️   Нажмите 'q' или 'ESC' в окне камеры для выхода")
    print("=" * 60)

    # Initialize tracker and mouse controller
    tracker = HandTracker()
    mouse = MouseController(smooth_factor=args.smooth, margin_x=args.margin_x, margin_y=args.margin_y)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if cv2.CAP_DSHOW else 0)
    if not cap.isOpened():
        # Fallback to default backend if CAP_DSHOW fails
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[Error] Could not open camera with index {args.camera}. Please check connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    prev_time = time.time()
    fps = 0.0
    start_timestamp_ms = int(time.time() * 1000)

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("[Warning] Failed to grab frame. Retrying...")
                time.sleep(0.01)
                continue

            # Mirror frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)

            # Monotonically increasing timestamp in ms
            curr_timestamp_ms = int(time.time() * 1000) - start_timestamp_ms

            # Process frame with MediaPipe
            hand_data = tracker.process_frame(frame, curr_timestamp_ms)

            is_left_pinched = False
            is_right_pinched = False
            if hand_data is not None:
                index_x, index_y = hand_data["index_norm"]
                is_left_pinched = hand_data["is_left_pinched"]
                is_right_pinched = hand_data["is_right_pinched"]

                if not args.no_mouse:
                    # Move cursor to index fingertip position
                    mouse.move_cursor(index_x, index_y)
                    # Trigger left click (thumb+middle) or right click (thumb+pinky)
                    mouse.update_click_states(is_left_pinched, is_right_pinched)

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # Draw HUD
            hud_frame = tracker.draw_hud(
                frame=frame,
                hand_data=hand_data,
                margin_x=args.margin_x,
                margin_y=args.margin_y,
                fps=fps,
                left_clicks=mouse.left_click_count,
                right_clicks=mouse.right_click_count,
                is_left_pinched=is_left_pinched,
                is_right_pinched=is_right_pinched
            )

            cv2.imshow("CV Virtual Mouse", hud_frame)

            # Exit key check: 'q' or Esc (27)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("[Info] Exit key pressed. Shutting down...")
                break

    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("[Info] Application closed cleanly.")

if __name__ == "__main__":
    main()

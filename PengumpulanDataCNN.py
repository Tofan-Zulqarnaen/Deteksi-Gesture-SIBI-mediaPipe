import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ==== Konfigurasi ====
BOX_X, BOX_Y, BOX_W, BOX_H = 100, 100, 300, 300
SAVE_SIZE = (64, 64)
SAVE_DIR = "gesture_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== MediaPipe ====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tracking_active = False
recording = False
start_time = None
finger_trails = []
tracked_finger = None  # Tambahan: menyimpan jari pertama yang berdiri

img_counter = len(os.listdir(SAVE_DIR))

def get_label_from_user(preview_img):
    while True:
        cv2.imshow("Preview - Type label in terminal", preview_img)
        print(">>> Ketik label untuk gambar ini lalu tekan ENTER:")
        label = input("Label: ").strip()
        if label:
            return label

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                finger_pairs = [(5, 8), (9, 12), (13, 16), (17, 20)]
                for base, tip in finger_pairs:
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                        x = int(hand_landmarks.landmark[tip].x * width)
                        y = int(hand_landmarks.landmark[tip].y * height)
                        if tracking_active:
                            if tracked_finger is None:
                                tracked_finger = tip  # Simpan jari pertama yang berdiri
                            if tip == tracked_finger:
                                finger_trails.append((x, y))
                        break  # Hentikan setelah satu jari aktif terdeteksi
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Gambar jejak jari
        for i in range(1, len(finger_trails)):
            cv2.line(image, finger_trails[i - 1], finger_trails[i], (0, 255, 0), 2)

        # Gambar bounding box
        cv2.rectangle(image, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (255, 0, 0), 2)

        if recording:
            elapsed_time = time.time() - start_time
            cv2.putText(image, f"Recording: {4 - int(elapsed_time)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed_time >= 4:
                recording = False
                tracking_active = False

                # Buat gambar jejak jari
                blank_image = np.zeros((BOX_H, BOX_W), dtype=np.uint8)
                for x, y in finger_trails:
                    if BOX_X <= x <= BOX_X + BOX_W and BOX_Y <= y <= BOX_Y + BOX_H:
                        x_norm = x - BOX_X
                        y_norm = y - BOX_Y
                        cv2.circle(blank_image, (x_norm, y_norm), 2, 255, -1)

                # Resize dan tampilkan preview
                img_resized = cv2.resize(blank_image, SAVE_SIZE)

                # Dapatkan label dari user
                label = get_label_from_user(img_resized)
                save_path = os.path.join(SAVE_DIR, f"{label}_{img_counter}.png")
                cv2.imwrite(save_path, img_resized)
                print(f"[INFO] Saved: {save_path}")
                img_counter += 1
                finger_trails.clear()
                tracked_finger = None  # Reset jari yang dilacak

        cv2.putText(image, "Press SPACE to start recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Data Collector", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            tracking_active = True
            recording = True
            start_time = time.time()
            finger_trails.clear()
            tracked_finger = None  # Reset jari yang dilacak

cap.release()
cv2.destroyAllWindows()

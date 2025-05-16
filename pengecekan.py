import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model CNN
MODEL_PATH = "gesture_model.h5"
model = load_model(MODEL_PATH)

# Label kategori dari dataset
categories = sorted(os.listdir("gesture_dataset"))

cap = cv2.VideoCapture(0)

# Bounding box untuk menggambar
BOX_X, BOX_Y, BOX_W, BOX_H = 100, 100, 300, 300

tracking_active = False
recording = False
start_time = None
finger_trails = []
tracked_finger = None  # Jari yang dilacak (tip index)

predicted_label = None
prediction_time = None

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                finger_pairs = [(5, 8), (9, 12), (13, 16), (17, 20)]  # Telunjuk hingga kelingking
                for base, tip in finger_pairs:
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                        x = int(hand_landmarks.landmark[tip].x * width)
                        y = int(hand_landmarks.landmark[tip].y * height)

                        if tracking_active:
                            if tracked_finger is None:
                                tracked_finger = tip  # Simpan jari pertama yang aktif
                            if tip == tracked_finger:
                                finger_trails.append((x, y))
                        break  # Hentikan pengecekan setelah satu jari ditemukan

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Gambar jejak jari
        for i in range(1, len(finger_trails)):
            cv2.line(image, finger_trails[i - 1], finger_trails[i], (0, 255, 0), 2)

        # Gambar bounding box
        cv2.rectangle(image, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (255, 0, 0), 2)

        if recording:
            elapsed_time = time.time() - start_time
            cv2.putText(image, f"Drawing: {4 - int(elapsed_time)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed_time >= 4:
                recording = False
                tracking_active = False

                # Konversi jejak ke gambar hitam putih
                blank_image = np.zeros((BOX_H, BOX_W), dtype=np.uint8)
                for x, y in finger_trails:
                    if BOX_X <= x <= BOX_X + BOX_W and BOX_Y <= y <= BOX_Y + BOX_H:
                        x_norm = x - BOX_X
                        y_norm = y - BOX_Y
                        cv2.circle(blank_image, (x_norm, y_norm), 2, 255, -1)

                # Resize dan normalisasi
                img_resized = cv2.resize(blank_image, (64, 64)) / 255.0
                img_resized = img_resized.reshape(1, 64, 64, 1)

                # Prediksi
                prediction = model.predict(img_resized)
                predicted_index = np.argmax(prediction)
                predicted_label = categories[predicted_index]
                confidence = prediction[0][predicted_index] * 100
                prediction_time = time.time()

                print(f"Predicted Gesture: {predicted_label} ({confidence:.2f}%)")
                finger_trails.clear()

        # Tampilkan prediksi jika masih dalam waktu 5 detik
        if predicted_label and (time.time() - prediction_time < 5):
            cv2.putText(image, f"Gesture: {predicted_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            predicted_label = None

        # Instruksi
        cv2.putText(image, "Press SPACE to start drawing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Gesture Recognition', image)

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

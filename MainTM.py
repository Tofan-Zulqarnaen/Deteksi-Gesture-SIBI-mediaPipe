import cv2
import mediapipe as mp
import os
import numpy as np
from numpy.linalg import norm

# Fungsi untuk menghitung Cosine Similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1.flatten(), vec2.flatten()) / (norm(vec1.flatten()) * norm(vec2.flatten()) + 1e-6)

# Fungsi untuk memuat template landmark dari dataset
def load_templates(template_dir):
    templates = {}
    for file_name in os.listdir(template_dir):
        if file_name.endswith('.txt'):
            label = file_name.split('_')[0]
            landmarks = np.loadtxt(os.path.join(template_dir, file_name))
            templates[label] = landmarks
    return templates

# Fungsi untuk menghitung bounding box berdasarkan landmark tangan
def calculate_bounding_box(landmarks, image_width, image_height):
    x_coordinates = [lm[0] * image_width for lm in landmarks]
    y_coordinates = [lm[1] * image_height for lm in landmarks]
    
    xmin, xmax = int(min(x_coordinates)), int(max(x_coordinates))
    ymin, ymax = int(min(y_coordinates)), int(max(y_coordinates))
    
    return xmin, ymin, xmax, ymax

# Inisialisasi MediaPipe dan OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Direktori tempat menyimpan template landmark
template_dir = 'landmark_dataset'
templates = load_templates(template_dir)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi MediaPipe hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal mendapatkan frame dari kamera")
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                current_landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    current_landmarks.append([x, y, z])
                
                xmin, ymin, xmax, ymax = calculate_bounding_box(current_landmarks, width, height)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                
                normalized_landmarks = []
                for landmark in current_landmarks:
                    x_rel = (landmark[0] * width - xmin) / (xmax - xmin)
                    y_rel = (landmark[1] * height - ymin) / (ymax - ymin)
                    z_rel = landmark[2]
                    normalized_landmarks.append([x_rel, y_rel, z_rel])

                normalized_landmarks = np.array(normalized_landmarks)
                
                best_match = None
                best_similarity = -1  # Nilai minimum Cosine Similarity
                
                for label, template_landmarks in templates.items():
                    similarity = cosine_similarity(normalized_landmarks, template_landmarks)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = label
                
                if best_match:
                    similarity_percentage = best_similarity * 100
                    cv2.putText(image, f"Gesture: {best_match} ({similarity_percentage:.2f}%)", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Pengecekan Gesture', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

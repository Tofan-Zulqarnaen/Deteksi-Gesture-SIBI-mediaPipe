import cv2
import mediapipe as mp
import os
import numpy as np

# Inisialisasi MediaPipe dan OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Membuat direktori dataset jika belum ada
dataset_dir = 'landmark_dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Fungsi untuk menghitung bounding box berdasarkan landmark tangan
def calculate_bounding_box(landmarks, image_width, image_height):
    # Dapatkan koordinat x dan y dari semua landmark
    x_coordinates = [lm.x * image_width for lm in landmarks]
    y_coordinates = [lm.y * image_height for lm in landmarks]
    
    # Cari nilai minimum dan maksimum untuk x dan y
    xmin, xmax = int(min(x_coordinates)), int(max(x_coordinates))
    ymin, ymax = int(min(y_coordinates)), int(max(y_coordinates))
    
    return xmin, ymin, xmax, ymax

# Inisialisasi MediaPipe hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal mendapatkan frame dari kamera")
            break
        
        frame = cv2.flip(frame, 1)

        # Ubah gambar ke RGB karena MediaPipe membutuhkan gambar RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteksi tangan dan landmark
        result = hands.process(image)
        
        # Ubah gambar kembali ke BGR untuk ditampilkan oleh OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ambil dimensi gambar
        height, width, _ = image.shape

        # Jika tangan terdeteksi
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Gambar landmark pada tangan yang terdeteksi
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Hitung bounding box di sekitar tangan
                xmin, ymin, xmax, ymax = calculate_bounding_box(hand_landmarks.landmark, width, height)

                # Gambar bounding box di sekitar tangan
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)  # Kotak kuning

                # Ekstraksi koordinat landmark relatif terhadap bounding box
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x_rel = (landmark.x * width - xmin) / (xmax - xmin)  # Normalisasi x ke dalam bounding box
                    y_rel = (landmark.y * height - ymin) / (ymax - ymin)  # Normalisasi y ke dalam bounding box
                    z_rel = landmark.z  # z tidak diubah karena kedalaman tidak terpengaruh oleh bounding box
                    landmarks.append([x_rel, y_rel, z_rel])

                landmarks = np.array(landmarks)  # Konversi ke numpy array untuk mudah disimpan

                # Tampilkan instruksi di layar
                cv2.putText(image, "Tekan 'spasi' untuk ambil data, Tekan 'q' untuk keluar", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Tunggu input dari keyboard
                key = cv2.waitKey(1) & 0xFF
                
                # Jika tombol spasi ditekan, ambil data landmark
                if key == ord(' '):
                    label = input("Masukkan label untuk gesture ini: ")
                    # Menyimpan koordinat landmark ke file
                    landmark_file = os.path.join(dataset_dir, f"{label}_{int(cv2.getTickCount())}.txt")
                    np.savetxt(landmark_file, landmarks, fmt='%.6f')  # Simpan sebagai file .txt
                    print(f"Data landmark dengan label '{label}' telah disimpan sebagai {landmark_file}!")
                
                # Jika tombol 'q' ditekan, keluar
                elif key == ord('q'):
                    break

        # Tampilkan hasil frame
        cv2.imshow('Pengumpulan Data Landmark Bahasa Isyarat', image)
        
        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release kamera
cap.release()
cv2.destroyAllWindows()

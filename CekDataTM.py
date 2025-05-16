import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi koneksi landmark tangan dari MediaPipe
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Fungsi untuk menggambar landmark dan garis koneksi
def draw_landmarks(landmarks, img_shape):
    # Buat gambar kosong (hitam) dengan ukuran yang ditentukan
    image = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    
    # Menggambar landmark (titik-titik tangan)
    for i, landmark in enumerate(landmarks):
        x = int(landmark[0] * img_shape[1])  # Scaling x berdasarkan lebar gambar
        y = int(landmark[1] * img_shape[0])  # Scaling y berdasarkan tinggi gambar
        # Gambar lingkaran kecil di setiap landmark
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    # Menggambar koneksi antar landmark (garis tangan)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        # Dapatkan koordinat dua titik yang terhubung
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        # Hitung posisi dalam gambar
        start_x = int(start_point[0] * img_shape[1])
        start_y = int(start_point[1] * img_shape[0])
        end_x = int(end_point[0] * img_shape[1])
        end_y = int(end_point[1] * img_shape[0])
        
        # Gambar garis antara dua titik
        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    
    return image

# Fungsi untuk membaca file .txt dan mengembalikan landmark
def load_landmark_from_file(filename):
    # Membaca file dan memuat data landmark
    landmarks = np.loadtxt(filename)
    return landmarks

# Nama file .txt dari landmark yang ingin digambar
landmark_file = 'landmark_dataset/A_872167870900.txt'  # Ubah ini ke file yang ingin kamu gambar

# Memuat landmark dari file
landmarks = load_landmark_from_file(landmark_file)

# Menentukan ukuran gambar berdasarkan rentang landmark
x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

# Tentukan ukuran gambar yang ingin digunakan
img_width = 640
img_height = int(img_width * (y_max - y_min) / (x_max - x_min))  # Menjaga rasio aspek berdasarkan landmark

# Buat ukuran gambar sesuai rentang landmark
img_shape = (img_height, img_width)

# Gambar landmark beserta garis koneksinya
image_with_landmarks = draw_landmarks(landmarks, img_shape)

# Tampilkan gambar hasil
cv2.imshow('Landmark Visualization with Connections', image_with_landmarks)

# Tunggu sampai ada tombol ditekan, lalu keluar
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- THAY ĐỔI 1: HỎI TÊN THAY VÌ ID ---
face_name = input('\n Nhập tên (viết liền không dấu, vd: Hoa) <return>: ')

# --- THAY ĐỔI 2: TẠO ĐƯỜNG DẪN VÀ THƯ MỤC MỚI ---
# Tạo đường dẫn đầy đủ, ví dụ: "dataSet/Hoa"
save_path = os.path.join("dataSet", face_name)

# Kiểm tra và tạo thư mục nếu nó chưa tồn tại
os.makedirs(save_path, exist_ok=True)
print(f"[INFO] Đã tạo/chuẩn bị thư mục: {save_path}")
# ---------------------------------------------

print("\n [INFO] Khởi tạo Camera. Nhìn thẳng vào camera...")
count = 0

while (True):

    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # --- THAY ĐỔI 3: LƯU VÀO THƯ MỤC MỚI ---
        # Tên file mới sẽ là 1.jpg, 2.jpg...
        file_name = str(count) + ".jpg"
        # Đường dẫn lưu file đầy đủ, ví dụ: "dataSet/Hoa/1.jpg"
        file_path = os.path.join(save_path, file_name)

        cv2.imwrite(file_path, gray[y:y + h, x:x + w])
        # ----------------------------------------

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # Nhấn ESC để thoát
        break
    elif count >= 30:  # Lấy đủ 30 ảnh
        break

print("\n [INFO] Đã lấy {count} ảnh. Đóng Camera ...")
cam.release()
cv2.destroyAllWindows()
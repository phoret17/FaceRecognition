import cv2
import os
from deepface import DeepFace

# 1. KHỞI TẠO CÁC BIẾN
# --------------------------------
# SỬA 1: Đảm bảo tên thư mục chính xác là "dataSet" (khớp với code kia)
db_path = "dataSet"

# Tải bộ phát hiện khuôn mặt của OpenCV
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# 2. "LÀM NÓNG" (BUILD DATABASE)
# --------------------------------
print("[INFO] Đang tải mô hình và build database (có thể mất vài giây)...")
try:
    _ = DeepFace.find(img_path="img_test.jpg",
                      db_path=db_path,
                      model_name="VGG-Face",
                      enforce_detection=False)
except ValueError:
    print("[INFO] Database đã được build. Bắt đầu nhận diện...")
except Exception as e:
    print(f"Lỗi khi build database (kiểm tra db_path): {e}")

# 3. VÒNG LẶP WEBCAM
# --------------------------------
while True:
    ret, img = cam.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(int(minH))),
    )

    for (x, y, w, h) in faces:
        pad = 20
        face_roi = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]

        label = "Unknown"
        color = (0, 0, 255)  # Màu đỏ

        # 4. PHẦN NHẬN DIỆN BẰNG DEEPFACE
        # --------------------------------
        try:
            dfs = DeepFace.find(img_path=face_roi,
                                db_path=db_path,
                                model_name="VGG-Face",
                                enforce_detection=False,
                                silent=True)

            if not dfs[0].empty:
                best_match = dfs[0].iloc[0]
                distance = best_match["distance"]

                if distance < 0.5:
                    identity_path = best_match["identity"]

                    # --- SỬA 2: LẤY TÊN THƯ MỤC CHA ---
                    # Logic cũ (lấy tên file):
                    # label = os.path.splitext(os.path.basename(identity_path))[0]

                    # Logic MỚI (lấy tên thư mục):
                    # Ví dụ: "dataSet/Hoa/1.jpg"
                    # os.path.dirname(identity_path) -> "dataSet/Hoa"
                    # os.path.basename(...) -> "Hoa"
                    label = os.path.basename(os.path.dirname(identity_path))
                    # ---------------------------------

                    color = (0, 255, 0)  # Màu xanh lá

        except Exception as e:
            pass
        # --------------------------------
        # KẾT THÚC PHẦN DEEPFACE
        # --------------------------------

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('Nhận diện khuôn mặt (DeepFace)', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# 5. DỌN DẸP
# --------------------------------
print("\n [INFO] Thoát chương trình.")
cam.release()
cv2.destroyAllWindows()
import os
import cv2
import math
import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, BooleanProperty
from blazeface_model import BlazeFaceDetector
from mobilefacenet_model import MobileFaceNetTFLite
from database import FaceDatabase
import psutil
import sys

p = psutil.Process(os.getpid())
p.cpu_affinity([0])         # Giới hạn CPU sử dụng cho tiến trình này
max_memory_bytes = 500 * 1024 * 1024  # 500 MB

def check_memory():
    mem = p.memory_info().rss
    if mem > max_memory_bytes:
        print("Đã vượt giới hạn bộ nhớ!")
        sys.exit(1)

kv_path = os.path.join(os.path.dirname(__file__), 'kv', 'main.kv')
Builder.load_file(kv_path)

class FaceRecognitionWidget(BoxLayout):
    display_name = StringProperty("")
    reward_info = StringProperty("")
    redeem_message = StringProperty("")
    show_confirm_button = BooleanProperty(False)
    show_name_input = BooleanProperty(False)
    # show_camera_small = BooleanProperty(False)
    # show_info_panel = BooleanProperty(True)
    my_dishes_text = StringProperty("")
    show_my_dishes = BooleanProperty(False)
    in_info_mode = BooleanProperty(False)   # True: panel info, False: nhận diện
    pending_name = StringProperty("")       # Tên chờ xác nhận
    pending_embedding = None                # Embedding chờ xác nhận
    pending_is_new = False                  # True nếu là người mới

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError("Không thể mở camera")
        self.detector = BlazeFaceDetector()
        self.recognizer = MobileFaceNetTFLite()
        self.database = FaceDatabase()
        self.face_frames_count = {}
        self.current_face_embedding = None
        self.current_face_id = None
        Clock.schedule_interval(self.update, 1.0/30)  # 30 FPS

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_frame)  # trả về [{'bbox':[x1,y1,x2,y2], 'landmarks':[(x,y),...]}, ...]

        if not self.in_info_mode:
            self.process_faces(rgb_frame, faces)
        self.display_frame(rgb_frame)

    def get_rotation_angle(self, landmarks):
        # landmarks: list of (x, y), theo thứ tự: right_eye, left_eye, nose, mouth, right_ear, left_ear
        left_eye = landmarks[1]
        right_eye = landmarks[0]
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def process_faces(self, rgb_frame, faces):
        face_detected = False
        for face in faces:
            bbox = face['bbox']
            landmarks = face['landmarks']
            x1, y1, x2, y2 = bbox
            face_detected = True
            face_crop = rgb_frame[y1:y2, x1:x2]

            face_id = "face"
            self.face_frames_count[face_id] = self.face_frames_count.get(face_id, 0) + 1

            if self.face_frames_count[face_id] == 5:
                angle = self.get_rotation_angle(landmarks)
                aligned_face = self.rotate_image(face_crop, -angle)  # xoay ngược góc nghiêng để căn chỉnh

                embedding = self.recognizer.get_embedding(aligned_face)
                name, score = self.database.match_face(embedding)

                # Lưu thông tin tạm thời, chuyển sang info mode
                self.pending_embedding = embedding
                if name is not None and score is not None and score > 0.5:
                    self.pending_name = name
                    self.pending_is_new = False
                else:
                    self.pending_name = ""
                    self.pending_is_new = True

                self.in_info_mode = True
                self.update_info_panel()
                self.face_frames_count.clear()  # reset đếm frame khi chuyển trạng thái
                break

            # Vẽ bounding box
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not face_detected:
            self.face_frames_count.clear()
            self.reset_ui()

    def display_frame(self, rgb_frame):
        flipped = np.flip(rgb_frame, 0)
        buf = flipped.tobytes()
        texture = Texture.create(size=(rgb_frame.shape[1], rgb_frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.ids.camera_view.texture = texture

    def update_ui_recognized(self, name):
        self.display_name = name
        self.show_confirm_button = True
        self.show_name_input = False
        self.show_camera_small = True
        self.show_info_panel = True
        self.reward_info = f"Điểm thưởng: {self.database.data.get(name, {}).get('reward_points', 0)}"

    def update_ui_unrecognized(self):
        self.display_name = ""
        self.show_confirm_button = True
        self.show_name_input = True
        self.show_camera_small = True
        self.show_info_panel = True
        self.reward_info = ""

    def update_info_panel(self):
        self.display_name = self.pending_name if self.pending_name else ""
        self.show_confirm_button = True
        self.show_name_input = self.pending_is_new
        self.show_camera_small = True
        self.show_info_panel = True
        # Hiển thị các nút chức năng chỉ khi ở info mode
        if self.in_info_mode:
            self.show_my_dishes = True
        else:
            self.show_my_dishes = False
        if not self.pending_is_new and self.pending_name:
            self.reward_info = f"Điểm thưởng: {self.database.data.get(self.pending_name, {}).get('reward_points', 0)}"
        else:
            self.reward_info = ""

    def back_to_recognition(self):
        self.in_info_mode = False
        self.show_my_dishes = False
        self.reset_ui()


    def reset_ui(self):
        self.face_frames_count.clear()
        self.display_name = ""
        self.show_confirm_button = False
        self.show_name_input = False
        self.show_camera_small = False
        self.show_info_panel = False
        self.reward_info = ""
        self.pending_name = ""
        self.pending_embedding = None
        self.pending_is_new = False
        self.in_info_mode = False
        self.show_my_dishes = False
        # Ẩn các nút "quay lại", "món của tôi" khi reset về nhận diện
        if hasattr(self.ids, "back_button"):
            self.ids.back_button.opacity = 0
            self.ids.back_button.disabled = True

    
    def on_my_dishes(self):
        # Lấy danh sách món từ database
        if self.current_face_id and self.current_face_id in self.database.data:
            dishes = self.database.data[self.current_face_id].get("dishes", [])
            if dishes:
                self.my_dishes_text = "Món của bạn:\n" + "\n".join(f"- {d}" for d in dishes)
            else:
                self.my_dishes_text = "Bạn chưa có món nào."
            self.show_my_dishes = True
        else:
            self.my_dishes_text = "Không tìm thấy thông tin người dùng."
            self.show_my_dishes = True

    def on_add_new_dish(self):
        # Ở đây bạn có thể mở popup hoặc thêm logic nhập món mới
        print("[INFO] Bấm nút 'Thêm món mới'")
        # Ví dụ đơn giản: thêm món mẫu
        if self.current_face_id and self.current_face_id in self.database.data:
            dishes = self.database.data[self.current_face_id].setdefault("dishes", [])
            new_dish = f"Món mới {len(dishes)+1}"
            dishes.append(new_dish)
            self.database.save()
            self.my_dishes_text = "Đã thêm món mới!\n" + "\n".join(f"- {d}" for d in dishes)
            self.show_my_dishes = True
        else:
            self.my_dishes_text = "Không tìm thấy thông tin người dùng."
            self.show_my_dishes = True

    def on_redeem_reward(self):
        # Đảm bảo rằng có một người dùng đã được nhận diện
        if self.current_face_id and self.current_face_id in self.database.data:
            current_rewards = self.database.data[self.current_face_id].get('reward_points', 0)
            redeem_cost = 10 # Giả sử đổi thưởng mất 10 điểm

            if current_rewards >= redeem_cost:
                # Trừ điểm thưởng
                self.database.data[self.current_face_id]['reward_points'] = current_rewards - redeem_cost
                self.database.save() # Lưu thay đổi vào database

                # Cập nhật hiển thị điểm thưởng ngay lập tức
                self.reward_info = f"Điểm thưởng: {self.database.data[self.current_face_id]['reward_points']}"
                self.redeem_message = f"Bạn đã đổi thưởng thành công! Còn lại {self.database.data[self.current_face_id]['reward_points']} điểm."
                print(f"[INFO] {self.current_face_id} đã đổi thưởng, còn {self.database.data[self.current_face_id]['reward_points']} điểm.")
            else:
                # Thông báo không đủ điểm
                self.redeem_message = f"Bạn không đủ điểm để đổi thưởng. Cần {redeem_cost} điểm."
                print(f"[INFO] {self.current_face_id} không đủ điểm để đổi thưởng.")
        else:
            self.redeem_message = "Vui lòng nhận diện khuôn mặt trước khi đổi thưởng."
            print("[INFO] Không có người dùng nào được nhận diện để đổi thưởng.")

    def on_confirm(self):
        try:
            if not self.pending_is_new and self.pending_name:
                # Người đã có, cập nhật reward
                self.database.update_reward_if_eligible(self.pending_name)
                self.display_name = self.pending_name
                self.reward_info = f"Điểm thưởng: {self.database.data.get(self.pending_name, {}).get('reward_points', 0)}"
                self.current_face_id = self.pending_name
                self.face_frames_count.clear()  # reset đếm frame khi xác nhận
            else:
                # Người mới, lấy tên nhập vào
                name = ""
                if hasattr(self.ids, "name_input"):
                    name = self.ids.name_input.text.strip()
                if not name:
                    self.display_name = ""
                    self.reward_info = ""
                    self.current_face_id = None
                    return
                if name in self.database.data:
                    self.display_name = "Tên đã tồn tại, hãy nhập tên khác."
                    self.reward_info = ""
                    self.current_face_id = None
                    return
                if self.pending_embedding is not None:
                    self.database.add_face(name, self.pending_embedding)
                    self.display_name = name
                    self.current_face_id = name
                    self.reward_info = f"Điểm thưởng: 1"
                    self.face_frames_count.clear()
                else:
                    self.display_name = ""
                    self.reward_info = ""
                    self.current_face_id = None
        except Exception as e:
            print(f"[ERROR] Lỗi trong on_confirm: {e}")

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()

class FaceRecognitionApp(App):
    def build(self):
        self.root_widget = FaceRecognitionWidget()
        return self.root_widget

    def on_stop(self):
        self.root_widget.on_stop()

if __name__ == '__main__':
    FaceRecognitionApp().run()
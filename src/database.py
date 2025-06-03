import os
import pickle
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FaceDatabase:
    def __init__(self, db_path="face_db.pkl"):
        self.db_path = db_path
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = {}  # {name: {"embedding":..., "reward_points":..., "last_access":...}}

    def add_face(self, name, embedding):
        now = time.time()
        # New entry
        self.data[name] = {
            "embedding": embedding,
            "reward_points": 1,
            "last_access": now
        }
        self.save()
        print(f"[INFO] Đã thêm khuôn mặt mới: {name}")

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)

    def match_face(self, embedding, threshold=0.5):
        if not self.data:
            return None, None
        names = list(self.data.keys())
        embeddings = np.array(np.array([self.data[name]["embedding"] for name in names]))
        sims = cosine_similarity([embedding], embeddings)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] > threshold:
            # Update last access time and increment reward points on recognition
            name = names[best_idx]
            return name, sims[best_idx]
        else:
            return None, None
    def update_reward_if_eligible(self, name):
        now = time.time()
        five_minutes = 300

        if name not in self.data:
            print(f"Không tìm thấy {name} trong database để cập nhật reward.")
            return

        last_access = self.data[name].get("last_access", 0)
        if now - last_access > five_minutes:
            self.data[name]["reward_points"] = self.data[name].get("reward_points", 0) + 1
            self.data[name]["last_access"] = now
            self.save()
            print(f"Đã tăng điểm thưởng cho {name}. Điểm hiện tại: {self.data[name]['reward_points']}")
        else:
            remaining_seconds = five_minutes - (now - last_access)
            remaining_minutes = int(remaining_seconds // 60)  # Lấy phần nguyên số phút còn lại
            print(f"'{name}' chưa đủ thời gian để nhận điểm thưởng. Thời gian còn lại: {remaining_minutes} phút.")

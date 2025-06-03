import pickle
import numpy as np
import datetime

def print_face_db_embeddings(db_path="face_db.pkl"):
    try:
        with open(db_path, "rb") as f:
            face_db = pickle.load(f)
    except FileNotFoundError:
        print(f"File {db_path} không tồn tại.")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    if not face_db:
        print("Database rỗng.")
        return

    print(f"Database có {len(face_db)} người:")
    for name, info in face_db.items():
        embedding = info.get("embedding", None)
        reward_points = info.get("reward_points", "Chưa có")
        last_access_ts = info.get("last_access", None)
        if last_access_ts:
            last_access = datetime.datetime.fromtimestamp(last_access_ts).strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_access = "Chưa có"
        
        if embedding is not None:
            # Nếu embedding là numpy array hoặc list, in 10 giá trị đầu
            emb_preview = embedding[:10]
        else:
            emb_preview = "Không có embedding"

        print(f"- {name}:")
        print(f"    embedding (10 giá trị đầu): {emb_preview}")
        print(f"    reward_points: {reward_points}")
        print(f"    last_access: {last_access}")

if __name__ == "__main__":
    print_face_db_embeddings()

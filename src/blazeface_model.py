import mediapipe as mp
class BlazeFaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence)

    def detect_faces(self, image):
        # MediaPipe yêu cầu ảnh RGB
        results = self.face_detection.process(image)
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                x2 = x1 + width
                y2 = y1 + height
                # Giới hạn trong kích thước ảnh
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Lấy landmark (6 điểm) và chuyển sang tọa độ tuyệt đối
                keypoints = detection.location_data.relative_keypoints
                landmarks = []
                for kp in keypoints:
                    landmarks.append((int(kp.x * w), int(kp.y * h)))

                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "landmarks": landmarks
                })
        return faces

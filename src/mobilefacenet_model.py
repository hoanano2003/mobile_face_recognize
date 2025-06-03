import tensorflow as tf
import numpy as np
from PIL import Image

class MobileFaceNetTFLite:
    def __init__(self, model_path="mobilefacenet_pretrain.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, face_img):
        img = Image.fromarray(face_img).resize((112, 112)).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
        return img_array

    def get_embedding(self, face_img):
        input_data = self.preprocess(face_img)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data.squeeze()  # Trả về vector embedding dạng numpy array

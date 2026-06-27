import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
        self.model = load_model("model/model.keras")
        
        
    def predict(self):
        image_name = self.filename
        test_image = image.load_img(image_name, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        predictions =self.model.predict(test_image)
        result = np.argmax(predictions, axis=1)[0]
        print(f"Predicted class index: {result}")

        classes = ["Cyst", "Normal", "Stone", "Tumor"]
        
        predicted_label = classes[result]

        return [{"image": predicted_label}]

       
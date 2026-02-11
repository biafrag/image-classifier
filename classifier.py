from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

model = MobileNetV2(weights='imagenet')

def classify_image(image):

    img_resized = image.resize((224, 224))

    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_ready = preprocess_input(img_batch)

    preds = model.predict(img_ready)
    scores = preds[0]
    winner_index = np.argmax(scores)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    confidence = float(decoded[2])
    return label, confidence


from io import BytesIO
from urllib import request

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Load model and get index of output class
interpreter = tflite.Interpreter(model_path="bees-wasps-v2.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def predict(url):
    # Load image and preprocess it
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    # Convert to numpy array
    img = np.asarray(img, dtype="float32")
    x = np.array(img)
    X = np.array([x]) / 255

    # Make prediction
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds[0].tolist()


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result

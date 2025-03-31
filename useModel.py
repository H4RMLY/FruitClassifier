import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from PIL import Image

img_width = 180
img_height = 180
class_names = ['rotten', 'fresh']

modelPath = "./rottenApplePredictionModel.tflite"
interpreter = tf.lite.Interpreter(model_path=modelPath)

apple_path = pathlib.Path("C:\\Users\\citiz\\Documents\\UNI\\IoT\\Original Image\\Fruits Original\\Apple\\Fresh")
allImgs = list(apple_path.glob('./*'))
print(len(allImgs))

for i in allImgs:
    img = tf.keras.utils.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
   
    classify_lite = interpreter.get_signature_runner('serving_default')
    classify_lite

    predictions_lite = classify_lite(rescaling_1_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)
    
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)





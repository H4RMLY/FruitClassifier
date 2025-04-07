import numpy as np
import tensorflow as tf
import pathlib
import keras

img_width = 180
img_height = 180
class_names = ['rotten', 'fresh']

modelPath = "./rottenPredictionModel.tflite"
interpreter = tf.lite.Interpreter(model_path=modelPath)

imgs_path = pathlib.Path("C:\\Users\\citiz\\Documents\\UNI\\IoT\\FruitClassifier-main\\TestModel")
allImgs = list(imgs_path.glob('All/*'))
print(len(allImgs))

print(interpreter.get_signature_list())

for i in allImgs:
    img = keras.utils.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
   
    classify_lite = interpreter.get_signature_runner('serving_default')
    classify_lite

    predictions_lite = classify_lite(keras_tensor=img_array)['output_0']
    score_lite = tf.nn.softmax(predictions_lite)
    
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)





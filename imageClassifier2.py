import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os

from tensorflow import keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers
import keras
from PIL import Image

# Point to image directory
data_dir = pathlib.Path("d:\FreshRottenAllFruit")

# Define batch size and image dimensions
batch_size = 10
img_height = 180
img_width = 180

# Create training set from all images
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # use 20% of images for testing 
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Create testing set from images
test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # use 20% of images for testing
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Identify class names from folders within directory
class_names = train_ds.class_names
print(class_names)

# Display a batch of images from the training set
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

# Cache images for better performance
# train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize images to [0,1]
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = len(class_names)

# Define model architecture (This model is not hight accuracy)
model = keras.models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
epochs=10 # Set amount of times to pass through training set
history = model.fit(
  train_ds,
  steps_per_epoch=int(len(train_ds)/batch_size),
  validation_data=test_ds,
  validation_steps=int(len(test_ds)/batch_size),
  epochs=epochs
)

# Display results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Convert the model to tfLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('rottenPredictionModel.tflite', 'wb') as f:
  f.write(tflite_model)
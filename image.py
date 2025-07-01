pip install tensorflow tensorflow_hub pillow matplotlib
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from google.colab import files  

uploaded = files.upload()
image_path = list(uploaded.keys())[0] 

module_handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(module_handle)

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.astype(np.float32)
    img_tensor = tf.expand_dims(img_array, axis=0)

    result = model(img_tensor)
    predicted_class = np.argmax(result, axis=-1)[0]

    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()

    predicted_label = labels[predicted_class]
    return predicted_label

predicted_label = predict_image(image_path)
print("Predicted Label:", predicted_label)

img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.title(predicted_label)
plt.show()

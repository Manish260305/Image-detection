# 🧠 Image Classifier with TensorFlow Hub (Google Colab Version)

This project uses a pre-trained MobileNetV2 model from [TensorFlow Hub](https://tfhub.dev) to classify images uploaded by the user in a **Google Colab** environment.

---

## 🚀 How It Works

1. Upload an image via the `files.upload()` function
2. The image is resized and normalized
3. TensorFlow Hub's MobileNetV2 model makes a prediction
4. The predicted label is displayed with the image

---

## 🖼️ Example

| Upload Image        | Predicted Label  |
|---------------------|------------------|
| `dog.jpg`           | Labrador retriever |
| `apple.png`         | Granny Smith apple |

---

## 🧪 Dependencies

Install these libraries (if you're running outside of Colab):

```bash
pip install tensorflow tensorflow_hub pillow matplotlib

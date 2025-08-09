import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. Load the Trained Model ---
# Load the model you saved from your notebook.
model = tf.keras.models.load_model('cifar10_model.keras')
print("Model loaded successfully.")

# --- 2. Define Class Names ---
# The class names must be in the same order as the model's output classes.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- 3. Create the Prediction Function ---
# This function will take a user-uploaded image and return the model's predictions.
def predict_image(input_image: Image.Image) -> dict:
    """
    Takes a PIL Image, preprocesses it, and returns a dictionary of class probabilities.
    """
    # Preprocess the image to match the model's input requirements.
    # The model was trained on 32x32 images.
    img = input_image.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    
    # Create a dictionary of class names and their probabilities
    confidences = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    
    return confidences

# --- 4. Create the Gradio Interface ---
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
    title="CIFAR-10 Image Classifier",
    description="Upload an image of an airplane, car, bird, etc., and the model will predict its class.",
    examples=[
        # You can add some example images here if you have them.
        # For now, we'll leave it empty.
    ]
)

# --- 5. Launch the App ---
if __name__ == "__main__":
    iface.launch()
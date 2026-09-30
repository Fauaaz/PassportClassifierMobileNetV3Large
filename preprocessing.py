from PIL import Image
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession(
    "fine_tuned_model_int8.onnx",
    providers=["CPUExecutionProvider"]
)

# Load user image
image = Image.open("new_pic.jpg").convert("RGB")

# Preprocess
image = image.resize((224, 224))

image = np.array(image).astype(np.float32) / 255.0

# Normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

image = (image - mean) / std

# HWC → CHW
image = np.transpose(image, (2, 0, 1))

# Add batch dimension
image = np.expand_dims(image, axis=0).astype(np.float32)

# Run model
input_name = session.get_inputs()[0].name

output = session.run(
    None,
    {input_name: image}
)[0]

# Get predicted class
prediction = np.argmax(output, axis=1)[0]

if prediction == 0:
    print('Image Approved')
else:
    print('Image Rejected')

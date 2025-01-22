import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import zipfile
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Unzip example images
with zipfile.ZipFile("examples.zip","r") as zip_ref:
    zip_ref.extractall(".")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN for face detection and InceptionResnetV1 for feature extraction
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

EXAMPLES_FOLDER = 'examples'
examples_names = os.listdir(EXAMPLES_FOLDER)
examples = []
for example_name in examples_names:
    example_path = os.path.join(EXAMPLES_FOLDER, example_name)

    label = example_name.split('_')[0]
    example = {
        'path': example_path,
        'label': label
    }
    examples.append(example)
np.random.shuffle(examples)  # Shuffle examples

def predict(input_image: Image.Image):
    """Predict the label of the input_image and generate face explainability with Grad-CAM."""
    # Detect face
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    
    face = face.unsqueeze(0)  # Add batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0

    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    # Apply Grad-CAM for explainability
    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"

        real_prediction = 1 - output.item()
        fake_prediction = output.item()

        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    
    return confidences, face_with_mask

# Create the interface for the deepfake detection app
interface = gr.Interface(
    fn=predict,  # The function to run for predictions
    inputs=[gr.Image(label="📸 Input Image", type="pil")],  # Image input component with icon
    outputs=[
        gr.Label(label="🔍 Class"),  # Output: Prediction label (real/fake) with icon
        gr.Image(label="🧑‍⚖️ Face with Explainability", type="numpy")  # Output: Face with Grad-CAM explanation
    ],
    examples=[[examples[i]["path"]] for i in range(10)],  # Examples for user to try (horizontal layout)
    cache_examples=True,  # Cache examples for faster load time
    title="🌟 Deepfake Image Detection 🌟",  # Title with icon
    description=(
        "<b style='font-size:18px;'>This system uses a deep learning model (InceptionResNetV1) pre-trained on VGGFace2 "
        "<b style='font-size:18px;'>to classify images as 'real' or 'fake'. The model's predictions are explained using "
        "<b style='font-size:18px;'>Grad-CAM, which highlights the important regions in the image that contributed to the decision."
    ),  # Simplified description
    theme="huggingface",  # Use Hugging Face theme
    article=(
        "## 🔍 Overview\n"
        "<b style='font-size:18px;'>This tool uses AI to classify images as real or fake and also explains the decision. It uses the **InceptionResNetV1** model for classification and **Grad-CAM** for visualizing which parts of the image helped make the decision.\n\n"
        "**<b style='font-size:18px;'>Developed by:**\n"
        "<b style='font-size:25px;'>Aditya Raj, Sanjana Nayak, Shreeya Pandey, Shivam Garg</b>\n\n"  # Increased font size for the credits
        "<b style='font-size:18px;'>This system not only detects deepfakes but also helps you understand why the AI made that decision."
    )  # Simplified and direct article
).launch(share=True)  # Launch the interface with a shareable link

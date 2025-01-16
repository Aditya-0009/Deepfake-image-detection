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
with zipfile.ZipFile("examples.zip", "r") as zip_ref:
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
    example = {
        'path': example_path
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

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Image(label="Input Image", type="pil"),
    ],
    outputs=[
        gr.components.Label(label="Class"),
        gr.components.Image(label="Face with Explainability", type="numpy")
    ],
    examples=[[examples[i]["path"]] for i in range(10)],  # Horizontal layout
    cache_examples=True,
    title="Deepfake Image Detection",
    description="This system uses a deep learning model (InceptionResNetV1) pre-trained on VGGFace2 to classify images as 'real' or 'fake'. The model's predictions are explained using Grad-CAM for visual interpretability, highlighting important regions in the image that contributed to the decision.",
    theme="compact",
    article="Developed by Aditya Raj, Sanjana Nayak, Rajvardhan Singh Sirohi for the 'Deepfake Image Detection'. \nThis tool demonstrates AI's power in real-time image classification and explainability using Grad-CAM and facial recognition."
).launch()

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Model import DualInputSketchFaceGenerator
import torchvision.transforms.functional as TF
import gradio as gr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualInputSketchFaceGenerator().to(device)
model.load_state_dict(torch.load(r"C:\Users\sanjana\OneDrive\Desktop\Computer Vision\sketch2face_generator90.pth", map_location=device))
# model.eval()

def create_edge_map(pil_img):
    gray = np.array(pil_img.convert("RGB"))
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor(),
transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def infer_gradio(pil_sketch):
    sketch = pil_sketch.resize((256, 256))
    edge = create_edge_map(sketch)
    original_size = pil_sketch.size
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    edge_tensor = transform(edge).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sketch_tensor, edge_tensor)
        output = (output.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu()

    result_image = TF.to_pil_image(output)
    result_image = result_image.resize(original_size)  
    edge = edge.resize(original_size)                 
    sketch = sketch.resize(original_size)
    return sketch, edge, result_image


    
demo = gr.Interface(
    fn=infer_gradio,
    inputs=gr.Image(type="pil", label="Upload Sketch"),
    outputs=[
        gr.Image(type="pil", label="Input Sketch"),
        gr.Image(type="pil", label="Edge Map"),
        gr.Image(type="pil", label="Generated Face")
    ],
    title="Sketch-to-Face Generator",
    description="Upload a sketch. This app generates an edge map and then synthesizes a realistic face using a GAN model.",
    allow_flagging="never"
)    

if __name__ == "__main__":
    demo.launch()

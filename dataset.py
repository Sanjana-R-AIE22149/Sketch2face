import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader


class SketchFaceFusion(Dataset):
    def __init__(self, sketch_dir, face_dir):
        self.sketch_dir = sketch_dir
        self.face_dir = face_dir
        self.image_names = [
            fname[:-4] for fname in os.listdir(face_dir) if fname.endswith(".jpg")
        ]
        
        self.base_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)
        ])
        
        self.to_tensor = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        name = self.image_names[idx]
        sketch_path = os.path.join(self.sketch_dir, name + "-sz1.jpg")
        face_path = os.path.join(self.face_dir, name + ".jpg")

        sketch = Image.open(sketch_path).convert("RGB")
        face = Image.open(face_path).convert("RGB")

        # Convert sketch to edge map
        edge = self.get_edge_map(sketch)

        sketch_tensor = self.base_transform(sketch)   # [3, 256, 256]
        edge_tensor = self.to_tensor(edge)            # [1, 256, 256], grayscale
        edge_tensor = edge_tensor.repeat(3, 1, 1)     # Convert to 3 channels

        input_tensor = torch.cat([sketch_tensor, edge_tensor], dim=0)  # [6, 256, 256]
        face_tensor = self.base_transform(face)                         # [3, 256, 256]

        return input_tensor, face_tensor

    def get_edge_map(self, pil_img):
        np_img = np.array(pil_img.resize((256, 256)).convert("L"))  # grayscale
        edges = cv2.Canny(np_img, threshold1=100, threshold2=200)
        edges_pil = Image.fromarray(edges)
        return edges_pil


dataset = SketchFaceFusion("Data/resized_sketches", "Data/resized_faces")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

sketch_edge, face = next(iter(dataloader))
print(sketch_edge.shape)  
print(face.shape)         

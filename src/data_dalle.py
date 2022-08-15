import os
import cv2
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from transformers import DetrFeatureExtractor
import torchvision.transforms as transforms

class Dalle(torch.utils.data.Dataset):
    """Kandinsky Patterns dataset.
    """

    def __init__(self, dataset, split):
        self.image_paths = load_images_and_labels()
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")

    def __getitem__(self, item):
        path = self.image_paths[item]
        print(f"getting {path}")
        image = Image.open(path)
        image = image.convert('RGB')

        #img = T.Resize((256,256))( img )
        #img = ImageEnhance.Contrast(img).enhance(.7)
        #img = ImageEnhance.Brightness(img).enhance(1.3)

        pixel_values = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'] #
        return path, pixel_values[0], transforms.PILToTensor()(image)

    def __len__(self):
        return len(self.image_paths)

def load_images_and_labels():
    """Load image paths and labels for kandinsky dataset.
    """
    from pathlib import Path
    pathlist = Path("/home/bjoern.aa/Dalle_Logic/dalle-mini/results/task3").glob('*')
    all_paths = []
    for path in pathlist:
        all_paths.append(str(path))


    return all_paths

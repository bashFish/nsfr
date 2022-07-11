
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch


import torchvision
import os
import random

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "custom_objects.json") #if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        #random.uniform(1.5, 1.9)

        img = ImageEnhance.Brightness(img).enhance(random.uniform(.9, 2.4))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(.7, 1.2))

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


class Detr(pl.LightningModule):

     def __init__(self, lr, lr_backbone, weight_decay, id2label):
         super().__init__()
         # replace COCO classification head with custom head

         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)

         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
         print(pixel_mask)
         outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

         return outputs

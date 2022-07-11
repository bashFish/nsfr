import sys

import torch
import torch.nn as nn

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

from slot_attention.model import SlotAttention_model
import sys
sys.path.insert(0, 'src/yolov5')


import torchvision.transforms as T
from PIL import  ImageEnhance


from PIL import Image, ImageDraw
import random
import os
from detr.detr import *
import torchvision.transforms as transforms
import numpy as np



class DETRMixPerceptionModule(nn.Module):
    """A perception module using YOLO.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    def __init__(self, e, d, device, train=False):
        super().__init__()
        self.e = e  # num of entities
        self.d = d  # num of dimension
        self.threshold = .6
        self.id2label = {0: 'red sphere', 1: 'blue sphere', 2: 'red cube', 3: 'blue cube', 4: 'red cylinder', 5: 'blue cylinder'}
        self.id2shape = {0:0,1:0,2:1,3:1,4:2,5:2}
        self.id2color = {0:0,1:1,2:0,3:1,4:0,5:1}
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        self.model = self.load_model() #'/home/bjoern.aa/Dalle_Logic/nsfr-old/src/detr/detr_tuned_101_new.ckpt', device=device)

        # function to transform e * d shape, YOLO returns class labels,
        # it should be decomposed into attributes and the probabilities.
        self.preprocess = DETRMixPreprocess(device)

    def load_model(self):
        print("Loading DETR model...")
        model =  torch.load("/home/bjoern.aa/Dalle_Logic/nsfr-old/src/detr/tuned_101_new_both_2.ckpt") #Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label=self.id2label) #torch.load(path) #, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label=self.id2label)
        #model.to(device)

        if not self.train_:
            for param in model.parameters():
                param.requires_grad = False
        return model

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_h, img_w = size[2:]
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def forward(self, input):

        path, pixel_values, image = input
        #pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values=pixel_values, pixel_mask=None)
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.threshold

        kept_probas = probas[keep]
        kept_bboxes_scaled = self.rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.shape)

        pil_image = transforms.ToPILImage()(image[0])
        np_image = np.array(pil_image)

        kept_objects = []
        for p, (xmin, ymin, xmax, ymax) in zip(kept_probas, kept_bboxes_scaled.tolist()):

            patch = np_image[int(ymin):int(ymax),int(xmin):int(xmax),:]
            average_color = np.average(patch, axis = (0,1))

            #TODO: brown and gray
            if average_color[2] > average_color[0]:
                color = 'blue'
                colorid= 0
            else:
                color = 'redbrown'
                colorid= 1

            #size = (xmax-xmin) * (ymax-ymin)
            cl = p.argmax()
            #text = f'{self.id2label[cl.item()]}: {p[cl.item()]:0.2f} {color}'
            y1,y2,x1,x2 = int(ymin),int(ymax),int(xmin),int(xmax)

            kept_objects.append([(xmin/pil_image.size[0], ymin/pil_image.size[1], xmax/pil_image.size[0], ymax/pil_image.size[1]),
                self.id2shape[cl.item()], p[cl.item()], self.id2color[cl.item()], (x1+(x2-x1)/2, y1+(y2-y1)/2), (x2-x1)*(y2-y1) ])

        #TODO: nonmax suprr could be interesting
        #yolo_output = self.pad_result(
        #    non_max_suppression(pred, max_det=self.e))
        return self.preprocess(kept_objects)


class DETRMixPreprocess(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        device (device): The device where the model to be loaded.
        img_size (int): The size of the (resized) image to normalize the xy-coordinates.
        classes (list(str)): The classes of objects.
        colors (tensor(int)): The one-hot encodings of the colors (repeated 3 times).
        shapes (tensor(int)): The one-hot encodings of the shapes (repeated 3 times).
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.classes = None
        #['red square', 'red circle', 'red triangle',  'yellow square', 'yellow circle',  'yellow triangle',  'blue square', 'blue circle', 'blue triangle']

        self.colors = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
        ])
        self.shapes = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
        ])

    def forward(self, kept_objects):
        """A preprocess funciton for the YOLO model. The format is: [x1, y1, x2, y2, prob, class].

        Args:
            x (tensor): The output of the YOLO model. The format is:

        Returns:
            Z (tensor): The preprocessed object-centric representation Z. The format is: [x1, y1, x2, y2, color1, color2, color3, shape1, shape2, shape3, objectness].
            x1,x2,y1,y2 are normalized to [0-1].
            The probability for each attribute is obtained by copying the probability of the classification of the YOLO model.
        """
        # kept_objects.append([(xmin/pil_image.size[0], ymin/pil_image.size[1], xmax/pil_image.size[0], ymax/pil_image.size[1]), cl.item(), p[cl.item()], colorid])

        #non_max_suppression(pred, max_det=self.e)

        object_list = []
        for obj in (kept_objects):
            color = self.colors[obj[-3]] * obj[-4]
            shape = self.shapes[obj[-5]] * obj[-4]
            obj = torch.cat([torch.tensor(obj[0]), torch.tensor(color), torch.tensor(shape), torch.tensor([obj[-4]]), torch.tensor(obj[-2]), torch.tensor([obj[-1]])], dim=-1)
            object_list.append(obj)
        if len(object_list) < 5:
            for _ in range(5-len(object_list)):
                object_list.append(torch.tensor([0]*14))

        return torch.stack(object_list, dim=1).to(self.device)



class YOLOPerceptionModule(nn.Module):
    """A perception module using YOLO.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    def __init__(self, e, d, device, train=False):
        super().__init__()
        self.e = e  # num of entities
        self.d = d  # num of dimension
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        self.model = self.load_model(
            path='src/weights/yolov5/best.pt', device=device)
        # function to transform e * d shape, YOLO returns class labels,
        # it should be decomposed into attributes and the probabilities.
        self.preprocess = YOLOPreprocess(device)

    def load_model(self, path, device):
        print("Loading YOLO model...")
        yolo_net = attempt_load(weights=path)
        yolo_net.to(device)
        if not self.train_:
            for param in yolo_net.parameters():
                param.requires_grad = False
        return yolo_net

    def forward(self, imgs):
        pred = self.model(imgs)[0]  # yolo model returns tuple
        # yolov5.utils.general.non_max_supression returns List[tensors]
        # with lengh of batch size
        # the number of objects can vary image to iamge
        yolo_output = self.pad_result(
            non_max_suppression(pred, max_det=self.e))
        return self.preprocess(yolo_output)

    def pad_result(self, output):
        """Padding the result by zeros.
            (batch, n_obj, 6) -> (batch, n_max_obj, 6)
        """
        padded_list = []
        for objs in output:
            if objs.size(0) < self.e:
                diff = self.e - objs.size(0)
                zero_tensor = torch.zeros((diff, 6)).to(self.device)
                padded = torch.cat([objs, zero_tensor], dim=0)
                padded_list.append(padded)
            else:
                padded_list.append(objs)
        return torch.stack(padded_list)


class SlotAttentionPerceptionModule(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
        model: The slot attention model.
    """

    def __init__(self, e, d, device, train=False):
        super().__init__()
        self.e = e  # num of entities -> n_slots=10
        self.d = d  # num of dimension -> encoder_hidden_channels=64
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        self.model = self.load_model()

    def load_model(self):
        """Load slot attention network.
        """
        if self.device == torch.device('cpu'):
            sa_net = SlotAttention_model(n_slots=10, n_iters=3, n_attr=18,
                                         encoder_hidden_channels=64,
                                         attention_hidden_channels=128, device=self.device)
            log = torch.load(
                "src/weights/slot_attention/best.pt", map_location=torch.device(self.device))
            sa_net.load_state_dict(log['weights'], strict=True)
            sa_net.to(self.device)
            if not self.train_:
                for param in sa_net.parameters():
                    param.requires_grad = False
            return sa_net
        else:
            sa_net = SlotAttention_model(n_slots=10, n_iters=3, n_attr=18,
                                         encoder_hidden_channels=64,
                                         attention_hidden_channels=128, device=self.device)
            log = torch.load("src/weights/slot_attention/best.pt")
            sa_net.load_state_dict(log['weights'], strict=True)
            sa_net.to(self.device)
            if not self.train_:
                for param in sa_net.parameters():
                    param.requires_grad = False
            return sa_net

    def forward(self, imgs):
        return self.model(imgs)


class YOLOPreprocess(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        device (device): The device where the model to be loaded.
        img_size (int): The size of the (resized) image to normalize the xy-coordinates.
        classes (list(str)): The classes of objects.
        colors (tensor(int)): The one-hot encodings of the colors (repeated 3 times).
        shapes (tensor(int)): The one-hot encodings of the shapes (repeated 3 times).
    """

    def __init__(self, device, img_size=128):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.classes = ['red square', 'red circle', 'red triangle',
                        'yellow square', 'yellow circle',  'yellow triangle',
                        'blue square', 'blue circle', 'blue triangle']
        self.colors = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([0, 0, 1]).to(device)
        ])
        self.shapes = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device)
        ])

    def forward(self, x):
        """A preprocess funciton for the YOLO model. The format is: [x1, y1, x2, y2, prob, class].

        Args:
            x (tensor): The output of the YOLO model. The format is:

        Returns:
            Z (tensor): The preprocessed object-centric representation Z. The format is: [x1, y1, x2, y2, color1, color2, color3, shape1, shape2, shape3, objectness].
            x1,x2,y1,y2 are normalized to [0-1].
            The probability for each attribute is obtained by copying the probability of the classification of the YOLO model.
        """
        batch_size = x.size(0)
        obj_num = x.size(1)
        object_list = []
        for i in range(obj_num):
            zi = x[:, i]
            class_id = zi[:, -1].to(torch.int64)
            color = self.colors[class_id] * zi[:, -2].unsqueeze(-1)
            shape = self.shapes[class_id] * zi[:, -2].unsqueeze(-1)
            xyxy = zi[:, 0:4] / self.img_size
            prob = zi[:, -2].unsqueeze(-1)
            obj = torch.cat([xyxy, color, shape, prob], dim=-1)
            object_list.append(obj)
        return torch.stack(object_list, dim=1).to(self.device)

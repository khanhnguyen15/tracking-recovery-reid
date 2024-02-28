'''
Extractor to extract features from an image
Adapted from: https://github.com/KaiyangZhou/deep-person-reid
Author: Kaiyang Zhou
'''


import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from .models import init_model, load_pretrained_weight

class FeatureExtractor(object):

    def __init__(
        self,
        model_path,
        image_size = (256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        device='cuda',
        verbose=True
    ):

        model = init_model(num_classed=1)
        model.eval()

        if os.path.isfile(model_path):
            load_pretrained_weight(model, model_path, device)

        transforms = []
        transforms.append(T.Resize(image_size))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=pixel_mean, std=pixel_std))

        preprocess = T.Compose(transforms)
        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        self.model = model
        self.device = device
        self.preprocess = preprocess
        self.to_pil = to_pil

    def __call__(self, input):
        
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features

'''
code adapted from https://github.com/vita-epfl/Deep-Visual-Re-Identification-with-Confidence 
created by: George Adami
'''
import os

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

def init_model(*args, **kwargs):
    return ResNet50(*args, **kwargs)

def load_pretrained_weight(model, weight_path, device='cuda'):
        
    assert os.path.isfile(weight_path), 'Cannot find path to "{}"'.format(weight_path)

    checkpoint = torch.load(weight_path, map_location=device)

    # update the model
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Loaded pretrained weights from "{}"'.format(weight_path))

    return checkpoint


class ResNet50(nn.Module):
    def __init__(self, num_classes=751, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048
        self.cam = False

    def forward(self, x):
        x = self.base(x)
        if self.cam:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
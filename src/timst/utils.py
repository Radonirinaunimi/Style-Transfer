# Utils

import torch
import logging
import numpy as np

from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)


def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert("RGB")

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape

    in_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image


def imconvert(tensor):
    tensor = tensor.cpu().clone().detach()
    tensor = tensor.numpy().squeeze()
    tensor = tensor.transpose(1, 2, 0)
    tensor = tensor * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406)
    )
    tensor = tensor.clip(0, 1)
    return tensor


def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "28": "conv5_1",
            "21": "conv4_2",
        }
    features = {}
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features


def gram_matrix(tensor):
    batch_size, depth, height, width = tensor.shape
    tensor = tensor.view(depth, -1)
    tensor = torch.mm(tensor, tensor.t())
    return tensor

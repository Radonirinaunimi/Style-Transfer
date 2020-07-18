# This file should contain the main codes that controls the whole
# behaviour of the package.

import torch
import logging
import argparse
import matplotlib.pyplot as plt

from torchvision import models
from timst.utils import imconvert
from timst.utils import load_image
from timst.utils import gram_matrix
from timst.utils import get_features

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def args_parser():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Image style transfer.")
    parser.add_argument("-i", "--image", help="Input image", required=True)
    parser.add_argument("-s", "--style", help="Style image", required=True)
    args = parser.parse_args()
    return args


def main():

    args = args_parser()
    content = load_image(args.image)
    style = load_image(args.style)

    # Import Vgg pre-trained model
    vgg = models.vgg19(pretrained=True).features
    # Set grad to False
    for param in vgg.parameters():
        param.requires_grad_(False)
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Cuda Availability: {torch.cuda.is_available()}")
    vgg.to(device)

    style_features = get_features(style, vgg)
    content_features = get_features(content, vgg)
    style_grams = {
        layer: gram_matrix(style_features[layer]) for layer in style_features
    }
    # Start with content image for fast convergence
    # One can generate random image
    target = content.clone().requires_grad_(True).to(device)

    style_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.8,
        "conv3_1": 0.5,
        "conv4_1": 0.3,
        "conv5_1": 0.1,
    }
    content_weight = 1  # alpha
    style_weight = 5e6  # beta

    # Proceed to training
    optimizer = torch.optim.Adam([target], lr=0.003)

    steps = 2400
    print_every = 400

    for i in range(1, steps + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean(
            (content_features["conv4_2"] - target_features["conv4_2"]) ** 2
        )
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean(
                (target_gram - style_gram) ** 2
            )
            style_loss += layer_style_loss / (d * h * w)
        total_loss = style_weight * style_loss + content_weight * content_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % print_every == 0:
            print("Total Loss: ", total_loss.item())
            plt.imshow(imconvert(target))

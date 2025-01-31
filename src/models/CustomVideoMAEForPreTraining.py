import os
import torch
import random
import numpy as np
import torchvision.io
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor, VideoMAEConfig

def CustomVideoMAEForPreTraining(VideoMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
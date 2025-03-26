"""
Based on https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN 
(which is based on https://github.com/seoungwugoh/ivs-demo)

This version is much simplified. 
In this repo, we don't have
- local control
- fusion module
- undo
- timers

but with XMem as the backbone and is more memory (for both CPU and GPU) friendly
"""

import os
import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import numpy as np
import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')

from model.network import XMem

from inference.inference_core import InferenceCore

from .interactive_utils import *
from .interaction import *
from .gui_utils import *


class streaming_inference():
    def __init__(self, net: XMem, 
                config, device):
        super().__init__()

        self.initialized = False
        self.num_objects = config['num_objects']
        self.config = config
        self.processor = InferenceCore(net, config)
        self.processor.set_all_labels(list(range(1, self.num_objects+1)))
        self.device = device

    def step(self,frame,mask):
        self.current_image = frame
        self.current_mask = mask
        self.current_prob = index_numpy_to_one_hot_torch(self.current_mask, self.num_objects+1).to(self.device)
        self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(self.current_image, self.device)

        self.current_prob = self.processor.step(self.current_image_torch, self.current_prob[1:])
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)

        #self.current_image = frame
        #self.current_prob = index_numpy_to_one_hot_torch(self.current_mask, self.num_objects+1).to(self.device)
        #self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(self.current_image, self.device)

        self.current_prob = self.processor.step(self.current_image_torch)
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)
        
        torch.cuda.empty_cache()
        self.processor.clear_memory()

        return self.current_prob,self.current_mask





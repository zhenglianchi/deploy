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
from .s2m_controller import S2MController
from .fbrs_controller import FBRSController

from .interactive_utils import *
from .interaction import *
from .resource_manager import ResourceManager
from .gui_utils import *


class streaming_inference():
    def __init__(self, net: XMem, 
                resource_manager: ResourceManager, 
                s2m_ctrl:S2MController, 
                fbrs_ctrl:FBRSController, config, device):
        super().__init__()

        self.initialized = False
        self.num_objects = config['num_objects']
        self.s2m_controller = s2m_ctrl
        self.fbrs_controller = fbrs_ctrl
        self.config = config
        self.processor = InferenceCore(net, config)
        self.processor.set_all_labels(list(range(1, self.num_objects+1)))
        self.res_man = resource_manager
        self.device = device

        # image num
        self.num_frames = len(self.res_man)

        # self.height, self.width = self.get_image(0).shape[:2]
        self.height, self.width = self.res_man.h, self.res_man.w
        
        # current frame info
        self.curr_frame_dirty = False
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8) 
        self.current_image_torch = None
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_prob = torch.zeros((self.num_objects, self.height, self.width), dtype=torch.float).to(self.device)

        # initialize visualization
        self.viz_mode = 'davis'
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.cursur = 0
        self.on_showing = None

        self.interacted_prob = None
        self.overlay_layer = None
        self.overlay_layer_torch = None
        self.propagating = False
        # the object id used for popup/layered overlay
        self.vis_target_objects = [1]

        self.load_current_image_mask()

        self.initialized = True


    def load_current_image_mask(self, no_mask=False):
        # 得到当前图像
        self.current_image = self.res_man.get_image(self.cursur)
        self.current_image_torch = None

        if not no_mask:
            # 得到当前mask
            loaded_mask = self.res_man.get_mask(self.cursur)
            if loaded_mask is None:
                self.current_mask.fill(0)
            else:
                self.current_mask = loaded_mask.copy()
            self.current_prob = None

    def load_current_torch_image_mask(self, no_mask=False):
        if self.current_image_torch is None:
            self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(self.current_image, self.device)

        if self.current_prob is None and not no_mask:
            self.current_prob = index_numpy_to_one_hot_torch(self.current_mask, self.num_objects+1).to(self.device)

    def compose_current_im(self):
        self.viz = get_visualization(self.viz_mode, self.current_image, self.current_mask, 
                            self.overlay_layer, self.vis_target_objects)


    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.cursur, self.current_mask)

    def on_prev_frame(self):
        self.cursur = max(0, self.cursur-1)

    def on_next_frame(self):
        self.cursur = min(self.cursur+1, self.num_frames-1)

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_next_frame
            self.on_propagation()

    def on_propagation(self):
        # start to propagate
        self.load_current_torch_image_mask()

        print('Propagation started.')
        self.current_prob = self.processor.step(self.current_image_torch, self.current_prob[1:])
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)
        # clear
        self.interacted_prob = None
        self.propagating = True
        # propagate till the end
        while self.propagating:
            self.propagate_fn()
            self.load_current_image_mask(no_mask=True)
            self.load_current_torch_image_mask(no_mask=True)
            self.current_prob = self.processor.step(self.current_image_torch)
            self.current_mask = torch_prob_to_numpy_mask(self.current_prob)
            self.save_current_mask()
            if self.cursur == 0 or self.cursur == self.num_frames-1:
                break
        self.propagating = False


    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = f"{self.config['workspace']}/visualization/"
        save_folder = self.config['workspace']
        if os.path.exists(image_folder):
            # Sorted so frames will be in order
            print(f'Exporting visualization to {self.config["workspace"]}/visualization.mp4')
            images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape
            # 10 is the FPS -- change if needed
            video = cv2.VideoWriter(f"{save_folder}/visualization.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))
            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))
            video.release()
            print(f'Visualization exported to {self.config["workspace"]}/visualization.mp4')
        else:
            print(f'No visualization images found in {image_folder}')


    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        self.brush_vis_map = cv2.circle(self.brush_vis_map, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, color_map[self.current_object], thickness=-1)
        self.brush_vis_alpha = cv2.circle(self.brush_vis_alpha, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, 0.5, thickness=-1)



    def update_interacted_mask(self):
        self.current_prob = self.interacted_prob
        self.current_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.curr_frame_dirty = False


    def update_config(self):
        if self.initialized:
            self.config['min_mid_term_frames'] = self.work_mem_min.value()
            self.config['max_mid_term_frames'] = self.work_mem_max.value()
            self.config['max_long_term_elements'] = self.long_mem_max.value()
            self.config['num_prototypes'] = self.num_prototypes_box.value()
            self.config['mem_every'] = self.mem_every_box.value()

            self.processor.update_config(self.config)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            mps.empty_cache()


    def on_import_mask(self,file_name):
        if len(file_name) == 0:
            return

        # 读取当前mask
        mask = self.res_man.read_external_image(file_name, size=(self.height, self.width))

        shape_condition = (
            (len(mask.shape) == 2) and
            (mask.shape[-1] == self.width) and 
            (mask.shape[-2] == self.height)
        )

        object_condition = (
            mask.max() <= self.num_objects
        )

        if not shape_condition:
            print(f'Expected ({self.height}, {self.width}). Got {mask.shape} instead.')
        elif not object_condition:
            print(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            print(f'Mask file {file_name} loaded.')
            self.current_image_torch = self.current_prob = None
            self.current_mask = mask
            self.save_current_mask()



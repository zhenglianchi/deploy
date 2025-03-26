import numpy as np
import torch
from PIL import Image
import uvicorn
from fastapi import FastAPI
from typing import Any, Dict
from fastapi.responses import JSONResponse
from PIL import Image
import logging
import traceback
from dataclasses import dataclass
import draccus
import json_numpy

json_numpy.patch()
import os
from os import path
# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from argparse import ArgumentParser
# Arguments parsing
parser = ArgumentParser()
parser.add_argument('--model', default='./saves/XMem.pth')
parser.add_argument('--s2m_model', default='saves/s2m.pth')
parser.add_argument('--fbrs_model', default='saves/fbrs.pth')
parser.add_argument('--images', help='Folders containing input images.', default=None)
parser.add_argument('--video', help='Video file readable by OpenCV.', default="video.mp4")
parser.add_argument('--workspace', help='directory for storing buffered images (if needed) and output masks', default=None)

parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)

parser.add_argument('--num_objects', type=int, default=4)

# Long-memory options
# Defaults. Some can be changed in the GUI.
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128) 

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', type=int, default=10)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
parser.add_argument('--size', default=480, type=int, 
        help='Resize the shorter side to this size. -1 to use original resolution. ')
args = parser.parse_args()
import torch

from model.network import XMem
from inference.interact.s2m_controller import S2MController
from inference.interact.fbrs_controller import FBRSController
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M

from inference.interact.streaming import streaming_inference
from contextlib import nullcontext

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# create temporary workspace if not specified
config = vars(args)
config['enable_long_term'] = True
config['enable_long_term_count_usage'] = True


with torch.cuda.amp.autocast(enabled=not args.no_amp) if device.type == 'cuda' else nullcontext():
    # Load our checkpoint
    network = XMem(config, args.model, map_location=device).to(device).eval()

    inference = streaming_inference(network, config, device)


# === Server Interface ===
class XMEM_Server:
    def __init__(self):
        pass

    def track_mask(self, payload: Dict[str, Any]) -> str:
        try:
            frame = np.array(Image.fromarray(payload["image"]).convert("RGB"))
            mask = np.array(Image.fromarray(payload["mask"]))

            prob = None
            prob,mask = inference.step(frame,mask)

            result = {"masks":mask,"prob":prob.detach().cpu().numpy()}

            return JSONResponse(result)
        
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.array, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.track_mask)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "10.129.152.163"                                         # Host IP Address
    port: int = 8008                                                    # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = XMEM_Server()
    server.run(cfg.host, port=cfg.port)


if __name__ == '__main__':
    deploy()
    inference.on_clear_memory()
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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
from sam2.build_sam import build_sam2_camera_predictor
import json_numpy
import cv2
import warnings
warnings.filterwarnings("ignore")

json_numpy.patch()

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)


sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)


# === Server Interface ===
class SAM2_Server:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict_mask(self, payload: Dict[str, Any]) -> str:
        try:
            frame = np.array(Image.fromarray(payload["image"]).convert("RGB"))
            points = payload["points"]
            labels = payload["labels"]
            ann_obj_id = payload["obj_ids"]
            if_init = payload["if_init"]
            
            if if_init:
                self.predictor.load_first_frame(frame)
            
                ann_frame_idx = 0  # the frame index we interact with
                
                for i in range(len(points)):
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=ann_obj_id[i], points=[points[i]], labels=[labels[i]], clear_old_points=False
                    )
                result = {"obj_ids":np.array(out_obj_ids),"masks":out_mask_logits.cpu().numpy()}

            else:
                out_obj_ids, out_mask_logits = self.predictor.track(frame)

                result = {"obj_ids":np.array(out_obj_ids),"masks":out_mask_logits.cpu().numpy()}

            return JSONResponse(result)
        
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_mask)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "10.129.149.177"                                         # Host IP Address
    port: int = 8006                                                    # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = SAM2_Server(predictor)
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()

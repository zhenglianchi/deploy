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


json_numpy.patch()
np.random.seed(3)

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
    def __init__(self):
        pass

    def predict_mask(self, payload: Dict[str, Any]) -> str:
        try:
            # Parse payload components

            frame = Image.fromarray(payload["image"]).convert("RGB")
            boxes = torch.tensor(payload["input_box"]).to(device)
            if_init = payload["if_init"]
            
            if if_init:
                predictor.load_first_frame(frame)
            
                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
                
                if boxes.shape[0] != 0:
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=boxes,
                    )

                result = {"state":"initialized"}

                return JSONResponse(result)

            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)

                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                # print(all_mask.shape)
                for i in range(0, len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                        np.uint8
                    ) * 255

                    all_mask = cv2.bitwise_or(all_mask, out_mask)

                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

            result = {"masks":np.array(all_mask),"scores":np.array(out_mask_logits),"frame":np.array(frame)}

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
    server = SAM2_Server()
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()

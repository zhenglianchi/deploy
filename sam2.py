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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json_numpy

json_numpy.patch()
# select the device for computationfrom typing import Any, Dict
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


np.random.seed(3)

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)


# === Server Interface ===
class SAM2_Server:
    def __init__(self):
        pass

    def predict_mask(self, payload: Dict[str, Any]) -> str:
        try:
            # Parse payload components

            image = np.array(Image.fromarray(payload["image"]).convert("RGB"))
            input_box = np.array(payload["input_box"].tolist())
            predictor.set_image(image)
            
            masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

            result = {"masks":masks,"scores":scores}

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
        self.app.post("/act")(self.predict_mask)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "10.129.38.192"                                         # Host IP Address
    port: int = 8006                                                    # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = SAM2_Server()
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()

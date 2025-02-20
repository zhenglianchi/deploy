import os.path
import argparse
import os
import random
from collections import defaultdict

import cv2
import re
import warnings
import numpy as np
from PIL import Image
import torch
import html

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


#正则表达式，查找<p></p>之间的内容
def extract_substrings(string):
    # first check if there is no-finished bracket
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]

    pattern = r'<p>(.*?)\}(?!<)'
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]

    return substrings

#检查两个框是否重叠
def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

#计算两个框的交并比
def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou

#保存图像为临时文件
def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/home/mhm314/MiniGPT-4/tmp_img/" + file_name
    visual_img.save(file_path)
    return file_path

#将mask转换为边界框
def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''

    return bbox

#对文本字符markdown进行转义
def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text

#将转义的字符恢复为原始字符
def reverse_escape(text):
    md_chars = ['\\<', '\\>']

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (210, 210, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors)
}

used_colors = colors

#将边界框放在图像上
def visualize_all_bbox_together(image, generation):
    if image is None:
        return None, ''

    generation = html.unescape(generation)

    image_width, image_height = image.size
    image = image.resize([500, int(500 / image_width * image_height)])
    image_width, image_height = image.size

    string_list = extract_substrings(generation)
    if string_list:  # it is grounding or detection
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for string in string_list:
            try:
                obj, string = string.split('</p>')
            except ValueError:
                print('wrong string: ', string)
                continue
            bbox_list = string.split('<delim>')
            flag = False
            for bbox_string in bbox_list:
                integers = re.findall(r'-?\d+', bbox_string)
                if len(integers) == 4:
                    x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                    left = x0 / bounding_box_size * image_width
                    bottom = y0 / bounding_box_size * image_height
                    right = x1 / bounding_box_size * image_width
                    top = y1 / bounding_box_size * image_height

                    entities[obj].append([left, bottom, right, top])

                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        integers = re.findall(r'-?\d+', generation)

        if len(integers) == 4:  # it is refer
            mode = 'single'

            entities = list()
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            left = x0 / bounding_box_size * image_width
            bottom = y0 / bounding_box_size * image_height
            right = x1 / bounding_box_size * image_width
            top = y1 / bounding_box_size * image_height
            entities.append([left, bottom, right, top])
        else:
            # don't detect any valid bbox to visualize
            return None, ''

    if len(entities) == 0:
        return None, ''

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)

    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):

        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    indices = list(range(len(entities)))

    new_image = image.copy()

    previous_bboxes = []
    # size of text
    text_size = 0.5
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2

    # num_bboxes = sum(len(x[-1]) for x in entities)
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    color_id = -1
    for entity_idx, entity_name in enumerate(entities):
        if mode == 'single' or mode == 'identify':
            bboxes = entity_name
            bboxes = [bboxes]
        else:
            bboxes = entities[entity_name]
        color_id += 1
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm)

            color = used_colors[entity_idx % len(used_colors)]  # tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            if mode == 'all':
                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size,
                                                               text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (
                            text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                for prev_bbox in previous_bboxes:
                    if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and \
                            prev_bbox['phrase'] == entity_name:
                        skip_flag = True
                        break
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    # original color
                                    bg_color = color
                                else:
                                    # white
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(
                                    np.uint8)

                    cv2.putText(
                        new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
                    )

                    previous_bboxes.append(
                        {'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})

    if mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color

        color_gen = color_iterator(colors)

        # Add colors to phrases and remove <p></p>
        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'

        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
    else:
        generation_colored = ''

    pil_image = Image.fromarray(new_image)
    return pil_image, generation_colored


def gradio_ask(user_message, chat_state, gr_img, img_list = []):
    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    else:
        mask = None

    if '[identify]' in user_message:
        # check if user provide bbox in the text input
        integers = re.findall(r'-?\d+', user_message)
        if len(integers) != 4:  # no bbox in text
            bbox = mask2bbox(mask)
            user_message = user_message + bbox

    if chat_state is None:
        chat_state = CONV_VISION.copy()

    img_list = chat.upload_img(gr_img, chat_state, img_list)

    chat.ask(user_message, chat_state)

    if '[identify]' in user_message:
        visual_img, _ = visualize_all_bbox_together(gr_img, user_message)
        if visual_img is not None:
            file_path = save_tmp_img(visual_img)

    return chat_state,img_list


def gradio_answer(chat_state, img_list, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[-1], torch.Tensor):
            chat.encode_img(img_list)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              temperature=temperature,
                              max_new_tokens=500,
                              max_length=2000)[0]
    return llm_message,chat_state


def gradio_visualize(llm_message, gr_img):
    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    file_path = None
    unescaped = reverse_escape(llm_message)
    visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped)
    if visual_img is not None:
        if len(generation_color):
            llm_message = generation_color
        file_path = save_tmp_img(visual_img)
    return file_path


chat = Chat(model, vis_processor, device=device)

# === Server Interface ===
class Minigpt4V2_Server:
    def __init__(self):
        self.chat_state = None

    def predict_answer(self, payload: Dict[str, Any]) -> str:
        try:
            # Parse payload components

            image, instruction = Image.fromarray(payload["image"]).convert("RGB"), payload["instruction"]
            self.chat_state,img_list = gradio_ask(instruction,self.chat_state,image)

            print("用户输入:",instruction)
            
            # Run MiniGPT4v2 Inference
            llm_message,self.chat_state = gradio_answer(self.chat_state,img_list,temperature=0.4)

            print("LLM answer:",llm_message)

            file_path = gradio_visualize(llm_message,image)
            if file_path:
                print("识别结果文件保存在",file_path)

            result = {"llm_message":llm_message}

            self.chat_state = None

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
        self.app.post("/act")(self.predict_answer)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "10.129.38.192"                                         # Host IP Address
    port: int = 8001                                                    # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = Minigpt4V2_Server()
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
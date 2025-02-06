# The following codes are based on OSWorld(https://github.com/xlang-ai/OSWorld/tree/main).
# For codes from mm_agents, please refer to the original OSWorld repository.

import base64
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
import io
from io import BytesIO
from typing import Dict, List
import yaml
import backoff
import pandas as pd
from datetime import datetime
import copy
import random
# import google.generativeai as genai
import openai
import requests
from PIL import Image
from google.api_core.exceptions import (
    InvalidArgument,
    ResourceExhausted,
    InternalServerError,
    BadRequest,
)
from requests.exceptions import SSLError

from mm_agents.accessibility_tree_wrap.heuristic_retrieve import (
    filter_nodes,
    draw_bounding_boxes,
)
from .prompts import (
    multi_step_action_w_thought_template_m03_long_ch,
    multi_step_prompt_1229,
    Call_User_Reflection_Action_Space,
    Reflection_Action_Space,
    no_thought_prompt_0103,
)

from .action_parser import (
    parse_action_qwen2vl,
    parsing_response_to_pyautogui_code,
    FINISH_WORD,
    WAIT_WORD,
    ENV_FAIL_WORD,
    CALL_USER
)

import math
import numpy as np
from transformers import ProcessorMixin, AutoProcessor
from openai import OpenAI

logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ["a11y_tree"]

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py


# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")

def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right", trust_remote_code=True)

def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image

def pil_img_2_bytes(image):
    # 创建一个BytesIO对象
    byte_io = io.BytesIO()

    # 将图像保存到BytesIO对象中，指定格式
    image.save(byte_io, format='PNG')  # 格式可以是'JPEG', 'PNG'等
    # 获取图像的字节表示
    image_bytes = byte_io.getvalue()
    return image_bytes

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    ]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text
                if '"' not in node.text
                else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith(
            "EditWrapper"
        ) and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (
                node_text
                if '"' not in node_text
                else '"{:}"'.format(node_text.replace('"', '""'))
            )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                (
                    node.get("{{{:}}}class".format(_attributes_ns), "")
                    if platform == "ubuntu"
                    else node.get("{{{:}}}class".format(class_ns_windows), "")
                ),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get("{{{:}}}screencoord".format(_component_ns), ""),
                node.get("{{{:}}}size".format(_component_ns), ""),
            )
        )

    return "\n".join(linearized_accessibility_tree)


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(
        ET.fromstring(accessibility_tree), platform=platform, check_image=True
    )
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(
        nodes, screenshot
    )

    return marks, drew_nodes, tagged_screenshot, element_list


def parse_actions_from_string(input_string):
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r"```json\s+(.*?)\s+```", input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r"```\s+(.*?)\s+```", input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_code_from_string(input_string):
    input_string = "\n".join(
        [line.strip() for line in input_string.split(";") if line.strip()]
    )
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = [
            "WAIT",
            "DONE",
            "FAIL",
        ]  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


def parse_code_from_som_string(input_string, masks):
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += (
            "tag_"
            + str(i + 1)
            + "="
            + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        )
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ["WAIT", "DONE", "FAIL"]:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    # enc = tiktoken.encoding_for_model("gpt-4")
    # tokens = enc.encode(linearized_accessibility_tree)
    # if len(tokens) > max_tokens:
    #     linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
    #     linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree

class PromptAgent:
    def __init__(
        self,
        platform="ubuntu",
        max_tokens=1000,
        top_p=0.9,
        top_k=1.0,
        temperature=0.0,
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
        max_trajectory_length=50,
        a11y_tree_max_tokens=10000,
        runtime_conf: dict = {}
    ):
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.runtime_conf = runtime_conf
        self.vlm = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="empty",
        ) # should replace with your UI-TARS server api
        self.infer_mode = self.runtime_conf["infer_mode"]
        self.prompt_style = self.runtime_conf["prompt_style"]
        self.input_swap = self.runtime_conf["input_swap"]
        self.language = self.runtime_conf["language"]
        self.max_steps = self.runtime_conf["max_steps"]

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        
        self.prompt_action_space = Reflection_Action_Space
        self.customize_action_parser = parse_action_qwen2vl
        self.action_parse_res_factor = 1000
        if self.infer_mode == "qwen2vl_user":
            self.prompt_action_space = Call_User_Reflection_Action_Space
    
        self.prompt_template = multi_step_action_w_thought_template_m03_long_ch
        self.system_prompt_template = multi_step_action_w_thought_template_m03_long_ch
        
        if self.prompt_style == "qwen2vl_user":
            self.prompt_template = multi_step_prompt_1229

        elif self.prompt_style == "qwen2vl_no_thought":
            self.prompt_template = no_thought_prompt_0103

        
        if "history_n" in self.runtime_conf:
            self.history_n = self.runtime_conf["history_n"]
        else:
            self.history_n = 5

    def predict(
        self, instruction: str, obs: Dict, last_action_after_obs: Dict = None
    ) -> List:
        """
        Predict the next action(s) based on the current observation.
        """

        # Append trajectory
        # print(len(self.observations), len(self.actions), len(self.actions))
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(
            self.thoughts
        ), "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length :]
                _actions = self.actions[-self.max_trajectory_length :]
                _thoughts = self.thoughts[-self.max_trajectory_length :]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(
            _observations, _actions, _thoughts
        ):
            # {{{1
            if self.observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

            else:
                raise ValueError(
                    "Invalid observation_type type: " + self.observation_type
                )  # 1}}}

        if last_action_after_obs is not None and self.infer_mode == "double_image":
            self.history_images.append(last_action_after_obs["screenshot"])

        self.history_images.append(obs["screenshot"])

        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = obs["screenshot"]
            try:
                linearized_accessibility_tree = (
                    linearize_accessibility_tree(
                        accessibility_tree=obs["accessibility_tree"],
                        platform=self.platform,
                    )
                    if self.observation_type == "screenshot_a11y_tree"
                    else None
                )
            except:
                linearized_accessibility_tree = None
            # logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append(
                    {
                        "screenshot": base64_image,
                        "accessibility_tree": linearized_accessibility_tree,
                    }
                )
            else:
                self.observations.append(
                    {"screenshot": base64_image, "accessibility_tree": None}
                )

        else:
            raise ValueError(
                "Invalid observation_type type: " + self.observation_type
            )  # 1}}}
        
        if self.infer_mode == "qwen2vl_user":
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                action_space=self.prompt_action_space,
                language=self.language
            )
        elif self.infer_mode == "qwen2vl_no_thought":
            user_prompt = self.prompt_template.format(
                instruction=instruction
            )

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        max_pixels = 1350 * 28 * 28
        min_pixels = 100 * 28 * 28
        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")
        max_image_nums_under_32k = int(32768*0.75/max_pixels*28*28)
        if len(self.history_images) > max_image_nums_under_32k:
            num_of_images = min(5, len(self.history_images))
            max_pixels = int(32768*0.75) // num_of_images

        for turn, image in enumerate(self.history_images):
            if len(images) >= 5:
                break
            try:
                image = Image.open(BytesIO(image))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            if image.width * image.height > max_pixels:
                """
                如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
                """
                resize_factor = math.sqrt(max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                image = image.resize((width, height))
            if image.width * image.height < min_pixels:
                resize_factor = math.sqrt(min_pixels / (image.width * image.height))
                width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
                image = image.resize((width, height))

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):

                    cur_image = images[image_num]
                    encoded_string = pil_to_base64(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                    })
                    image_num += 1
                    
                messages.append({
                    "role": "assistant",
                    "content": [history_response]
                })

            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1
        
        else:
            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1

        try_times = 3
        while True:
            if try_times <= 0:
                print(f"Reach max retry times to fetch response from client, as error flag.")
                return "client error", ["DONE"], []
            try:
                
                response = self.vlm.chat.completions.create(
                    model="ui-tars",
                    messages=messages,
                    frequency_penalty=1,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p
                )
                # print(response.choices[0].message.content)
                prediction = response.choices[0].message.content.strip()
                
                prediction = response[0]["prediction"].strip()
                parsed_responses = self.customize_action_parser(
                    prediction,
                    self.action_parse_res_factor,
                    self.runtime_conf["screen_height"],
                    self.runtime_conf["screen_width"]
                )
                break
            except Exception as e:
                print(f"Error when fetching response from client, with response: {response}")
                prediction = None
                try_times -= 1
                
        if prediction is None:
            return "client error", ["DONE"], []
        
        self.history_responses.append(prediction)
        self.thoughts.append(prediction)

        try:
            parsed_responses = self.customize_action_parser(
                prediction,
                self.action_parse_res_factor,
                self.runtime_conf["screen_height"],
                self.runtime_conf["screen_width"]
            )
        except Exception as e:
            print(f"Parsing action error: {prediction}, with error:\n{e}")
            return f"Parsing action error: {prediction}, with error:\n{e}", ["DONE"], []

        actions = []
        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:

                if parsed_response["action_type"] == FINISH_WORD:
                    self.actions.append(actions)
                    return prediction, ["DONE"], parsed_responses
                
                elif parsed_response["action_type"] == WAIT_WORD:
                    self.actions.append(actions)
                    return prediction, ["WAIT"], parsed_responses
                
                elif parsed_response["action_type"] == ENV_FAIL_WORD:
                    self.actions.append(actions)
                    return prediction, ["FAIL"], parsed_responses 

                elif parsed_response["action_type"] == CALL_USER:
                    self.actions.append(actions)
                    return prediction, ["FAIL"], parsed_responses 
            
            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_response,
                self.runtime_conf["screen_height"],
                self.runtime_conf["screen_width"],
                self.input_swap
            )
            actions.append(pyautogui_code)

        self.actions.append(actions)

        if len(self.history_responses) >= self.max_trajectory_length:
            # Default to FAIL if exceed max steps
            actions = ["FAIL"]

        return prediction, actions, parsed_responses

    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
            # General exceptions
            SSLError,
            # OpenAI exceptions
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            # Google exceptions
            InvalidArgument,
            ResourceExhausted,
            InternalServerError,
            BadRequest,
            # Groq exceptions
            # todo: check
        ),
        interval=30,
        max_tries=10,
    )
    
    def parse_actions(self, response: str, masks=None):

        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions
        elif self.observation_type in ["som"]:
            # parse from the response
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions

    def reset(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []

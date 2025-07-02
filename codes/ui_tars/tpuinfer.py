#!/usr/bin/env python3
import os
os.environ["PT_XLA_DEBUG"] = "1"  # Enable XLA debug messages

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import pyautogui
import time

# SPMD
xr.use_spmd(auto=True)
assert xr.is_spmd()

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

def _mp_fn(index):
    device = xm.xla_device()
    logging.info(f"Process {index} -> device {device}")
    torch.randn(1, device=device)
    xm.mark_step()

    if xm.is_master_ordinal():
        logging.info(f"[SPMD] TPU cores: {xr.global_runtime_device_count()}")

    # Load model and processor
    logging.info("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B", use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "ByteDance-Seed/UI-TARS-1.5-7B",
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    xm.mark_step()
    logging.info("Model loaded and ready.")

    for step in range(20):
        logging.info(f"=== Step {step} START ===")
        if xm.is_master_ordinal():
            screenshot = pyautogui.screenshot()
            img_tensor = pil_to_tensor(screenshot)
        else:
            img_tensor = torch.zeros(3, 1, 1, dtype=torch.uint8)

        logging.info("Broadcasting image tensor...")
        tensor_list = [img_tensor.to(device)]
        xm.collective_broadcast(tensor_list)
        xm.mark_step()
        logging.info("Broadcast complete.")

        img_tensor = tensor_list[0].to('cpu')
        if xm.is_master_ordinal():
            screenshot = to_pil_image(img_tensor)
        else:
            screenshot = None

        prompt = COMPUTER_USE_DOUBAO.format(instruction="Find a folder called ui-tars", language="English")
        conv = [{"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": prompt}]}]

        logging.info("Tokenizing inputs...")
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        logging.info("Inputs ready.")

        if xm.is_master_ordinal():
            logging.info("Generating response...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        xm.mark_step()
        logging.info("Generation complete.")

        if xm.is_master_ordinal():
            w, h = screenshot.size
            mh, mw = smart_resize(h, w)
            in_len = inputs['input_ids'].shape[1]
            gen_ids = output_ids[:, in_len:]
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            logging.info(f"Raw output: {text[:200]}")

            actions = parse_action_to_structure_output(text, 1000, mh, mw, "qwen25vl")
            if not actions:
                logging.warning("No actions parsed — exiting.")
                break

            code = parsing_response_to_pyautogui_code(actions, h, w)
            logging.info(f"Generated code: {code[:200]}")

            if code.strip() == "DONE":
                logging.info("Task complete.")
                break

            logging.info("Executing code...")
            try:
                exec(code)
                logging.info("Code executed.")
                time.sleep(2)
            except Exception as e:
                logging.error(f"Execution error: {e}")
                break

        xm.rendezvous(f"step_complete_{step}")
        logging.info(f"=== Step {step} END ===")

    if xm.is_master_ordinal():
        logging.info("All steps completed.")

if __name__ == "__main__":
    logging.info("Starting xmp.spawn...")
    xmp.spawn(_mp_fn)

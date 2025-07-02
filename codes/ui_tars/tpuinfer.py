# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp

# Enable SPMD auto-sharding
xr.use_spmd(auto=True)
assert xr.is_spmd(), "SPMD mode not enabled!"

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import pyautogui

# For custom parsing / prompts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

def _mp_fn(index):
    device = xm.xla_device()
    torch.randn(1, device=device)  # warm-up
    xm.mark_step()

    if xm.is_master_ordinal():
        print(f"[SPMD] Running with {xr.global_runtime_device_count()} TPU cores")

    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # Load weights once (duplicate due to SPMD logical device)
    if xm.is_master_ordinal():
        print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    xm.mark_step()

    if xm.is_master_ordinal():
        print("[SPMD] Model loaded and replicated across shards.")

    # Main inference loop
    for step in range(20):
        if xm.is_master_ordinal():
            print(f"\n--- Step {step + 1}/20 ---")
            screenshot = pyautogui.screenshot()
        else:
            screenshot = None

        lst = [screenshot]
        xm.collective_broadcast(lst)
        screenshot = lst[0]

        formatted_prompt = COMPUTER_USE_DOUBAO.format(
            instruction="Find a folder called ui-tars", language="English"
        )
        full_conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": formatted_prompt}
            ]
        }]

        inputs = processor.apply_chat_template(
            full_conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        if xm.is_master_ordinal():
            print("Generating response...")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )

        xm.mark_step()

        if xm.is_master_ordinal():
            original_w, original_h = screenshot.size
            mh, mw = smart_resize(original_h, original_w)
            input_len = inputs['input_ids'].shape[1]
            gen_ids = output_ids[:, input_len:]
            raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            print("\n--- Raw Model Output ---")
            print(raw_text)

            actions = parse_action_to_structure_output(raw_text, 1000, mh, mw, "qwen25vl")
            if not actions:
                print("No valid actions parsed — exiting.")
                break

            py_code = parsing_response_to_pyautogui_code(actions, original_h, original_w)
            print("\n--- PyAutoGUI Code ---")
            print(py_code)

            if py_code.strip() == "DONE":
                print("Task complete!")
                break

            print("Executing actions...")
            try:
                exec(py_code)
                time.sleep(2)
            except Exception as e:
                print(f"Error executing actions: {e}")
                break

        xm.rendezvous(f"step_complete_{step}")

    if xm.is_master_ordinal():
        print("\n--- UI Agent Completed ---")

if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=(), nprocs=None)

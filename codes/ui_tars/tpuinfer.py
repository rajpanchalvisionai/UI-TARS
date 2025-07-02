#!/usr/bin/env python3
# tpu_spmd_infer.py

import os
# === Set before importing torch_xla ===
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_DEBUG", "1")
os.environ.setdefault("XLA_USE_SPMD", "1")
os.environ.setdefault("XLA_AUTO_SPMD", "1")

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import pyautogui, sys, time, os
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

def _mp_fn(index):
    # === Enable SPMD auto-sharding ===
    xr.use_spmd(auto=True)
    assert xr.is_spmd(), "SPMD not enabled"

    device = xm.xla_device()
    torch.randn(1, device=device); xm.mark_step()

    if xm.is_master_ordinal():
        print("All TPU processes ready, SPMD mode ON")

    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # Load model and processor (bf16)
    if xm.is_master_ordinal():
        print("Master loading model")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()
    model.to(device)

    xm.mark_step()
    xm.rendezvous("model_ready")
    if xm.is_master_ordinal():
        print("Model ready across TPU cores with SPMD")

    # Main loop
    for step in range(20):
        if xm.is_master_ordinal():
            print(f"\n--- STEP {step+1}/20 ---")
            screenshot = pyautogui.screenshot()
        else:
            screenshot = None

        buffer = [screenshot]; xm.collective_broadcast(buffer)
        screenshot = buffer[0]

        user_instruction = "Find a folder called ui-tars"
        prompt = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language="English")
        conv = [{"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": prompt}]}]

        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k,v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        if xm.is_master_ordinal():
            print("Generating response...")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500, pad_token_id=processor.tokenizer.eos_token_id)
        xm.mark_step()

        if xm.is_master_ordinal():
            orig_w, orig_h = screenshot.size
            ih, iw = smart_resize(orig_h, orig_w)
            in_len = inputs['input_ids'].shape[1]
            generated = output_ids[:, in_len:]
            out_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            print("\nRAW OUTPUT:", out_text)

            parsed = parse_action_to_structure_output(out_text, 1000, ih, iw, "qwen25vl")
            if not parsed:
                print("No actions → stopping."); break

            code = parsing_response_to_pyautogui_code(parsed, orig_h, orig_w)
            print("Generated PyAutoGUI Code:\n", code)
            if code.strip() == "DONE":
                print("Task completed."); break

            try:
                exec(code)
                time.sleep(2)
            except Exception as e:
                print("Exec error:", e); break

        xm.rendezvous("step_complete")

    if xm.is_master_ordinal():
        print("=== SPMD TPU Agent Finished ===")

if __name__ == "__main__":
    xmp.spawn(_mp_fn)

# -*- coding: utf-8 -*-
import os, sys, time, torch
from PIL import Image
import pyautogui
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

# Enable SPMD auto-sharding
xr.use_spmd(auto=True)
assert xr.is_spmd(), "SPMD mode not enabled!"

# Set HF cache directory
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")  # customize as needed

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

def _mp_fn(index):
    device = xm.xla_device()
    torch.randn(1, device=device); xm.mark_step()
    if xm.is_master_ordinal():
        print(f"[SPMD] Running on {xr.global_runtime_device_count()} TPU cores")

    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"
    cache_dir = os.environ["HF_HOME"]

    processor = AutoProcessor.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir
    ).to(device)
    model.eval()
    xm.mark_step()
    if xm.is_master_ordinal():
        print(f"[SPMD] Model & processor loaded from {cache_dir}")

    for step in range(20):
        if xm.is_master_ordinal():
            print(f"\n🌟 Step {step + 1}/20")
            screenshot = pyautogui.screenshot()
            img_tensor = pil_to_tensor(screenshot)  # C × H × W
        else:
            img_tensor = torch.zeros(3, 1, 1, dtype=torch.uint8)

        print("[Debug] Broadcasting image tensor…")
        tlist = [img_tensor.to(device)]
        xm.collective_broadcast(tlist)
        img_tensor = tlist[0].to("cpu")
        if xm.is_master_ordinal():
            screenshot = to_pil_image(img_tensor)
        else:
            screenshot = None

        print("[Debug] Applying processor template…")
        prompt = COMPUTER_USE_DOUBAO.format(instruction="Find a folder called ui-tars", language="English")
        conv = [{"role": "user", "content": [{"type":"image","image":screenshot},{"type":"text","text":prompt}]}]
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        if xm.is_master_ordinal():
            print("[Debug] Starting generation…")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        xm.mark_step()
        if xm.is_master_ordinal():
            print("[Debug] Generation complete.")

        xm.rendezvous(f"post_gen_{step}")

        if xm.is_master_ordinal():
            w, h = screenshot.size
            mh, mw = smart_resize(h, w)
            in_len = inputs['input_ids'].shape[1]
            gen = output_ids[:, in_len:]
            text = processor.batch_decode(gen, skip_special_tokens=True)[0]
            print("\n--- Raw Output ---")
            print(text)

            actions = parse_action_to_structure_output(text, 1000, mh, mw, "qwen25vl")
            if not actions:
                print("❌ No valid actions — exiting.")
                break

            code = parsing_response_to_pyautogui_code(actions, h, w)
            print("\n--- PyAutoGUI Code ---")
            print(code)

            if code.strip() == "DONE":
                print("✅ Task completed.")
                break

            print("➡️ Executing actions…")
            try:
                exec(code)
                time.sleep(2)
            except Exception as e:
                print("⚠️ Error executing:", e)
                break

        xm.rendezvous(f"step_complete_{step}")

    if xm.is_master_ordinal():
        print("\n🏁 UI Agent Finished.")

if __name__ == "__main__":
    xmp.spawn(_mp_fn)

# --- MODIFIED IMPORTS FOR MODERN FSDP ---
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.fsdp as xla_fsdp
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# This is the specific transformer block class for the Qwen2.5-VL model
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5VLAttention 
# --- END MODIFIED IMPORTS ---

from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import time
import pyautogui
import sys
import math

# Add parent directory to sys.path to import action_parser and prompt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

# --- FSDP PARALLELISM SETUP ---
def _mp_fn(index):
    device = xm.xla_device()

    # --- Model and Processor Setup ---
    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # --- Load Model and Shard with MODERN FSDP ---
    if xm.is_master_ordinal():
        print("Master process loading processor...")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    
    # --- THIS IS THE KEY API CHANGE ---
    # Define the wrapping policy using the standard PyTorch method.
    # We tell it to find and shard any layer of type Qwen2_5VLAttention.
    qwen_fsdp_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5VLAttention,
        },
    )
    # --- END API CHANGE ---

    if xm.is_master_ordinal():
        print("Master process loading model for sharding...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    
    if xm.is_master_ordinal():
        print("Applying FSDP and sharding the model across all 8 cores...")
    
    # We still use XlaFSDP, but now we pass it the new, correct policy.
    model = xla_fsdp.XlaFSDP(model, auto_wrap_policy=qwen_fsdp_policy)
    
    model.to(device)
    model.eval()

    xm.rendezvous('model_ready')
    if xm.is_master_ordinal():
        print("Model sharded and ready on all cores.")

    COORDINATE_PARSING_FACTOR = 1000

    max_steps = 20
    for step in range(max_steps):
        if xm.is_master_ordinal():
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            print("Taking screenshot...")
            screenshot = pyautogui.screenshot()
            img = screenshot
        else:
            img = Image.new('RGB', (100, 100))

        object_list = [img]
        torch.distributed.broadcast_object_list(object_list, src=0)
        img = object_list[0]

        if xm.is_master_ordinal():
            original_width, original_height = img.size
            print(f"Screenshot taken. Original dimensions: {original_width}x{original_height}")
            model_input_height, model_input_width = smart_resize(original_height, original_width)
            print(f"Image will be processed by model at effective dimensions: {model_input_width}x{model_input_height}")
        else:
            original_width, original_height, model_input_height, model_input_width = 1, 1, 1, 1

        user_instruction = "Find a folder called ui-tars"
        formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')

        full_conversation = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": formatted_prompt_text}]}
        ]

        if xm.is_master_ordinal():
            print("Applying chat template and tokenizing...")
        inputs = processor.apply_chat_template(
            full_conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)

        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        if xm.is_master_ordinal():
            print("Generating response on all cores...")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=500, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
            )

        if xm.is_master_ordinal():
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, input_length:]
            raw_model_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print("\n--- Raw Model Output ---")
            print(raw_model_output_text)
            print("------------------------")

            parsed_actions = parse_action_to_structure_output(
                raw_model_output_text,
                factor=COORDINATE_PARSING_FACTOR,
                origin_resized_height=model_input_height,
                origin_resized_width=model_input_width,
                model_type="qwen25vl"
            )

            if not parsed_actions:
                print("No valid action parsed. Stopping.")
                break

            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_actions, image_height=original_height, image_width=original_width
            )

            print("\n--- Generated PyAutoGUI Code ---")
            print(pyautogui_code)
            print("------------------------------")

            if pyautogui_code.strip() == "DONE":
                print("Task finished by agent.")
                break

            print("Executing PyAutoGUI code...")
            try:
                exec(pyautogui_code)
                print("Code executed. Waiting a moment...")
                time.sleep(2)
            except Exception as e:
                print(f"Error executing PyAutoGUI code: {e}")
                break
        
        xm.rendezvous('step_complete')

    if xm.is_master_ordinal():
        print("\n--- UI Agent Finished ---")

# --- Launcher for torchrun ---
if __name__ == '__main__':
    _mp_fn(0)

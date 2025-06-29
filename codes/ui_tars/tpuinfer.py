# --- FINAL WORKING IMPORTS ---
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
# --- END IMPORTS ---

from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import time
import pyautogui
import sys

# Add parent directory to sys.path to import action_parser and prompt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO


def qwen_fsdp_policy(module, recurse, unwrapped_params):
    return transformer_auto_wrap_policy(
        module, recurse, unwrapped_params,
        transformer_layer_cls={Qwen2Attention}
    )


# --- FSDP PARALLELISM SETUP ---
def _mp_fn(index):
    # NOTE: xmp.spawn handles all necessary initialization.
    # We DO NOT call dist.init_process_group ourselves.
    device = xm.xla_device()
    
    # We must initialize torch.distributed to use the broadcast function,
    # but we do it *after* xmp has set up the world.
    dist.init_process_group('xla')

    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # --- MODEL LOADING (Main Process Only) ---
    # To save memory, only the master process loads the model from disk.
    if xm.is_master_ordinal():
        print("Master process loading model and processor...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
    else:
        # Other processes wait for the master.
        processor = None
        model = None
    
    # --- BROADCAST MODEL & PROCESSOR ---
    # The master sends the loaded model and processor to all other processes.
    # This ensures every process has an identical copy before sharding.
    object_list = [model, processor]
    dist.broadcast_object_list(object_list, src=0)
    model, processor = object_list
    
    # --- FSDP WRAPPING (All Processes) ---
    # Now that every process has the model, they all participate in sharding.
    if xm.is_master_ordinal():
        print("Applying FSDP and sharding the model across all TPU cores...")
    
    model = XlaFullyShardedDataParallel(
        model, 
        auto_wrap_policy=qwen_fsdp_policy
    )
    
    model.to(device)
    model.eval()

    xm.rendezvous('model_ready')
    if xm.is_master_ordinal():
        print("Model sharded and ready on all cores.")

    # --- MAIN AGENT LOOP ---
    max_steps = 20
    for step in range(max_steps):
        if xm.is_master_ordinal():
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            print("Taking screenshot...")
            screenshot = pyautogui.screenshot()
            
            # --- PRE-PROCESSING (Main Process Only) ---
            user_instruction = "Find a folder called ui-tars"
            formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')
            full_conversation = [
                {"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": formatted_prompt_text}]}
            ]
            inputs = processor.apply_chat_template(
                full_conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
            # Create a list of tensors to broadcast
            tensors_to_broadcast = [inputs['input_ids'], inputs['pixel_values']]
        else:
            # Other processes create empty placeholders to receive the broadcasted data
            tensors_to_broadcast = [torch.empty((1, 1342), dtype=torch.long), torch.empty((1, 3, 336, 336), dtype=torch.float32)]

        # --- BROADCAST TENSORS ---
        # The master sends the processed input_ids and pixel_values to all other processes.
        dist.broadcast(tensors_to_broadcast[0], src=0)
        dist.broadcast(tensors_to_broadcast[1], src=0)
        
        # All processes reassemble the `inputs` dictionary
        inputs = {'input_ids': tensors_to_broadcast[0].to(device), 'pixel_values': tensors_to_broadcast[1].to(device).to(torch.bfloat16)}

        # --- GENERATION (All Processes) ---
        if xm.is_master_ordinal():
            print("Generating response on all cores...")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=500, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
            )

        # --- EXECUTION (Main Process Only) ---
        if xm.is_master_ordinal():
            original_width, original_height = screenshot.size
            model_input_height, model_input_width = smart_resize(original_height, original_width)
            
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, input_length:]
            raw_model_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print("\n--- Raw Model Output ---")
            print(raw_model_output_text)

            # ... (rest of your parsing and pyautogui execution code, which is already correct) ...
            parsed_actions = parse_action_to_structure_output(raw_model_output_text, 1000, model_input_height, model_input_width, "qwen25vl")
            if not parsed_actions:
                break
            pyautogui_code = parsing_response_to_pyautogui_code(parsed_actions, original_height, original_width)
            print("\n--- Generated PyAutoGUI Code ---")
            print(pyautogui_code)
            if pyautogui_code.strip() == "DONE":
                break
            exec(pyautogui_code)
            time.sleep(2)

        xm.rendezvous('step_complete')

    if xm.is_master_ordinal():
        print("\n--- UI Agent Finished ---")


# --- Launcher for xmp.spawn ---
if __name__ == '__main__':
    # You were correct to use xmp.spawn() for this setup.
    xmp.spawn(_mp_fn, nprocs=8)

# --- FINAL WORKING IMPORTS ---
import torch
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
    # NOTE: No dist.init_process_group. xmp.spawn handles the context.
    device = xm.xla_device()
    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # To save memory, only the master process downloads the model.
    if xm.is_master_ordinal():
        print("Master process loading model and processor...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
    else:
        # Other processes wait for the master to download.
        xm.rendezvous('download_done')
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)

    if xm.is_master_ordinal():
        xm.rendezvous('download_done') # Master signals download is complete.
        
    if xm.is_master_ordinal():
        print("Applying FSDP and sharding the model across all TPU cores...")
    
    model = XlaFullyShardedDataParallel(
        model, 
        auto_wrap_policy=qwen_fsdp_policy
    )
    
    model.to(device)
    model.eval()

    xm.rendezvous('model_ready') # Wait for all processes to finish sharding.
    if xm.is_master_ordinal():
        print("Model sharded and ready on all cores.")

    # --- MAIN AGENT LOOP ---
    max_steps = 20
    for step in range(max_steps):
        # Master process handles the GUI and preprocessing
        if xm.is_master_ordinal():
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            print("Taking screenshot...")
            screenshot = pyautogui.screenshot()
            
            user_instruction = "Find a folder called ui-tars"
            formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')
            full_conversation = [{"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": formatted_prompt_text}]}]
            inputs = processor.apply_chat_template(
                full_conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
            tensors_to_broadcast = [inputs['input_ids'].to(device), inputs['pixel_values'].to(device)]
        else:
            # --- THIS IS THE FINAL FIX ---
            # Other processes create empty placeholders DIRECTLY ON THE TPU DEVICE.
            tensors_to_broadcast = [
                torch.empty((1, 1342), dtype=torch.long, device=device), 
                torch.empty((1, 3, 336, 336), dtype=torch.float32, device=device)
            ]
            # --- END FIX ---

        # The broadcast will now succeed because all tensors are on the TPU.
        xm.collective_broadcast(tensors_to_broadcast)
        
        # All processes reassemble the `inputs` dictionary
        inputs = {'input_ids': tensors_to_broadcast[0], 'pixel_values': tensors_to_broadcast[1].to(torch.bfloat16)}

        # --- GENERATION (All Processes Participate) ---
        if xm.is_master_ordinal():
            print("Generating response on all cores...")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=500, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
            )

        # --- EXECUTION (Main Process Only) ---
        if xm.is_master_ordinal():
            original_width, original_height = 1024, 768
            model_input_height, model_input_width = smart_resize(original_height, original_width)
            
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, input_length:]
            raw_model_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print("\n--- Raw Model Output ---")
            print(raw_model_output_text)

            parsed_actions = parse_action_to_structure_output(raw_model_output_text, 1000, model_input_height, model_input_width, "qwen25vl")
            if not parsed_actions:
                print("No valid action parsed. Stopping.")
                break

            pyautogui_code = parsing_response_to_pyautogui_code(parsed_actions, original_height, original_width)
            print("\n--- Generated PyAutoGUI Code ---")
            print(pyautogui_code)
            
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

        # All processes wait here to stay in sync
        xm.rendezvous('step_complete')

    if xm.is_master_ordinal():
        print("\n--- UI Agent Finished ---")


# --- Launcher for xmp.spawn ---
if __name__ == '__main__':
    # We use the default nprocs=None to let xmp use all available cores.
    xmp.spawn(_mp_fn)

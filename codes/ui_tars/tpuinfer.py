# --- FINAL WORKING IMPORTS FOR TENSOR PARALLELISM ---
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.experimental import spmd                         # The key new library
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

# --- TENSOR PARALLELISM SETUP ---
def _mp_fn(index):
    dist.init_process_group('xla')
    device = xm.xla_device()
    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # --- Create a Device Mesh ---
    # We define a 1D mesh of all 8 available devices for sharding.
    num_devices = xm.xla_device_count()
    mesh = spmd.Mesh(torch.arange(num_devices), (num_devices,), mesh_names=('model',))

    # --- Load Model (on CPU first) ---
    # Only the master process loads from disk to prevent race conditions.
    if xm.is_master_ordinal():
        print("Master process loading model and processor...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
    else:
        processor = None
        model = None
    
    # Broadcast the loaded objects to all other processes.
    object_list = [model, processor]
    dist.broadcast_object_list(object_list, src=0)
    model, processor = object_list
    
    # --- THIS IS THE KEY: SHARD THE MODEL FOR TENSOR PARALLELISM ---
    # We manually shard the linear layers across the 'model' dimension of our mesh.
    # This splits the computation itself, solving the vmem issue.
    if xm.is_master_ordinal():
        print("Applying Tensor Parallelism and sharding the model...")
        
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
                # Shard attention-related linear layers
                spmd.shard_module(module, mesh, "model", (("model", None), (None,)))
            elif "gate_proj" in name or "up_proj" in name:
                 # Shard MLP-related linear layers
                spmd.shard_module(module, mesh, "model", (("model", None), (None,)))
            elif "down_proj" in name:
                spmd.shard_module(module, mesh, "model", ((None, "model"), (None,)))

    model.to(device)
    model.eval()

    xm.rendezvous('model_ready')
    if xm.is_master_ordinal():
        print("Model sharded with Tensor Parallelism and ready on all cores.")

    # --- MAIN AGENT LOOP ---
    max_steps = 20
    for step in range(max_steps):
        if xm.is_master_ordinal():
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            print("Taking screenshot...")
            screenshot = pyautogui.screenshot()
        else:
            screenshot = None # Placeholder for non-master processes

        # Broadcast the screenshot to all processes
        object_list = [screenshot]
        dist.broadcast_object_list(object_list, src=0)
        screenshot = object_list[0]
        
        # All processes prepare the inputs
        user_instruction = "Find a folder called ui-tars"
        formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')
        full_conversation = [{"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": formatted_prompt_text}]}]
        inputs = processor.apply_chat_template(full_conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # --- GENERATION WITH TENSOR PARALLELISM ---
        if xm.is_master_ordinal():
            print("Generating response using Tensor Parallelism on all cores...")
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)

        # --- EXECUTION (Main Process Only) ---
        if xm.is_master_ordinal():
            original_width, original_height = screenshot.size
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

        xm.rendezvous('step_complete')

    if xm.is_master_ordinal():
        print("\n--- UI Agent Finished ---")


# --- Launcher for xmp.spawn ---
if __name__ == '__main__':
    xmp.spawn(_mp_fn)

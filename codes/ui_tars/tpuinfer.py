# --- FINAL MODERN IMPORTS ---
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
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
    # This launcher function is run on each of the 8 TPU cores.
    device = xm.xla_device()
    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # --- THIS IS THE MODERN API FOR SHARDING ---
    # 1. Create a Device Mesh
    # This describes our 8 TPU cores as a single parallel group.
    device_mesh = init_device_mesh("xla", (xm.xla_device_count(),))

    # 2. Load Model on CPU
    # We load on the master process to prevent race conditions.
    if xm.is_master_ordinal():
        print("Master process loading model and processor...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
        # Put the model in eval mode before sharding
        model.eval()
    else:
        # We need a placeholder structure on other processes.
        # Transformers' `from_config` is the most efficient way to do this.
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration(config)
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

    # 3. Shard the Model
    # `parallelize_module` is the modern PyTorch function. It automatically
    # figures out how to split the layers across the device mesh.
    if xm.is_master_ordinal():
        print("Applying modern Tensor Parallelism and sharding the model...")

    model = parallelize_module(model, device_mesh)
    model.to(device) # Move the sharded model to the TPU

    # Wait for all processes to finish setup
    xm.rendezvous('model_ready')
    if xm.is_master_ordinal():
        print("Model sharded with modern API and ready on all cores.")

    # --- MAIN AGENT LOOP ---
    max_steps = 20
    for step in range(max_steps):
        # We need a placeholder for the screenshot on non-master nodes
        screenshot = None
        if xm.is_master_ordinal():
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            print("Taking screenshot...")
            screenshot = pyautogui.screenshot()

        # The master process sends the screenshot to all other processes
        # using the xla-native broadcast, which doesn't need init_process_group.
        screenshot_list = [screenshot]
        xm.collective_broadcast(screenshot_list)
        screenshot = screenshot_list[0]
        
        # All processes now have the same screenshot and prepare the inputs
        user_instruction = "Find a folder called ui-tars"
        formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')
        full_conversation = [{"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": formatted_prompt_text}]}]
        inputs = processor.apply_chat_template(full_conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # --- GENERATION WITH TENSOR PARALLELISM ---
        if xm.is_master_ordinal():
            print("Generating response on all cores...")
        
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

        # All processes wait here to stay in sync
        xm.rendezvous('step_complete')

    if xm.is_master_ordinal():
        print("\n--- UI Agent Finished ---")


# --- Launcher for xmp.spawn ---
if __name__ == '__main__':
    # We use the default nprocs=None to let xmp use all available cores.
    xmp.spawn(_mp_fn)

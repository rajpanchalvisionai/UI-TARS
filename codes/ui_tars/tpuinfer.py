# --- FINAL MODERN IMPORTS FOR EXPLICIT TENSOR PARALLELISM ---
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
import torch_xla.runtime as runtime
# --- END IMPORTS ---

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import os
import time
import pyautogui
import sys

# Add parent directory to sys.path to import action_parser and prompt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

def _mp_fn(index):
    # Enable XLA debug logging
    os.environ.setdefault("XLA_DEBUG", "1")
    
    device = xm.xla_device()
    torch.randn(1, device=device)
    xm.mark_step()

    if xm.is_master_ordinal():
        print("All processes have successfully initialized their TPU cores.")

    model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # Create 1D device mesh over all TPU cores
    world_size = runtime.world_size()
    device_mesh = init_device_mesh("xla", (world_size,))
    if xm.is_master_ordinal():
        print(f"Device mesh initialized with world_size = {world_size}")

    # Load model & processor
    if xm.is_master_ordinal():
        print("Master loading model and processor...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        model.eval()
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        with torch.device('meta'):
            model = Qwen2_5_VLForConditionalGeneration(config)
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

    # Synchronize before sharding
    xm.rendezvous('before_shard')
    if xm.is_master_ordinal():
        print("Applying EXPLICIT Tensor Parallelism and sharding the model...")

    model = parallelize_module(
        model,
        device_mesh,
        {
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        },
    )
    xm.mark_step()
    xm.rendezvous('after_shard')

    if xm.is_master_ordinal():
        print("Model sharded with explicit plan and ready on all cores.")

    model.to(device)

    xm.rendezvous('model_ready')

    # Main interaction loop
    max_steps = 20
    for step in range(max_steps):
        screenshot = None
        if xm.is_master_ordinal():
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            screenshot = pyautogui.screenshot()

        screenshot_list = [screenshot]
        xm.collective_broadcast(screenshot_list)
        screenshot = screenshot_list[0]

        user_instruction = "Find a folder called ui-tars"
        formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')
        full_conversation = [{"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": formatted_prompt_text}]}]
        inputs = processor.apply_chat_template(
            full_conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        if xm.is_master_ordinal():
            print("Generating response on all cores...")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        xm.mark_step()  # Ensure TPU synchronizes after generation

        if xm.is_master_ordinal():
            original_width, original_height = screenshot.size
            model_input_height, model_input_width = smart_resize(original_height, original_width)

            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, input_length:]
            raw_model_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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

if __name__ == '__main__':
    xmp.spawn(_mp_fn)

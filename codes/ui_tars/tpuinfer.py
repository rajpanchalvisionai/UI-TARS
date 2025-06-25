import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import time
import pyautogui
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

model_name = "ByteDance-Seed/UI-TARS-1.5-7B"
COORDINATE_PARSING_FACTOR = 1000
NUM_TPU_CORES = 4  # v4-8 = 4 chips, 8 cores; adjust as needed

def run_ui_agent(user_instruction, max_steps=10):
    print(f"\n--- Starting UI Agent ---")
    print(f"Task: {user_instruction}")

    conversation_history = []

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")
        try:
            screenshot = pyautogui.screenshot()
            img = screenshot
            original_width, original_height = img.size
            print(f"Screenshot taken. Original dimensions: {original_width}x{original_height}")
            model_input_height, model_input_width = smart_resize(original_height, original_width)
            print(f"Image will be processed by model at effective dimensions: {model_input_width}x{model_input_height}")

            formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')
            full_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": formatted_prompt_text}
                    ]
                }
            ]

            print("Applying chat template and tokenizing...")
            inputs = processor.apply_chat_template(
                full_conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)

            print(f"Input tokens shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"Pixel values shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")

            print("Generating response...")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                xm.mark_step()

            print("Response generated.")
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, input_length:]
            raw_model_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print("\n--- Raw Model Output ---")
            print(raw_model_output_text)
            print("------------------------")

            print("Parsing action from model output...")
            parsed_actions = parse_action_to_structure_output(
                raw_model_output_text,
                factor=COORDINATE_PARSING_FACTOR,
                origin_resized_height=model_input_height,
                origin_resized_width=model_input_width,
                model_type="qwen25vl"
            )
            print("Action parsed.")

            if not parsed_actions:
                print("No valid action parsed. Stopping.")
                break

            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_actions,
                image_height=original_height,
                image_width=original_width
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

        except Exception as e:
            print(f"An error occurred during step {step + 1}: {e}")
            break

    print("\n--- UI Agent Finished ---")

def _mp_fn(index, user_task, max_steps):
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Process {index}: Acquiring TPU device...")
    device = xm.xla_device()
    print(f"Process {index}: TPU device acquired: {device}")

    print(f"Process {index}: Loading processor and model for TPU...")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    print(f"Process {index}: Model and processor loaded successfully and moved to TPU.")

    # Make processor/model global for run_ui_agent
    global processor
    global model
    global device

    run_ui_agent(user_task, max_steps)

if __name__ == "__main__":
    user_task = "Find a folder called ui-tars"
    max_steps = 20
    xmp.spawn(_mp_fn, args=(user_task, max_steps), nprocs=NUM_TPU_CORES)

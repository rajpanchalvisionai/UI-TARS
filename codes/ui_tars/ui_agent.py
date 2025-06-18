import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
import os
import time
import pyautogui
import sys
import math
import json

# Add parent directory to sys.path to import action_parser and prompt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
from prompt import COMPUTER_USE_DOUBAO

# --- Model and Processor Setup ---
model_name = "ByteDance-Seed/UI-TARS-2B-SFT"

# Load processor and model
try:
    print("Attempting to load processor and vLLM model...")
    # Override preprocessor_config.json
    preprocessor_config_path = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        f"models--{model_name.replace('/', '--')}/snapshots",
        "refs/main/preprocessor_config.json"
    )
    if os.path.exists(preprocessor_config_path):
        with open(preprocessor_config_path, "r") as f:
            config = json.load(f)
        config["size"] = {"shortest_edge": 224, "longest_edge": 896}
        with open(preprocessor_config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Updated preprocessor_config.json with size settings.")

    processor = AutoProcessor.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        size={"shortest_edge": 224, "longest_edge": 896}
    )
    
    # Initialize vLLM
    llm = LLM(
        model=model_name,
        quantization="bitsandbytes",
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        dtype="float16",
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True
    )
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("-" * 50)
    print("TROUBLESHOOTING LOADING:")
    print("Ensure vLLM 0.9.1 and transformers 4.51.1 are installed.")
    print("Verify CUDA 12.8 with `nvcc --version` and `torch.cuda.is_available()`.")
    print("Check VRAM with `nvidia-smi` and reduce `gpu_memory_utilization`.")
    print("If error persists, try vLLM 0.6.2 or Transformers fallback.")
    print("-" * 50)
    exit()

# --- Configuration ---
COORDINATE_PARSING_FACTOR = 1000

# --- Main Agent Loop ---
def run_ui_agent(user_instruction, max_steps=10):
    """
    Runs the UI agent for a given instruction.

    Args:
        user_instruction (str): The task the agent needs to perform.
        max_steps (int): Maximum number of steps to execute.
    """
    print(f"\n--- Starting UI Agent ---")
    print(f"Task: {user_instruction}")

    conversation_history = []

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")

        try:
            # 1. Take Screenshot
            print("Taking screenshot...")
            screenshot = pyautogui.screenshot()
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

            # Get original dimensions
            original_width, original_height = img.size
            print(f"Screenshot taken. Original dimensions: {original_width}x{original_height}")

            # Calculate dimensions the model *would have seen* after smart resizing
            model_input_height, model_input_width = smart_resize(original_height, original_width)
            print(f"Image will be processed by model at effective dimensions: {model_input_width}x{model_input_height}")

            # 2. Format Prompt
            formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')

            # 3. Prepare Conversation Structure
            full_conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image"
                        },
                        {
                            "type": "text",
                            "text": formatted_prompt_text
                        }
                    ]
                }
            ]

            # 4. Process Input
            print("Processing image and text...")
            processed_inputs = processor(
                text=formatted_prompt_text,
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Debug: Check processed inputs
            if 'pixel_values' in processed_inputs:
                print(f"Pixel values shape: {processed_inputs['pixel_values'].shape}, dtype: {processed_inputs['pixel_values'].dtype}")
            if 'input_ids' in processed_inputs:
                print(f"Input tokens shape: {processed_inputs['input_ids'].shape}")

            # 5. Prepare Prompt for vLLM
            prompt = processor.apply_chat_template(
                full_conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            prompt = prompt.replace("<|image_pad|>", "<|image|>")

            # Debug: Check final prompt and chat template
            print(f"Chat template: {processor.tokenizer.chat_template}")
            print(f"Final prompt: {prompt}")

            # 6. Generate Response with vLLM
            print("Generating response with vLLM...")
            sampling_params = SamplingParams(
                temperature=0.2,
                max_tokens=500,
                top_p=0.95,
                stop=[processor.tokenizer.eos_token]
            )
            outputs = llm.generate([prompt], sampling_params)
            print("Response generated.")

            # Extract generated text
            raw_model_output_text = outputs[0].outputs[0].text.strip()
            print("\n--- Raw Model Output ---")
            print(raw_model_output_text)
            print("------------------------")

            # 7. Parse Action from Output
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

            # 8. Convert to PyAutoGUI code & Execute
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

# --- How to Run ---
if __name__ == "__main__":
    user_task = "Open firefox browser"
    run_ui_agent(user_task, max_steps=10)
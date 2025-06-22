import torch
import torch_xla.core.xla_model as xm  # <-- 1. Import torch_xla for TPU
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

# --- Model and Processor Setup ---
model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

# --- 2. GET TPU DEVICE AND LOAD MODEL ---
# Get the TPU device
try:
    print("Acquiring TPU device...")
    device = xm.xla_device()
    print(f"TPU device acquired: {device}")
except Exception as e:
    print(f"Error acquiring TPU device: {e}")
    print("Ensure you are running on a Google Cloud TPU VM and torch_xla is installed.")
    exit()


# Load processor and model for TPU
try:
    print("Attempting to load processor and model for TPU...")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)  # Match saved model's processor

    # Load the model WITHOUT any GPU-specific configurations.
    # TPUs work best with bfloat16 precision.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better TPU performance
    )
    
    # Explicitly move the model to the TPU device
    print("Moving model to TPU device...")
    model.to(device)

    model.eval()  # Set model to evaluation mode
    print("Model and processor loaded successfully and moved to TPU.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("-" * 50)
    print("TROUBLESHOOTING TPU LOADING:")
    print("Ensure you have installed torch and torch_xla correctly.")
    print("Installation command: pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html")
    print("Check your internet connection for downloading model weights.")
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
            img = screenshot

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
                            "type": "image",
                            "image": img
                        },
                        {
                            "type": "text",
                            "text": formatted_prompt_text
                        }
                    ]
                }
            ]

            # 4. Process Input & Generate
            print("Applying chat template and tokenizing...")
            inputs = processor.apply_chat_template(
                full_conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)  # <-- IMPORTANT: Move input tensors to the TPU device

            # Debug: Check input tokens
            print(f"Input tokens shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"Pixel values shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")

            print("Generating response...")
            with torch.no_grad():
                # Use xm.mark_step() for better performance, tells XLA to execute the graph
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                xm.mark_step()
                
            print("Response generated.")

            # Decode the generated output
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, input_length:]
            raw_model_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print("\n--- Raw Model Output ---")
            print(raw_model_output_text)
            print("------------------------")

            # 5. Parse Action from Output
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

            # 6. Convert to PyAutoGUI code & Execute
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
    user_task = "Find a folder called ui-tars"
    run_ui_agent(user_task, max_steps=20)

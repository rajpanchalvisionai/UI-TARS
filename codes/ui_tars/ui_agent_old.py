import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
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
# model_name = "ByteDance-Seed/UI-TARS-2B-SFT"

model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

# Load processor and model
try:
    print("Attempting to load processor and model...")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)  # Match saved model's processor
    
    # Define quantization configuration for 8-bit quantization with CPU offloading
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    # Load model with quantization and automatic device mapping
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    
    model.eval()  # Set model to evaluation mode
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("-" * 50)
    print("TROUBLESHOOTING LOADING:")
    print("Ensure you have transformers 4.49.0 installed (`pip install transformers==4.49.0`).")
    print("Ensure you have the latest bitsandbytes installed (`pip install -U bitsandbytes`).")
    print("Check your internet connection for downloading model weights.")
    print("If on a low-memory system, quantization and CPU offloading are enabled.")
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
            ).to(model.device)

            # Debug: Check input tokens
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
    user_task = "I'm on ubuntu. Open browser and search for 'doge'."
    run_ui_agent(user_task, max_steps=20)


# ui_agent.py

# import torch
# # use transformers 4.49.0
# from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
# from PIL import Image
# import requests
# from io import BytesIO
# import os
# import time
# import pyautogui
# import sys
# import math # Import math for smart_resize

# # Add parent directory to sys.path to import action_parser and prompt
# # Adjust path if necessary based on your file structure
# # Assuming action_parser.py and prompt.py are in the same directory as ui_agent.py
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize
# from prompt import COMPUTER_USE_DOUBAO # Use the computer prompt template

# # --- Model and Processor Setup ---
# # model_name = "ByteDance-Seed/UI-TARS-2B-SFT"
# model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

# # Load processor and model
# try:
#     print("Attempting to load processor and model...")
#     processor = AutoProcessor.from_pretrained(model_name)
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         model_name,
#         device_map='cuda',            # Use None for CPU
#         torch_dtype=torch.float16,  # float32 is safest for CPU
#         # low_cpu_mem_usage=True # Might help with memory, but can sometimes cause issues
#         attn_implementation="flash_attention_2"
#     )
#     # model = model.cpu()            # Ensure model is on CPU
#     model.eval() # Set model to evaluation mode
#     print("Model and processor loaded successfully.")
# except Exception as e:
#     print(f"Error loading model or processor: {e}")
#     print("-" * 50)
#     print("TROUBLESHOOTING LOADING:")
#     print("Ensure you have transformers 4.49.0 installed (`pip install transformers==4.49.0`).")
#     print("Check your internet connection for downloading model weights.")
#     print("If on a low-memory system, consider reducing batch size or using quantization (though not enabled here).")
#     print("-" * 50)
#     exit()

# # --- Configuration ---
# # This factor is used internally by the model's image processing and parsing
# # The default in action_parser is 28, but the model might implicitly work with 1000?
# # Let's stick to 1000 for parsing as seen in original tests, but check action_parser details.
# # The action_parser uses IMAGE_FACTOR=28 for smart_resize, and 1000 for scaling coordinates in parse_action_to_structure_output.
# # Let's use 1000 as the scaling factor passed to the parser.
# # Update: Looking at action_parser, it uses IMAGE_FACTOR=28 for smart_resize, but *divides* parsed coords by smart_resize_width/height for qwen25vl.
# # Then parsing_response_to_pyautogui_code multiplies by *original* width/height.
# # So, parse_action_to_structure_output needs the smart_resized dimensions the model processed.
# # Let's pass the output of smart_resize to parse_action_to_structure_output.
# # The 'factor' parameter in parse_action_to_structure_output seems unused for qwen25vl type.
# # Let's pass smart_resize results directly.
# COORDINATE_PARSING_FACTOR = 1000 # This factor seems to be related to how coordinates were normalized, often 1000x1000 grid.

# # --- Main Agent Loop ---
# def run_ui_agent(user_instruction, max_steps=10):
#     """
#     Runs the UI agent for a given instruction.

#     Args:
#         user_instruction (str): The task the agent needs to perform.
#         max_steps (int): Maximum number of steps to execute.
#     """
#     print(f"\n--- Starting UI Agent ---")
#     print(f"Task: {user_instruction}")

#     # Keep track of conversation history (basic)
#     # For simplicity in this demo, we'll just include the last turn or keep it single-turn
#     # A real agent would build this over time.
#     # Let's start with a single turn: Initial prompt + Image -> Generate Action
#     # Subsequent turns would involve adding the previous step's thought/action/observation.
#     # This example will be single-action per run for simplicity, but structured for potential multi-turn.

#     conversation_history = []

#     for step in range(max_steps):
#         print(f"\n--- Step {step + 1}/{max_steps} ---")

#         try:
#             # 1. Take Screenshot
#             print("Taking screenshot...")
#             screenshot = pyautogui.screenshot()
#             img = screenshot # pyautogui.screenshot returns PIL Image directly

#             # Get original dimensions
#             original_width, original_height = img.size
#             print(f"Screenshot taken. Original dimensions: {original_width}x{original_height}")

#             # Calculate dimensions the model *would have seen* after smart resizing
#             # This is needed for the parser to correctly interpret coordinates
#             # action_parser.smart_resize uses IMAGE_FACTOR=28, MIN/MAX_PIXELS from its own constants
#             model_input_height, model_input_width = smart_resize(original_height, original_width)
#             print(f"Image will be processed by model at effective dimensions: {model_input_width}x{model_input_height}")


#             # 2. Format Prompt
#             # Use the computer prompt template and fill in the instruction
#             formatted_prompt_text = COMPUTER_USE_DOUBAO.format(instruction=user_instruction, language='English')

#             # For the first turn, the user's content includes the image and the full prompt.
#             # In subsequent turns, the conversation history would be added,
#             # and the current user content would be just the image (as the model generates the next action based on image + history).
#             # However, UI-TARS prompt template includes the instruction each turn, so let's stick to that.
#             current_user_content = [
#                 {
#                     "type": "image",
#                     "image": img # Pass the PIL Image object directly
#                 },
#                 {
#                     "type": "text",
#                     "text": formatted_prompt_text
#                 }
#             ]

#             # Add current turn to conversation history structure (simplified for this demo)
#             # This structure is mainly for apply_chat_template
#             conversation_turn = {"role": "user", "content": current_user_content}

#             # If we had model responses to include, they would be added here:
#             # if step > 0 and last_model_response:
#             #    conversation_history.append({"role": "assistant", "content": last_model_response}) # Need to store raw text response

#             # Build the final conversation structure for the template
#             # For this simple demo, let's just use the current turn structure with the initial prompt repeated (as per template style)
#             # A more complex agent would manage history carefully.
#             # Let's just pass the single turn for now, as the template includes the full instruction anyway.
#             # Correct approach based on typical chat templates: history + current turn.
#             # Let's simulate history by adding the current turn's *prompt* text to the history structure *before* applying template, but *after* taking screenshot.
#             # This part is tricky as template application varies. Qwen-VL might expect image first, then text prompt including instruction.
#             # Let's revert to the structure from uitarsimageinfer.py example, but put the structured prompt in the text part.

#             # Correct structure for apply_chat_template with image+text
#             # [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Prompt text..."}]}]
#             # If history is needed, previous turns go before this.
#             # Example with simple history:
#             # history = [
#             #    {"role": "user", "content": [{"type": "image", "image": img_prev}, {"type": "text", "text": "Previous prompt..."}]},
#             #    {"role": "assistant", "content": "Previous model response text"}
#             # ]
#             # current_turn = {"role": "user", "content": [{"type": "image", "image": current_img}, {"type": "text", "text": "Current prompt..."}]}
#             # full_conversation = history + [current_turn]

#             # For this demo, let's just use the current turn with the latest screenshot and the full structured prompt text.
#             full_conversation = [
#                  {
#                      "role": "user",
#                      "content": [
#                          {
#                              "type": "image",
#                              "image": img # Pass the PIL Image object directly
#                          },
#                          {
#                              "type": "text",
#                              "text": formatted_prompt_text # Use the structured prompt
#                          }
#                      ]
#                  }
#             ]


#             # 3. Process Input & Generate
#             print("Applying chat template and tokenizing...")
#             inputs = processor.apply_chat_template(
#                 full_conversation,
#                 add_generation_prompt=True,
#                 tokenize=True,
#                 return_dict=True,
#                 return_tensors="pt"
#             ).to(model.device) # model is on CPU, so this goes to CPU

#             print("Generating response...")
#             with torch.no_grad():
#                 output_ids = model.generate(
#                     **inputs,
#                     max_new_tokens=500, # Allow for longer responses
#                     do_sample=False,    # Use greedy decoding
#                     pad_token_id=processor.tokenizer.eos_token_id
#                 )
#             print("Response generated.")

#             # Decode the generated output
#             input_length = inputs.input_ids[0].shape[0]
#             generated_ids = [output_ids[0, input_length:]]
#             raw_model_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
#             print("\n--- Raw Model Output ---")
#             print(raw_model_output_text)
#             print("------------------------")

#             # 4. Parse Action from Output
#             print("Parsing action from model output...")
#             # The parse_action_to_structure_output needs the dimensions the model processed the image at
#             # (which parse_action_to_structure_output calculates using smart_resize internally,
#             # but the original_resized_height/width arguments seem intended for this).
#             # Let's pass the smart_resized dimensions we calculated earlier.
#             # The 'factor' argument (1000) seems less critical for qwen25vl according to action_parser code comments, but let's include it.
#             parsed_actions = parse_action_to_structure_output(
#                 raw_model_output_text,
#                 factor=COORDINATE_PARSING_FACTOR, # This factor seems unused for qwen25vl in parser logic
#                 origin_resized_height=model_input_height, # Pass the dimensions model saw
#                 origin_resized_width=model_input_width,   # Pass the dimensions model saw
#                 model_type="qwen25vl" # Specify model type for parser logic
#             )
#             print("Action parsed.")
#             # print("Parsed Actions:", parsed_actions) # Uncomment for detailed debug

#             if not parsed_actions:
#                 print("No valid action parsed. Stopping.")
#                 break

#             # 5. Convert to PyAutoGUI code & Execute
#             # The parsing_response_to_pyautogui_code needs the *original* image dimensions
#             # to scale the 0-1 relative coordinates (output by parse_action_to_structure_output)
#             # back to pixel coordinates for pyautogui.
#             pyautogui_code = parsing_response_to_pyautogui_code(
#                 parsed_actions,
#                 image_height=original_height, # Pass original screenshot height
#                 image_width=original_width    # Pass original screenshot width
#             )

#             print("\n--- Generated PyAutoGUI Code ---")
#             print(pyautogui_code)
#             print("------------------------------")

#             if pyautogui_code.strip() == "DONE":
#                 print("Task finished by agent.")
#                 break

#             print("Executing PyAutoGUI code...")
#             try:
#                 # Be careful with exec! Only execute trusted code.
#                 # In this case, it's code we generated based on model output and a parser.
#                 # Ensure parser is robust against generating malicious code.
#                 exec(pyautogui_code)
#                 print("Code executed. Waiting a moment...")
#                 time.sleep(2) # Wait for the action to potentially take effect
#             except Exception as e:
#                 print(f"Error executing PyAutoGUI code: {e}")
#                 # Decide how to handle execution errors - maybe break, maybe log and continue
#                 break # Stop on execution error for this demo

#             # Check if the last action was 'finished' - already handled by checking pyautogui_code == "DONE"

#         except Exception as e:
#             print(f"An error occurred during step {step + 1}: {e}")
#             # Decide how to handle other errors - maybe retry, maybe break
#             break # Stop on general error for this demo

#     print("\n--- UI Agent Finished ---")

# # --- How to Run ---
# if __name__ == "__main__":
#     # Define your user instruction here
#     # Example: "Open the browser and navigate to www.google.com"
#     # Example: "Click the search bar and type 'how to use ui tars'"
#     # user_task = "Move the mouse to the center of the screen and click." # Simple task for initial test
#     user_task = "Open firefox browser" # Simple task for initial test
#     # user_task = "Open the Firefox browser." # More complex task, might require finding the icon
#     # user_task = "Find the search bar on the current page and type 'hello world'." # Requires screenshot understanding

#     run_ui_agent(user_task, max_steps=10) # Run for a few steps
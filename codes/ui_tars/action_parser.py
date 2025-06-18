# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import re
import ast
import math

# Constants for image resizing logic
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 16384 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200


def convert_point_to_coordinates(text, is_answer=False):
    # This function seems historical or for specific standalone use cases.
    # The primary coordinate parsing for actions happens in parse_action_to_structure_output.
    # Keeping it as is but note its potential limited use in the current agent flow.

    # Pattern for integer coordinates like <point>200 300</point>
    pattern_int = r"<point>(\d+)\s+(\d+)</point>"
    text = re.sub(pattern_int, r"(\1,\2)", text) # Replace with tuple string (x,y)

    # Pattern for float coordinates like <point>X: 0.5 Y: 0.5</point>
    pattern_float = r"<point>X: (\d+\.?\d*) Y: (\d+\.?\d*)</point>"
    text = re.sub(pattern_float, r"(\1,\2)", text) # Replace with tuple string (x,y)

    # Remove [EOS] marker
    text = re.sub(r"\[EOS\]", "", text)

    # Additional: Handle raw (N,N) or [N,N] strings if they appear outside <point>
    # This might not be the intended use of this function.
    # Let's keep its scope limited to the <point> transformations it was designed for.


    return text.strip()


def parse_action(action_str):
    """
    Parses a single action string (e.g., "click(point='...')") into a structure.
    Uses ast.parse in eval mode with a tuple wrapper to handle the string as a call expression.
    Safely extracts parameter values from literal nodes using ast.literal_eval.
    """
    action_str = action_str.strip()
    if not action_str:
        return None # Handle empty strings

    # Wrap the action string in parentheses and a comma to make it a tuple containing the expression.
    # This allows ast.parse(mode='eval') to parse it as an expression tuple.
    # E.g., "click(start='(986,12)')" becomes " (click(start='(986,12)'),) "
    wrapped_action_str = f"({action_str},)"

    try:
        # Parse the wrapped string as an expression
        node = ast.parse(wrapped_action_str, mode='eval')

        # Ensure the node is an Expression containing a Tuple with exactly one element
        if not isinstance(node, ast.Expression) or not isinstance(node.body, ast.Tuple) or len(node.body.elts) != 1:
             raise ValueError("Expression structure incorrect (expected tuple of one element)")

        # The single element in the tuple should be the function call
        call = node.body.elts[0]

        # Ensure the element is a function call
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call expression inside the tuple")

        # Get function name
        func_name = None
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # Handles methods like obj.method, gets 'method'. Assumes simple function calls.
            func_name = call.func.attr
        # else: func_name remains None if target is more complex (unlikely for UI-TARS actions)

        if func_name is None:
             raise ValueError(f"Could not determine function name from call target type {type(call.func).__name__}")


        # Get keyword arguments
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if key is None:
                 print(f"Warning: Skipping positional argument found for function '{func_name}' in action '{action_str}'")
                 continue # UI-TARS actions primarily use keywords

            # Safely evaluate the value node using ast.literal_eval
            # This extracts the actual Python literal value.
            # E.g., for start='(986,12)', this will result in the *string* '(986,12)'
            # For hotkey='ctrl v', this will result in the *string* 'ctrl v'
            # For content='abc', this will result in the *string* 'abc'
            # For duration=0.5, this will result in the *float* 0.5
            # For button='left', this will result in the *string* 'left'
            try:
                # ast.literal_eval handles strings, numbers, tuples, lists, dicts, booleans, None
                value = ast.literal_eval(kw.value)
                kwargs[key] = value
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Warning: Failed to safely evaluate argument '{key}' node type {type(kw.value).__name__} in action '{action_str}': {e}")
                # If safe evaluation fails, try to get the source code representation of the node
                try:
                    kwargs[key] = ast.get_source_segment(action_str, kw.value)
                    print(f"Attempted to get source segment for '{key}': {kwargs[key]}")
                except Exception:
                     kwargs[key] = None # Fallback to None

        # UI-TARS actions primarily use keyword arguments. Positional arguments are ignored by this parser logic.

        return {'function': func_name, 'args': kwargs}

    except Exception as e:
        # Print the action string that caused the failure for easier debugging
        print(f"Failed to parse action string '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    """Escapes single quotes within a string for safe use in Python string literals."""
    # Replace a single quote ' with \'
    # Use replace directly, it's simpler and safer than complex regex for this case.
    return text.replace("'", "\\'")


def round_by_factor(number: float, factor: int) -> float:
    """Returns the closest number to 'number' that is divisible by 'factor'."""
    # Allow float input, return float.
    if factor == 0: return number
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> float:
    """Returns the smallest number >= 'number' that is divisible by 'factor'."""
    if factor == 0: return number
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> float:
    """Returns the largest number <= 'number' that is divisible by 'factor'."""
    if factor == 0: return number
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image dimensions to be divisible by 'factor', within pixel limits,
    while maintaining/limiting aspect ratio. Returns (height, width) as integers.
    This calculates the expected input dimensions of the model's vision encoder.
    """
    if min(height, width) <= 0:
         # print(f"Warning: Got non-positive dimensions ({width}x{height}) for smart_resize. Returning minimal valid size.")
         return max(factor, 1), max(factor, 1)

    # Calculate current aspect ratio
    current_ratio = max(height, width) / min(height, width)

    # Limit aspect ratio first if it exceeds MAX_RATIO
    if current_ratio > MAX_RATIO:
        # print(f"Warning: Aspect ratio {current_ratio:.2f} exceeds limit {MAX_RATIO}. Adjusting dimensions.")
        if height > width:
             height = int(width * MAX_RATIO)
        else:
             width = int(height * MAX_RATIO)
        # Recalculate ratio after adjustment (for potential logging/debug)
        current_ratio = max(height, width) / min(height, width)
        # print(f"Adjusted dimensions for ratio limit: {width}x{height} (ratio: {current_ratio:.2f})")


    # Calculate target dimensions rounded by factor
    h_bar_initial = max(factor, round_by_factor(height, factor))
    w_bar_initial = max(factor, round_by_factor(width, factor))

    current_pixels = h_bar_initial * w_bar_initial

    # Adjust based on pixel limits while attempting to preserve aspect ratio from *current* dims
    # Using initial calculated dims (h_bar_initial, w_bar_initial) for the scaling beta calculation
    # might be closer to how model might internally handle it after initial rounding.
    initial_calculated_pixels = h_bar_initial * w_bar_initial

    h_bar, w_bar = h_bar_initial, w_bar_initial # Start with the factor-rounded dimensions

    if initial_calculated_pixels > max_pixels:
        # print(f"Warning: Pixels {initial_calculated_pixels} exceed max limit {max_pixels}. Scaling down.")
        # Scale factor based on the rounded dimensions
        scale_factor = math.sqrt(max_pixels / initial_calculated_pixels)
        # Apply factor and floor, then round by factor again
        h_bar = floor_by_factor(h_bar * scale_factor, factor)
        w_bar = floor_by_factor(w_bar * scale_factor, factor)
        # Ensure they are still at least 'factor' after flooring/rounding
        h_bar = max(factor, h_bar)
        w_bar = max(factor, w_bar)
        # print(f"Scaled down dims: {w_bar}x{h_bar}")

    elif initial_calculated_pixels < min_pixels:
        # print(f"Warning: Pixels {initial_calculated_pixels} are below min limit {min_pixels}. Scaling up.")
         # Scale factor based on the rounded dimensions
        scale_factor = math.sqrt(min_pixels / initial_calculated_pixels)
        # Apply factor and ceil, then round by factor again
        h_bar = ceil_by_factor(h_bar * scale_factor, factor)
        w_bar = ceil_by_factor(w_bar * scale_factor, factor)
        # Ensure they are still at least 'factor' after ceiling/rounding
        h_bar = max(factor, h_bar)
        w_bar = max(factor, w_bar)
        # print(f"Scaled up dims: {w_bar}x{h_bar}")

    # Final rounding by factor and check minimum size just to be absolutely sure
    h_bar = max(factor, round_by_factor(h_bar, factor))
    w_bar = max(factor, round_by_factor(w_bar, factor))


    return h_bar, w_bar


def parse_action_to_structure_output(text,
                                     factor, # Appears unused for model_type="qwen25vl" coord scaling
                                     origin_resized_height, # Height of image as model likely processed it (from smart_resize)
                                     origin_resized_width,  # Width of image as model likely processed it (from smart_resize)
                                     model_type="qwen25vl", # Model type affects coordinate interpretation
                                     max_pixels=None, # Not directly used here, passed to smart_resize if needed
                                     min_pixels=None):# Not directly used here, passed to smart_resize if needed
    """
    Parses the raw model output text into a structured list of action dictionaries.
    Extracts Thought/Reflection/Summary, parses action calls, and normalizes
    coordinate parameters by converting model-space coordinates (pixels) to 0-1 relative.
    """
    text = text.strip()
    if not text:
        return [] # Return empty list if text is empty

    # --- Handle Thought/Reflection/Action_Summary ---
    # Initialize variables to None at the start
    reflection, thought, action_summary = None, None, None # <-- FIX: Initialize action_summary

    # Find the "Action: " prefix to split the text. Everything before is considered thought section.
    action_match = re.search(r"Action: (.*)", text, re.DOTALL)

    action_part = ""
    thought_section = text # Assume whole text is thought section if no "Action:" prefix

    if action_match:
         action_part = action_match.group(1).strip()
         thought_section = text[:action_match.start()].strip()
    else:
        # If "Action: " is not found, maybe the model only output thought/reflection?
        # Or the format is unexpected. Log and return empty action list.
        print(f"Warning: Could not find 'Action: ' prefix in model output. No action will be parsed.")
        print(f"Output was:\n{text}")
        return [] # Return empty list if no action section


    # Parse Thought/Reflection/Summary from the thought section
    # Using temporary copies to parse sequentially without modifying the original section string
    current_thought_section = thought_section

    reflection_match = re.search(r"Reflection: (.+?)(?=Action_Summary: |Thought: |$)", current_thought_section, re.DOTALL)
    if reflection_match:
        reflection = reflection_match.group(1).strip()
        # Remove matched reflection from the current processing string
        current_thought_section = re.sub(r"Reflection: (.+?)", "", current_thought_section, count=1, flags=re.DOTALL).strip()

    action_summary_match = re.search(r"Action_Summary: (.+?)(?=Thought: |$)", current_thought_section, re.DOTALL)
    if action_summary_match:
        action_summary = action_summary_match.group(1).strip()
        # Remove matched summary
        current_thought_section = re.sub(r"Action_Summary: (.+?)", "", current_thought_section, count=1, flags=re.DOTALL).strip()

    thought_match = re.search(r"Thought: (.*)", current_thought_section, re.DOTALL) # Capture everything after "Thought: " in the remainder
    if thought_match:
         thought = thought_match.group(1).strip()
    # Else, if no "Thought: " prefix found but there's remaining text in current_thought_section,
    # that remaining text *is* the thought.
    elif current_thought_section:
         # If the thought_section contained text but no "Thought:" prefix was found,
         # treat the remaining text as the thought. This handles cases where
         # the model just puts thought text without the explicit "Thought: " label
         # after other sections, or if the pattern misses it.
         thought = current_thought_section.strip()


    # Combine Action_Summary into Thought if it exists
    if action_summary is not None: # This check is now safe because action_summary is initialized
         if thought is None:
             thought = f"Action Summary: {action_summary}"
         elif not thought.strip().startswith("Action Summary:"): # Avoid double prefix if thought already included it
             thought = f"Action Summary: {action_summary}\n{thought}"


    # --- Parse Action String(s) ---
    # Split multiple actions separated by ")\n\n". This is a heuristic based on observed output format.
    # This might break if a parameter value contains ")\n\n" even if escaped.
    # A more robust approach would be stateful parsing or a grammar.
    raw_action_strings = action_part.split(")\n\n")

    # Ensure each split part ends with ')' if it was removed by the split, then trim whitespace.
    # Filter out any empty strings resulting from the split.
    action_strings = [s.strip() + ')' if s.strip() and not s.strip().endswith(')') else s.strip() for s in raw_action_strings]
    action_strings = [s for s in action_strings if s]

    parsed_actions_list = [parse_action(action_str) for action_str in action_strings]

    actions = []
    for action_instance, raw_action_str in zip(parsed_actions_list, action_strings):
        # Include thought and reflection with *each* parsed action for downstream use
        # Although typically thought/reflection applies to the *block* of actions,
        # attaching it to each makes the data structure simpler if actions are processed independently later.
        # The parsing_response_to_pyautogui_code function currently only uses the first item's thought/reflection.
        action_data = {
            "reflection": reflection,
            "thought": thought,
            "action_type": None, # Will fill this in
            "action_inputs": {}, # Will fill this in
            # "raw_action_string": raw_action_str # Optional: Keep the raw action string
        }


        if action_instance is None:
            print(f"Skipping unparseable action string: {raw_action_str}")
            # Decide if you want to skip or raise an error. Raising is safer during development.
            # raise ValueError(f"Failed to parse action string: {raw_action_str}")
            continue # Skip this action

        action_data["action_type"] = action_instance.get("function")
        params = action_instance.get("args", {})

        action_inputs = {}
        for param_name, param_value in params.items():
            # param_value here is the result of ast.literal_eval from parse_action
            # It could be a string, number, tuple, list, boolean, None.

            # Skip parameters with None value (e.g., if ast.literal_eval failed in parse_action)
            if param_value is None:
                 print(f"Warning: Skipping None parameter '{param_name}' in action '{raw_action_str}'")
                 continue
            # Skip empty string values for most parameters, but not 'content'
            if param_value == "" and param_name.lower() != "content":
                 continue

            # --- Coordinate Parameter Handling ---
            # Check if this parameter name suggests it contains coordinates
            # Normalize parameter names for consistent downstream processing
            is_coordinate_param = False
            normalized_coord_key = None # Key to use in action_inputs (e.g., 'start_point')

            param_name_lower = param_name.strip().lower()
            if "box" in param_name_lower:
                 is_coordinate_param = True
                 # Use a standard box key, preserving start/end if present
                 normalized_coord_key = 'start_box' if 'start' in param_name_lower else ('end_box' if 'end' in param_name_lower else 'box')
            elif "point" in param_name_lower:
                 is_coordinate_param = True
                 # Use a standard point key, preserving start/end if present
                 normalized_coord_key = 'start_point' if 'start' in param_name_lower else ('end_point' if 'end' in param_name_lower else 'point')
                 if normalized_coord_key == 'point': # Map generic 'point' to 'start_point' by default
                      normalized_coord_key = 'start_point'
            elif param_name_lower == 'start': # Map 'start' to 'start_point'
                 is_coordinate_param = True
                 normalized_coord_key = 'start_point'
            elif param_name_lower == 'end': # Map 'end' to 'end_point'
                 is_coordinate_param = True
                 normalized_coord_key = 'end_point'

            if is_coordinate_param:
                coord_input_value = param_value # Value from parse_action (string, tuple, list, etc.)

                # Attempt to convert coord_input_value into a list of floats [x,y] or [x1,y1,x2,y2]
                # and scale from model pixel space to 0-1 relative if they seem to be in model pixel space.
                scaled_numbers = None
                numbers = [] # List to hold extracted float numbers

                try:
                    if isinstance(coord_input_value, str):
                        # If it's a string, it might be '<point>...</point>' or '(N,N)' etc.
                        coord_string = coord_input_value.strip()

                        # 1. Try regex for <point>X: F Y: F</point> format
                        point_float_match = re.search(r"<point>\s*X:\s*(\d+\.?\d*)\s*Y:\s*(\d+\.?\d*)\s*</point>", coord_string)
                        if point_float_match:
                            numbers = [float(point_float_match.group(1)), float(point_float_match.group(2))]
                        else:
                            # 2. Try regex for <point>N N</point> format
                            point_int_match = re.search(r"<point>\s*(\d+)\s+(\d+)\s*</point>", coord_string)
                            if point_int_match:
                                numbers = [float(point_int_match.group(1)), float(point_int_match.group(2))]
                            else:
                                # 3. Try ast.literal_eval for formats like '(N,N)', '[N,N]', '(N,N,N,N)', etc. within the string
                                try:
                                    # Use ast.literal_eval on the string content
                                    evaled_string_data = ast.literal_eval(coord_string)
                                    if isinstance(evaled_string_data, (tuple, list)):
                                        # Ensure all elements are numbers
                                        numbers = [float(n) for n in evaled_string_data if isinstance(n, (int, float))]
                                        if len(numbers) != len(evaled_string_data):
                                             print(f"Warning: Some elements in evaluated coordinate list were not numbers: {evaled_string_data}")
                                             numbers = [] # Clear if not all numbers
                                    elif isinstance(evaled_string_data, (int, float)):
                                         # If it evaluates to a single number (unlikely for coords, but maybe a typo?)
                                         numbers = [float(evaled_string_data)]
                                    else:
                                         raise ValueError(f"Evaluated string is not a list/tuple/number: {type(evaled_string_data).__name__}")
                                except (ValueError, SyntaxError, TypeError) as eval_e:
                                     # Literal evaluation failed, assume it's not a parsable coordinate string literal
                                     # print(f"Debug: ast.literal_eval failed on coordinate string '{coord_string}': {eval_e}") # Debugging
                                     numbers = [] # Indicate parsing failed


                    elif isinstance(coord_input_value, (tuple, list)):
                         # If it's already a tuple or list (from parse_action's literal_eval)
                         # Ensure all elements are numbers
                         numbers = [float(n) for n in coord_input_value if isinstance(n, (int, float))]
                         if len(numbers) != len(coord_input_value):
                              print(f"Warning: Some elements in coordinate list were not numbers: {coord_input_value}")
                              numbers = [] # Clear if not all numbers

                    elif isinstance(coord_input_value, (int, float)):
                         # If it's a raw number (less likely for coords)
                         numbers = [float(coord_input_value)]
                         print(f"Warning: Raw number ({coord_input_value}) treated as single coordinate.")
                         # This case needs clarification on how a single number should be interpreted as a coordinate.

                    else:
                        raise ValueError(f"Coordinate value is unexpected primary type: {type(coord_input_value).__name__}")

                    # --- Scaling Logic (from Model Pixel space to 0-1 Relative) ---
                    # Assume numbers are pixels relative to the model's input image dimensions.
                    # Scale these pixels to 0-1 relative coordinates.
                    # Need origin_resized_width/height from smart_resize output
                    scaled_numbers = []
                    if model_type == "qwen25vl" and origin_resized_width > 0 and origin_resized_height > 0:
                        if len(numbers) == 2: # (x, y) point in model pixel space
                             x, y = numbers
                             # Scale to 0-1 relative
                             scaled_numbers = [x / origin_resized_width, y / origin_resized_height]
                        elif len(numbers) == 4: # (x1, y1, x2, y2) box in model pixel space
                             x1, y1, x2, y2 = numbers
                             # Scale to 0-1 relative
                             scaled_numbers = [x1 / origin_resized_width, y1 / origin_resized_height,
                                               x2 / origin_resized_width, y2 / origin_resized_height]
                        else:
                             # print(f"Warning: Coordinate numbers list has unexpected length ({len(numbers)}) after parsing for scaling.")
                             scaled_numbers = None # Indicate failure
                    elif len(numbers) in [2, 4]:
                         # If model_type is not qwen25vl or dims are invalid, but we got 2 or 4 numbers.
                         # What should these numbers represent? Pixels? 0-1 relative already?
                         # The original code divided by 'factor' for non-qwen25vl, which was 1000.
                         # If the model outputs 0-1 relative directly, no scaling is needed here.
                         # If it outputs 0-1000, divide by 1000.
                         # Let's assume for non-qwen25vl (or if dims invalid) that the numbers are already 0-1 relative or model-specific.
                         # For safety, let's just use them as is if scaling logic isn't clear/applicable.
                         # print(f"Warning: Skipping scaling for model type '{model_type}' or invalid dimensions ({origin_resized_width}x{origin_resized_height}). Using numbers as is.")
                         scaled_numbers = numbers # Use raw numbers

                    # If scaling resulted in a valid list of 2 or 4 numbers
                    if scaled_numbers is not None and len(scaled_numbers) in [2, 4]:
                        # Store the scaled 0-1 relative coordinates as a string representation of a list
                        action_inputs[normalized_coord_key] = str(scaled_numbers)
                        # print(f"Parsed and scaled '{coord_input_value}' (from param '{param_name}') -> {action_inputs[normalized_coord_key]}") # Debugging print
                    else:
                        # Parsing or scaling failed
                        action_inputs[normalized_coord_key] = None # Indicate failure
                        print(f"Failed to parse/scale coordinate data '{coord_input_value}' (from param '{param_name}'). Resulting numbers: {numbers}, scaled_numbers: {scaled_numbers}.")

                except Exception as e:
                    # Catch any other errors during coordinate processing
                    print(f"Error processing coordinate data '{coord_input_value}' (from param '{param_name}'): {e}")
                    action_inputs[normalized_coord_key] = None # Indicate parsing failure


            # --- Other Parameter Handling (Non-Coordinates) ---
            else:
                # For 'content', apply escape_single_quotes if it's a string
                if param_name_lower == 'content' and isinstance(param_value, str):
                    action_inputs[param_name.strip()] = escape_single_quotes(param_value)
                else:
                    # Store other parameters as they are (strings, numbers, booleans etc. from ast.literal_eval)
                    # Use the original param_name casing here, not lower
                    action_inputs[param_name.strip()] = param_value

        # Add the processed action to the list
        if action_data["action_type"] is not None: # Only add if action type was successfully parsed
            action_data["action_inputs"] = action_inputs # Assign the processed inputs
            actions.append(action_data)
        else:
             print(f"Skipping action due to missing type: {raw_action_str}")


    # After parsing all actions, if the first one is 'finished', handle that.
    # The parsing_response_to_pyautogui_code checks for "DONE" return value.
    # Let's explicitly check here and add a 'finished' action if it exists in the output.
    # The prompt has `finished(content='xxx')` as an action.
    # If the model outputs this, it should be parsed as a regular action.
    # The parsing_response_to_pyautogui_code function already checks action_type == "finished".


    return actions


def parsing_response_to_pyautogui_code(responses,
                                       image_height: int, # Original screenshot height (pixels)
                                       image_width: int,  # Original screenshot width (pixels)
                                       input_swap: bool = True) -> str:
    '''
    Takes a list of parsed action dictionaries (output of parse_action_to_structure_output)
    and converts them into a string of pyautogui code.
    Assumes coordinate values in action_inputs are string representations of
    0-1 relative coordinate lists ([x,y] or [x1,y1,x2,y2]).

    Args:
        responses: A list of parsed action dictionaries.
        image_height: The height of the original screenshot (for scaling).
        image_width: The width of the original screenshot (for scaling).
        input_swap: Whether to use clipboard for 'type' action.
    Returns:
        Generated pyautogui code string, or "DONE" if a finished action is present.
    '''

    pyautogui_code_lines = ["import pyautogui", "import time"]
    # Add pyperclip import only if input_swap is True and 'type' action is likely
    # For simplicity, add it unconditionally if input_swap is True
    if input_swap:
        # Check if pyperclip is available, add a note if not (more robust than crashing)
        try:
             import pyperclip
             pyautogui_code_lines.append("import pyperclip")
        except ImportError:
             pyautogui_code_lines.append("# Warning: pyperclip not found. Clipboard typing is disabled.")
             input_swap = False # Disable input swap if pyperclip is missing


    # Ensure responses is a list
    if isinstance(responses, dict):
        responses = [responses]

    # Include thought/reflection/observation in comments for the first action block if available
    # Check the first item for thought/reflection keys
    if responses and (responses[0].get("reflection") or responses[0].get("thought")):
         reflection_text = responses[0].get("reflection", "")
         thought_text = responses[0].get("thought", "")
         # Observation is not currently added to the action dict by parse_action_to_structure_output,
         # but if it were, you could include it here.

         comment_block_lines = ["'''"]
         if reflection_text:
             comment_block_lines.append("Reflection:\n" + reflection_text)
         if thought_text:
             if len(comment_block_lines) > 1: comment_block_lines.append("") # Add newline if preceded by content
             comment_block_lines.append("Thought:\n" + thought_text)
         comment_block_lines.append("'''")

         if len(comment_block_lines) > 2: # Only add if there's content beyond just the quotes
             pyautogui_code_lines.append("\n".join(comment_block_lines))


    # --- Helper to get pixel coordinates from action_inputs ---
    def get_pixel_coords_from_input(action_inputs, possible_keys, width, height):
         """
         Looks for coordinate string (string repr of [x,y] or [x1,y1,x2,y2] 0-1 relative)
         in action_inputs using possible_keys and scales to pixels.
         Returns pixel coords (px1, py1, px2, py2) or (None, None, None, None).
         Expects input coordinate lists to be 0-1 relative.
         """
         coord_list_str = None
         found_key = None
         for key in possible_keys:
              coord_list_str = action_inputs.get(key)
              if coord_list_str is not None:
                   found_key = key
                   break # Found a coordinate key

         if coord_list_str is None:
              # print(f"Debug: No coordinate key found among {possible_keys} in inputs {action_inputs}") # Too verbose
              return None, None, None, None # No coordinate found

         # Ensure the found value is actually a string representation of a list before trying to eval
         # Check if it looks like "[...]" or "(...)"
         if not isinstance(coord_list_str, str) or not (coord_list_str.strip().startswith('[') and coord_list_str.strip().endswith(']')) and not (coord_list_str.strip().startswith('(') and coord_list_str.strip().endswith(')')):
              print(f"Warning: Value for key '{found_key}' was not a string representation of a list/tuple: {coord_list_str} (type: {type(coord_list_str).__name__})")
              return None, None, None, None

         try:
             # Evaluate the string "[x,y]" or "[x1,y1,x2,y2]" into a Python list/tuple
             # This string comes from parse_action_to_structure_output and should be 0-1 relative coordinates.
             coords_rel = eval(coord_list_str)

             if isinstance(coords_rel, (list, tuple)):
                 if len(coords_rel) == 2:
                     # It's a point [x_rel, y_rel] (0-1 relative)
                     x_rel, y_rel = coords_rel
                     px = round(float(x_rel) * width, 3)
                     py = round(float(y_rel) * height, 3)
                     # Ensure coordinates are within screen bounds (optional safety)
                     px = max(0, min(px, width - 1))
                     py = max(0, min(py, height - 1))
                     return px, py, px, py # Return point as min/max for consistency (top-left, bottom-right of a 1x1 pixel area)
                 elif len(coords_rel) == 4:
                     # It's a box [x1_rel, y1_rel, x2_rel, y2_rel] (0-1 relative)
                     x1_rel, y1_rel, x2_rel, y2_rel = coords_rel
                     px1 = round(float(x1_rel) * width, 3)
                     py1 = round(float(y1_rel) * height, 3)
                     px2 = round(float(x2_rel) * width, 3)
                     py2 = round(float(py2) * height, 3) # <-- FIX: Use py2, not py2 here
                     py2 = round(float(y2_rel) * height, 3) # Corrected line

                     # Ensure coordinates are within screen bounds (optional safety)
                     px1 = max(0, min(px1, width - 1))
                     py1 = max(0, min(py1, height - 1))
                     px2 = max(0, min(px2, width - 1))
                     py2 = max(0, min(py2, height - 1))
                     # Ensure x1<=x2, y1<=y2 just in case
                     px1, px2 = sorted([px1, px2])
                     py1, py2 = sorted([py1, py2])
                     return px1, py1, px2, py2 # Return box coordinates
                 else:
                     print(f"Warning: get_pixel_coords_from_input received list of unexpected length ({len(coords_rel)}) for key '{found_key}': {coords_rel}")
                     return None, None, None, None
             else:
                 print(f"Warning: get_pixel_coords_from_input received non-list/tuple after eval for key '{found_key}': {type(coords_rel).__name__} = {coords_rel}")
                 return None, None, None, None

         except Exception as e:
             print(f"Error evaluating coordinate string '{coord_list_str}' for key '{found_key}': {e}")
             return None, None, None, None # Indicate failure


    # --- Action Type Handling ---
    for response_id, response in enumerate(responses):
        action_type = response.get("action_type")
        action_inputs = response.get("action_inputs", {})

        # Check for finished action first
        if action_type in ["finished"]:
            # If finished, return "DONE" string
            return "DONE"

        # Add a sleep before subsequent actions (unless it's the very first action)
        if response_id > 0:
            pyautogui_code_lines.append(f"\ntime.sleep(0.5)") # Add sleep before action
            # pyautogui_code_lines.append(f"# --- Action {response_id + 1} ({action_type}) ---")


        # Action Type specific code generation
        if action_type == "hotkey":
            key_param = action_inputs.get("key") or action_inputs.get("hotkey")
            if key_param and isinstance(key_param, str):
                keys = key_param.split()
                convert_keys = []
                for key in keys:
                    # Map common names to pyautogui keys (lowercase comparison)
                    key_lower = key.lower()
                    if key_lower == "space": key = ' '
                    elif key_lower == "arrowleft": key = "left"
                    elif key_lower == "arrowright": key = "right"
                    elif key_lower == "arrowup": key = "up"
                    elif key_lower == "arrowdown": key = "down"
                    # Add more mappings if needed (e.g., esc -> escape, enter -> enter, win -> win)
                    elif key_lower == "esc": key = "escape"
                    elif key_lower == "enter": key = "enter"
                    elif key_lower == "win": key = "win" # Windows key
                    elif key_lower == "cmd": key = "command" # Mac command key
                    elif key_lower == "alt": key = "alt" # Ensure alt is mapped
                    elif key_lower == "ctrl": key = "ctrl"
                    elif key_lower == "shift": key = "shift"

                    convert_keys.append(key) # Use the potentially remapped key

                if convert_keys:
                     # Use repr() to ensure strings are quoted correctly in the generated code
                     pyautogui_code_lines.append(f"pyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})")
                else:
                     pyautogui_code_lines.append(f"# Hotkey action with empty or invalid keys: {key_param}")
            else:
                 pyautogui_code_lines.append(f"# Hotkey action missing 'key' or 'hotkey' parameter.")


        elif action_type in ["press", "keydown", "release", "keyup"]:
            # Mapping:
            # 'press' -> pyautogui.press (keyDown then keyUp)
            # 'keydown' -> pyautogui.keyDown
            # 'release' -> pyautogui.keyUp
            # 'keyup' -> pyautogui.keyUp (alias)

            key_param = action_inputs.get("key") or action_inputs.get(action_type) # Check 'key' or parameter named after action_type
            if key_param and isinstance(key_param, str):
                key = key_param.lower()
                if key == "arrowleft": key = "left"
                elif key == "arrowright": key = "right"
                elif key == "arrowup": key = "up"
                elif key == "arrowdown": key = "down"
                elif key == "space": key = " "
                elif key == "esc": key = "escape"
                elif key == "enter": key = "enter"
                 # Add more key mappings as needed

                if action_type == "press":
                    pyautogui_code_lines.append(f"pyautogui.press({repr(key)})") # pyautogui.press handles down/up
                elif action_type == "keydown":
                    pyautogui_code_lines.append(f"pyautogui.keyDown({repr(key)})")
                elif action_type in ["release", "keyup"]:
                     pyautogui_code_lines.append(f"pyautogui.keyUp({repr(key)})")
            else:
                 pyautogui_code_lines.append(f"# {action_type.capitalize()} action missing 'key' parameter.")


        elif action_type == "type":
            content = action_inputs.get("content", "")
            # content should already have single quotes escaped from parse_action_to_structure_output
            # ensure it's a string, default to empty string if None
            content = str(content) if content is not None else ""

            # Check if the content ends with a newline character
            ends_with_newline = content.endswith("\n") or content.endswith("\\n")
            # Strip trailing newlines for typing, the Enter press is separate
            stripped_content = content.rstrip("\\n").rstrip("\n") if ends_with_newline else content

            # Use repr() for the content string to handle internal quotes/backslashes correctly in the generated code
            content_repr = repr(stripped_content)

            if stripped_content:
                if input_swap:
                    # Ensure pyperclip is imported
                    # Check if "import pyperclip" is already in the lines to avoid duplicates
                    if "import pyperclip" not in pyautogui_code_lines:
                         # Check if pyperclip is available before adding import and using it
                         try:
                              import pyperclip
                              pyautogui_code_lines.insert(0, "import pyperclip")
                         except ImportError:
                              pyautogui_code_lines.append("# Warning: pyperclip not found. Clipboard typing is disabled.")
                              input_swap = False # Disable for this action

                    if input_swap: # Proceed only if input_swap is still True
                         pyautogui_code_lines.append(f"pyperclip.copy({content_repr})")
                         pyautogui_code_lines.append(f"pyautogui.hotkey('ctrl', 'v')")
                         pyautogui_code_lines.append(f"time.sleep(0.1)") # Short sleep after paste
                    else:
                         # Fallback to pyautogui.write if pyperclip is missing
                         pyautogui_code_lines.append(f"pyautogui.write({content_repr}, interval=0.01)")
                         pyautogui_code_lines.append(f"time.sleep(0.1)")

                else:
                    # pyautogui.write types characters with an interval. Does not interpret \n.
                    pyautogui_code_lines.append(f"pyautogui.write({content_repr}, interval=0.01)") # Small interval
                    pyautogui_code_lines.append(f"time.sleep(0.1)") # Short sleep after typing


            if ends_with_newline:
                 # If the original content ended with a newline, simulate pressing Enter.
                 pyautogui_code_lines.append(f"pyautogui.press('enter')")
                 pyautogui_code_lines.append(f"time.sleep(0.1)") # Short sleep after enter


        elif action_type in ["drag", "select"]:
            # Look for start/end coordinates using their normalized keys
            sx1, sy1, sx2, sy2 = get_pixel_coords_from_input(action_inputs, ['start_point', 'start_box', 'point', 'start'], image_width, image_height) # Added 'point', 'start' as fallback keys
            ex1, ey1, ex2, ey2 = get_pixel_coords_from_input(action_inputs, ['end_point', 'end_box', 'end'], image_width, image_height) # Added 'end' as fallback key


            if (sx1 is not None and sy1 is not None and ex1 is not None and ey1 is not None):
                # Use the center of the start area and end area for drag
                # Handle cases where it's a single point coordinate (sx1==sx2, sy1==sy2)
                start_center_x = (sx1 + sx2) / 2
                start_center_y = (sy1 + sy2) / 2
                end_center_x = (ex1 + ex2) / 2
                end_center_y = (ey1 + ey2) / 2

                pyautogui_code_lines.append(f"pyautogui.moveTo({start_center_x}, {start_center_y})")
                pyautogui_code_lines.append(f"time.sleep(0.1)") # Small pause after move before drag starts
                pyautogui_code_lines.append(f"pyautogui.dragTo({end_center_x}, {end_center_y}, duration=0.5)") # Default duration 0.5s
            else:
                pyautogui_code_lines.append(f"# Failed to parse coordinates for {action_type}. Looked for start/end_point/box/start/end.")


        elif action_type == "scroll":
            # Look for the scroll target coordinate: 'point' or 'start_point' or 'start_box' or 'start'
            px1, py1, px2, py2 = get_pixel_coords_from_input(action_inputs, ['point', 'start_point', 'start_box', 'start'], image_width, image_height) # Added 'start'


            # Use the center if coordinate was found
            scroll_x, scroll_y = None, None
            if px1 is not None and py1 is not None:
                 scroll_x = (px1 + px2) / 2
                 scroll_y = (py1 + py2) / 2

            direction = action_inputs.get("direction", "").lower()
            scroll_amount = 10 # Default scroll amount (lines/units) - adjust as needed

            # Determine scroll direction and amount for pyautogui.scroll (vertical) or hscroll (horizontal)
            v_scroll_amount = 0
            h_scroll_amount = 0

            if "up" in direction:
                v_scroll_amount = scroll_amount
            elif "down" in direction:
                v_scroll_amount = -scroll_amount
            elif "left" in direction:
                 h_scroll_amount = -scroll_amount
            elif "right" in direction:
                 h_scroll_amount = scroll_amount
            else:
                 pyautogui_code_lines.append(f"# Warning: Unknown scroll direction: {direction}. No scroll action generated.")
                 # Do nothing if direction is unclear

            # Generate the appropriate pyautogui scroll command
            if v_scroll_amount != 0:
                # pyautogui.scroll syntax: scroll(clicks, x=..., y=...)
                if scroll_x is not None and scroll_y is not None:
                    # Scroll at specific coordinates
                    pyautogui_code_lines.append(f"pyautogui.scroll({v_scroll_amount}, x={scroll_x}, y={scroll_y})")
                else:
                    # Scroll at the current mouse position or active element
                    pyautogui_code_lines.append(f"pyautogui.scroll({v_scroll_amount})")
            elif h_scroll_amount != 0:
                 # pyautogui.hscroll syntax: hscroll(clicks, x=..., y=...)
                 if scroll_x is not None and scroll_y is not None:
                    # Scroll horizontally at specific coordinates
                    pyautogui_code_lines.append(f"pyautogui.hscroll({h_scroll_amount}, x={scroll_x}, y={scroll_y})")
                 else:
                    # Scroll horizontally at the current mouse position or active element
                    pyautogui_code_lines.append(f"pyautogui.hscroll({h_scroll_amount})")


        elif action_type in [
                "click", "left_single", "left_double", "right_single", "hover", "long_press"
        ]:
            # Look for the primary target coordinate using its normalized keys: 'point', 'start_point', 'start_box', or just 'start'
            px1, py1, px2, py2 = get_pixel_coords_from_input(action_inputs, ['point', 'start_point', 'start_box', 'start'], image_width, image_height) # Added 'start'


            if px1 is not None and py1 is not None:
                 # Use the center of the area as the click/hover location
                 # Handle cases where it's a single point coordinate (px1==px2, py1==py2)
                 target_x = (px1 + px2) / 2
                 target_y = (py1 + py2) / 2

                 # Ensure mouse is at target before clicking/dragging (often good practice)
                 pyautogui_code_lines.append(f"pyautogui.moveTo({target_x}, {target_y})")
                 pyautogui_code_lines.append(f"time.sleep(0.1)") # Short pause after move

                 # Generate the specific click/hover/long_press command
                 if action_type == "left_single" or action_type == "click":
                     pyautogui_code_lines.append(f"pyautogui.click(button='left')") # Click at current mouse position
                 elif action_type == "left_double":
                     pyautogui_code_lines.append(f"pyautogui.doubleClick(button='left')") # Double click at current position
                 elif action_type == "right_single":
                     pyautogui_code_lines.append(f"pyautogui.click(button='right')") # Right click at current position
                 elif action_type == "hover":
                     # moveTo already done, no further action needed for hover
                     pass # pyautogui.moveTo performs the hover implicitly
                 elif action_type == "long_press":
                     pyautogui_code_lines.append(f"pyautogui.mouseDown(button='left')")
                     pyautogui_code_lines.append(f"time.sleep(1.0)") # Long press duration (adjust if needed)
                     pyautogui_code_lines.append(f"pyautogui.mouseUp(button='left')")

            else:
                pyautogui_code_lines.append(f"# Failed to parse coordinates for {action_type}. Looked for point/start_point/start_box/start.")

        # Handle Mobile specific actions that might appear unexpectedly in a computer task
        elif action_type in ["open_app", "press_home", "press_back"]:
             # These actions do not have standard PyAutoGUI equivalents for desktop control.
             # Add comments indicating they are mobile actions.
             pyautogui_code_lines.append(f"# Action '{action_type}' is typically for mobile tasks and is not implemented for computer control.")
             pyautogui_code_lines.append(f"pass # No action executed for {action_type}")


        elif action_type == "wait":
             # Implement the wait action (sleep for a predefined duration)
             wait_duration = 5 # Default wait duration in seconds
             # Check if a duration parameter was provided (e.g., wait(duration=10))
             duration_param = action_inputs.get("duration")
             if duration_param is not None:
                 try:
                     # Allow float or int duration
                     wait_duration = float(duration_param)
                     if wait_duration < 0:
                         print(f"Warning: Negative wait duration provided ({duration_param}). Using default {5}s.")
                         wait_duration = 5
                 except (ValueError, TypeError):
                      print(f"Warning: Invalid wait duration provided ({duration_param}). Using default {5}s.")
                      wait_duration = 5

             pyautogui_code_lines.append(f"time.sleep({wait_duration})")


        else:
            # Catch any other unrecognized action types
            pyautogui_code_lines.append(f"\n# Unrecognized action type: {action_type}. No code generated.")
            pyautogui_code_lines.append(f"pass # Placeholder for unrecognized action")


    # Return the concatenated string of pyautogui code lines
    return "\n".join(pyautogui_code_lines)


def add_box_token(input_string):
    # This function's purpose seems to be adding specific tokens around coordinates
    # in the text string itself, likely for visualization or training data prep.
    # It's not part of the core execution flow (model output -> parse -> action).
    # Keeping the original logic as provided, assuming its specific token requirements.

    # Look for the "Action: " part to separate it from Thought/Reflection
    parts = input_string.split("Action: ", 1)
    prefix = parts[0] + "Action: " if len(parts) > 1 else ""
    action_part = parts[-1] if len(parts) > 1 else input_string # If no "Action: ", process the whole string?

    # Split action part into individual action strings based on common separator
    raw_action_strings = action_part.split(")\n\n")

    processed_action_strings = []
    for action_str in raw_action_strings:
        action_str = action_str.strip()
        if action_str and not action_str.endswith(')'):
             action_str += ')' # Add back missing parenthesis if split removed it

        if not action_str: continue # Skip empty strings

        # Pattern to find parameter assignments like `param='value'` or `param=(value)` or `param=[value]`
        # Need to be careful not to match inside other quoted strings.
        # This regex finds `word=value` where value is a quoted string or a Python literal (tuple, list, number, etc.)
        # It's hard to perfectly parse Python syntax with regex, but this tries to find assignments.
        param_assignment_pattern = re.compile(r"(\w+)\s*=\s*((['\"]).*?\3|[\w\.\-+\(\)\[\]]+)", re.DOTALL) # Match quoted strings or basic literals

        updated_action = action_str
        # Iterate through parameter assignments within this action string
        for param_match in param_assignment_pattern.finditer(action_str):
             param_name = param_match.group(1)
             param_value_full_text = param_match.group(2) # The full text of the value part, including quotes/brackets
             # param_value_content = param_match.group(3) # Content inside quotes (only if quoted)
             original_param_assignment_text = param_match.group(0) # Full text `name=value`

             # Check if this parameter name suggests it contains coordinates
             param_name_lower = param_name.lower()
             if "box" in param_name_lower or "point" in param_name_lower or param_name_lower in ['start', 'end']:
                  # Now find the coordinate pattern *within* the value's text representation.
                  # Look for <point>...</point> or (N,N) or [N,N,N,N] etc.
                  # This regex looks for known coordinate formats within the parameter value string.
                  # It needs to be robust to spaces and potential quotes.
                  # Target formats: <point>...</point>, (N,N), [N,N], (N,N,N,N), [N,N,N,N], (X:F Y:F) - though the last one is unlikely in raw form.
                  # Let's focus on formats potentially found *inside quotes* or as *raw literals*.
                  coord_pattern_in_value = re.compile(
                      r"(<point>\s*X:\s*\d+\.?\d*\s*Y:\s*\d+\.?\d*\s*</point>|" # <point>X: F Y: F</point>
                      r"<point>\s*\d+\s+\d+\s*</point>|"                      # <point>N N</point>
                      r"\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)|"                 # (N,N) or (F,F)
                      r"\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]|"                 # [N,N] or [F,F]
                      r"\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)|" # (N,N,N,N) or (F,F,F,F)
                      r"\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]"  # [N,N,N,N] or [F,F,F,F]
                      r")"
                  )

                  updated_param_value_full_text = param_value_full_text
                  # Use finditer to get all coordinate matches in the value's text representation
                  last_end = 0
                  temp_updated_value_text = ""
                  for coord_match_iter in coord_pattern_in_value.finditer(param_value_full_text):
                       # Append text before this match
                       temp_updated_value_text += param_value_full_text[last_end : coord_match_iter.start()]
                       # Append the tokenized coordinate
                       found_coord_text = coord_match_iter.group(0)
                       temp_updated_value_text += f"<|box_start|>{found_coord_text}<|box_end|>"
                       last_end = coord_match_iter.end()

                  # Append any remaining text after the last match
                  temp_updated_value_text += param_value_full_text[last_end:]
                  updated_param_value_full_text = temp_updated_value_text

                  # If the parameter value text was updated, rebuild the full parameter assignment string
                  if updated_param_value_full_text != param_value_full_text:
                      new_param_assignment_text = f"{param_name}={updated_param_value_full_text}"
                       # Replace the original parameter assignment text with the new one in the action string
                       # Use replace with count=1 in case the same assignment appears multiple times (unlikely but safer)
                      updated_action = updated_action.replace(original_param_assignment_text, new_param_assignment_text, 1)

            # else: not a coordinate parameter, add as is

        processed_action_strings.append(updated_action)


    # Step 5: Reconstruct the final string
    # Add back the original prefix (Thought/Reflection/Action: ) if it existed
    final_string = prefix + "\n\n".join(processed_action_strings)

    return final_string
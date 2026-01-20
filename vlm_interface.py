import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import gc

# Define model globally but initialize as None
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
model = None
processor = None

def load_vlm():
    global model, processor
    if model is None:
        print(f"--> [VLM] Lazy Loading {MODEL_ID}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype="auto", 
            device_map="auto" 
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor

def get_vlm_analysis(video_path):
    """
    Analyzes the robot video and returns a description of the failure/success.
    """
    print(f"--> [VLM] Analyzing video: {video_path}")
    
    # Lazy Load
    model, processor = load_vlm()
    
    # Improved prompt for Reward Design
    prompt_text = (
        "You are a Physics Compliance Officer. Watch the video frame-by-frame.\n"
        "Locate the robotic arm. Describe the motion of the robotic arm."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 420 * 560, # Keep resolution moderate for speed
                    "fps": 4.0, 
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Clean up GPU memory for the text tensors
    del inputs, generated_ids
    torch.cuda.empty_cache()
    
    return output_text[0]

def get_vlm_score(video_path):
    """
    Analyzes the video and returns a tuple (score, rationale).
    Score is an integer 0-100.
    Rationale is the text explanation.
    """
    print(f"--> [VLM] Scoring video: {video_path}")
    
    # Lazy Load
    model, processor = load_vlm()

    # Prompt from Qwen.py
    prompt_text = (
        "You are a Physics Compliance Officer. Watch the video frame-by-frame.\n"
        "Task is to lift the cube from the table. Give a score from 0 to 100 to the robotic arm,\n"
        "100 is the best meaning robotic arm was successfull lifting the cube, 0 is the worst meaning robotic arm was not even doing related movement to lift the cube.\n"
        ""
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 420 * 560,
                    "fps": 4.0, 
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Preparation
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Cleanup
    del inputs, generated_ids
    torch.cuda.empty_cache()

    # Parse Score
    import re
    score = 0
    # Look for number 0-100. We take the last number found as it's often the conclusion
    # OR we look for "Score: X" pattern.
    # Pattern 1: Explicit "Score: 100"
    match = re.search(r"Score:\s*(\d+)", output_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
    else:
        # Pattern 2: Just find any number.
        numbers = re.findall(r"\d+", output_text)
        if numbers:
            # Filter for valid range 0-100
            valid_nums = [int(n) for n in numbers if 0 <= int(n) <= 100]
            if valid_nums:
                score = valid_nums[-1] # Take the last one (usually "I give it a score of X")
    
    # return
    print(f"    > VLM Score: {score}")
    return score, output_text
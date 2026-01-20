import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. Load Model (Use bfloat16 for speed/memory if on Ampere+ GPU, else float16)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 2. Improved Prompt (Chain of Thought)
# We ask it to look for the GAP between the object and the table.
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
                "video": "rl-video-step-23.mp4",
                # FIX 1: Higher resolution to see the gap
                "max_pixels": 420 * 560, 
                # FIX 2: Higher FPS to catch the motion
                "fps": 4.0, 
            },
            {"type": "text", "text": prompt_text},
        ],
    }
]

# 3. Inference
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

# Generate slightly more tokens to allow for reasoning
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(f"\n[QWEN ANALYSIS]:\n{output_text[0]}")
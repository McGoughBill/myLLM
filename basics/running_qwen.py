from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os

local_qwen_path = '/Users/bill/Documents/qwen_3.5_8B_VL'

access_token = os.getenv("HF_TOKEN")
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    local_qwen_path,local_files_only=True,dtype="auto", device_map="auto",
    token=access_token, trust_remote_code=True
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(local_qwen_path,local_files_only=True,dtype="auto", device_map="auto")

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/Users/bill/Documents/Screenshot 2026-05-22 at 19.43.43.png",
            },
            {"type": "text", "text": "The attached image is a screenshot of a Pokemon Fire Red. "
                                     "Specifically, we are in Pallet town. Please desribe the surrounding features of the location,"
                                     "including descriptions of how far (up,down,left,right) those features are. Keep your descriptions brief and informative."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    token=access_token
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
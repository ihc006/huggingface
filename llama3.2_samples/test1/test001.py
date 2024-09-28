import requests as rq
import torch
from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor

# from transformers import LlamaForCausalLM, LlamaTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# model = MllamaForConditionalGeneration.from_pretrained(
#    model_id,
#    torch_dtype=torch.bfloat16,
#    device_map="auto",
#)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# processor = AutoProcessor.from_pretrained(model_id)

url = "rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
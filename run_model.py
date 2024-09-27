from internvl2_helper import OVInternVLChatModel
from transformers import AutoTokenizer
from pathlib import Path
import PIL
from internvl2_helper import load_image
from transformers import TextIteratorStreamer
from threading import Thread
import requests

model_id = "InternVL2-4B"
pt_model_id = "E:/Models/" + model_id
model_dir = pt_model_id + "-int4-openvino"
model_dir = Path(model_dir)

device = "GPU"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
ov_model = OVInternVLChatModel(model_dir, device)

EXAMPLE_IMAGE = Path("examples_image1.jpg")
EXAMPLE_IMAGE_URL = "https://huggingface.co/OpenGVLab/InternVL2-2B/resolve/main/examples/image1.jpg"

if not EXAMPLE_IMAGE.exists():
    img_data = requests.get(EXAMPLE_IMAGE_URL).content
    with EXAMPLE_IMAGE.open("wb") as handler:
        handler.write(img_data)

pixel_values = load_image(EXAMPLE_IMAGE, max_num=12)

print("pixel values:", pixel_values.shape)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_config = dict(max_new_tokens=100, do_sample=True, streamer=streamer)
question = "<image>\n请简单描述下这个图片."

print(f"User: {question}\n")
print("Assistant:")

thread = Thread(
    target=ov_model.chat,
    kwargs=dict(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        history=None,
        return_history=False,
        generation_config=generation_config,
    ),
)
thread.start()

generated_text = ""
# Loop through the streamer to get the new text as it is generated
for new_text in streamer:
    if new_text == ov_model.conv_template.sep:
        break
    generated_text += new_text
    print(new_text, end="", flush=True)  # Print each new chunk of generated text on the same line

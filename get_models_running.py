import os
import argparse
# Change the cache location
os.environ['TRANSFORMERS_CACHE'] = "/scratch/s4790383/.cache"
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import pandas
import torch
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
import base64


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # GENERAL STUFF
    parser.add_argument("-l", "--lvlm", type=str, choices=["qwen", "llava", "blip", "all"], default="qwen",
                        help="Choose which LVLM system to run: Qwen2, LLaVa-1.6, InstructBLIP or all of them")

    args = parser.parse_args()
    return args



def run_qwen(messages):
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

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
    inputs = inputs.to("cuda:0")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


def zero_shot(image, prompt, lvlm):

    #image_input = {}

    if lvlm == "qwen":
        image_input = {
                        "type": "image",
                        "image": image
                    }
    elif lvlm == "llava":
        image_input = {
                        "type": "image_url",
                        "image_url": {"url": image}
                    }

    messages = [
        {
            "role": "user",
            "content": [
                image_input,
                {"type": "text", 
                    "text": prompt
                    # "text": "Describe this image."
                    },
            ],
        }
    ]

    return messages


def run_llava(prompt_pipe):
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                              #torch_dtype=torch.float16, 
                                                              torch_dtype="auto", 
                                                              device_map="auto",
                                                              low_cpu_mem_usage=True,
                                                              load_in_4bit=True) 
    model.to("cuda:0")

    inputs = processor.apply_chat_template(prompt_pipe,
                                           add_generation_prompt=True,
                                           tokenize=True, 
                                           return_dict=True,
                                           return_tensors="pt").to("cuda:0")

    # inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))


def load_data():

    # Load the separate splits
    #train_data = load_dataset("anson-huang/mirage-news", split="train")
    #valid_data = load_dataset("anson-huang/mirage-news", split="validation")
    #test_data  = load_dataset("anson-huang/mirage-news", split="test")

    # return train_data, valid_data, test_data
    return load_dataset("anson-huang/mirage-news")


def main():
    args = create_arg_parser()
    data = load_data()
    df_train = data["train"].to_pandas()

    sample = df_train.sample(n = 1)
    image_bytes = sample.iloc[0, 0]["bytes"]
    misinfo_label = sample.iloc[0, 1]
    text_sample = sample.iloc[0, 2]

    # Converting the image to base64 for Qwen input (and maybe for LLaVa?)
    image_base64 = base64.b64encode(image_bytes).decode()
    image_input = f"data:image;base64,{image_base64}"

    prompt = f"You are a misinformation detection model. I provide you an image and related to it caption. The entry can be a real snippet from a news article, or an AI-generated entry. Please determine whether the entry is real or fake and elaborate why. Text: {text_sample}"

    if args.lvlm == "all":
        prompt_pipe = zero_shot(image_input, prompt, "qwen")
        run_qwen(prompt_pipe)

        prompt_pipe = zero_shot(image_input, prompt, "llava")
        run_llava(prompt_pipe)

    elif args.lvlm == "qwen":
        prompt_pipe = zero_shot(image_input, prompt, args.lvlm)
        run_qwen(prompt_pipe)


    elif args.lvlm == "llava":
        prompt_pipe = zero_shot(image_input, prompt, args.lvlm)
        run_llava(prompt_pipe)

    print(f"TRUE LABEL: {misinfo_label}")


if __name__ == "__main__":
    main()

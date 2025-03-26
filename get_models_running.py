import os
import argparse
# Change the cache location
os.environ['TRANSFORMERS_CACHE'] = "/scratch/s4790383/.cache"
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image
from datasets import load_dataset
import pandas
import torch
import sys
torch.cuda.empty_cache()
import base64


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # GENERAL STUFF
    parser.add_argument("-l", "--lvlm", type=str, choices=["qwen", "llava", "idefics", "all"], default="qwen",
                        help="Choose which LVLM system to run: Qwen2, LLaVa-1.6, InstructBLIP or all of them")
    parser.add_argument("-p", "--prompt", type=str, choices=["zero-shot", "few-shot", "both"], default="one-shot",
                        help="Choose which prompt type to apply: one-shot, few-shot, or both")

    args = parser.parse_args()
    return args


def load_sample(sample, data, index, is_bytes=False):

    # Retrieve the label
    if sample.iloc[1] == 0: misinfo_label = "True"
    else: misinfo_label = "Fake"

    # Prepare the prompt
    text_sample = sample.iloc[2]
    prompt = f"You are a misinformation detection model. I provide you an image and related to it caption. The entry can be a real snippet from a news article, or an AI-generated entry. Determine whether the entry is real or fake. Provide firstly the label, REAL or FAKE, and then explain why. Text: {text_sample}"

    # Prepare the image in bytes
    if is_bytes:
        image_bytes = sample.iloc[0]["bytes"]
        # Converting the image to base64 for Qwen input (and maybe for LLaVa?)
        image_base64 = base64.b64encode(image_bytes).decode()
        image_input = f"data:image;base64,{image_base64}"

    # Image as PIL (load via the path)
    else:
        # image_pil_path = data["train"][sample.index.to_list()[0]]["image"]
        image_pil_path = data[index]["image"]
        image_input = load_image(image_pil_path)

    return {"data_index" : index + 1, # Restore the entry position based on dataset
            "caption": text_sample, 
            "image": image_input,
            "prompt": prompt, 
            "label": misinfo_label
            }


def zero_shot(image, prompt, lvlm):

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

    elif lvlm == "idefics":
        image_input = {"type": "image"}


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    return messages


def few_shot(prompt1, prompt2, prompt3):
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt1},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The entry is FAKE. Neither the woman on the left nor on the right looks like Christopher Dodd's wife, Jackie Clegg. Also, senate discusses the question regarding the actions of the president and the country and does not allow to announce personal matters."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt2},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The entry is REAL. A little more than half of California voters ended up supporting Proposition 8, outlawing same-sex marriage in the state. The measure was immediately challenged in court, and in 2013, the U.S. Supreme Court ruled that the defendants in the case had no legal standing, which meant that Proposition 8 was blocked and same-sex marriage could continue. Despite this, a lot of protests started to show. The entry supports the real event, as shown on the image with the anti-same-sex marriage slogans like 'Marriage = Man + Woman'."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt3},
        ]
    }       
    ]
    
    return messages


def choose_prompt(args, model, sample1_dict, sample2_dict, input_dict):
    if args.prompt == "zero-shot":
        prompt_pipe = zero_shot(input_dict["image"],
                                input_dict["prompt"],
                                model)
        image_input = input_dict["image"]

    elif args.prompt == "few-shot":
        prompt_pipe = few_shot(sample1_dict["prompt"],
                                sample2_dict["prompt"],
                                input_dict["prompt"])
        image_input = [sample1_dict["image"], sample2_dict["image"], input_dict["image"]]

    elif args.prompt == "both": pass # TODO: write the code for both prompt types

    return prompt_pipe, image_input


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


def run_idefics(prompt_pipe, image_input):
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3",
                                                   torch_dtype="auto",
                                                   device_map="auto",
                                                   low_cpu_mem_usage=True
                                                   ).to("cuda:0")

    # Convert the prompt pipe to model-friendy input
    prompt = processor.apply_chat_template(prompt_pipe, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image_input, return_tensors="pt") # add a list of images if few-shot
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(generated_texts)
    print("Done")


def run_lvlm(prompt_pipe, image_input, model):
    match model:
        case "qwen": model_name = "Qwen/Qwen2-VL-7B-Instruct"
        case "llava": model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        case "idefics": model_name = "HuggingFaceM4/Idefics3-8B-Llama3"
        case _: sys.exit("No valid LVLM name provided, aborting the program")

    # Initialize the model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name,
                                                   torch_dtype="auto",
                                                   device_map="auto",
                                                   low_cpu_mem_usage=True
                                                   ).to("cuda:0")

    # Prepare the input
    prompt = processor.apply_chat_template(prompt_pipe, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image_input, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    # Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(generated_texts)


def main():
    args = create_arg_parser() # TODO: write a check of the arguments
    model_list = ["qwen", "llava", "idefics"]

    # Load the separate data splits
    data = load_dataset("anson-huang/mirage-news", split="train")
    #valid_data = load_dataset("anson-huang/mirage-news", split="validation")
    #test_data  = load_dataset("anson-huang/mirage-news", split="test")

    #data = load_dataset("anson-huang/mirage-news")
    #df_train = data["train"].to_pandas()
    df_train = data.to_pandas()

    # Load the dataset entries

    sample1_dict = load_sample(df_train.iloc[1], data, 1) # Fake entry
    print(sample1_dict)
    sample2_dict = load_sample(df_train.iloc[5003], data, 5003) # True entry
    print(sample2_dict)
    sample = df_train.sample()
    input_dict = load_sample(sample.iloc[0], data, sample.index.to_list()[0]) # Random entry TODO ensure it is neither of the above
    print(input_dict)

    # Create the prompt and run the models
    if args.lvlm == "all":
        for model in model_list:
            prompt_pipe, image_input = choose_prompt(args, model, sample1_dict, sample2_dict, input_dict)
            run_lvlm(prompt_pipe, image_input, model)

    else:
        prompt_pipe, image_input = choose_prompt(args, args.lvlm, sample1_dict, sample2_dict, input_dict)
        run_lvlm(prompt_pipe, image_input, args.lvlm)

    # elif args.lvlm == "qwen":
    #     prompt_pipe = zero_shot(image_input, prompt, args.lvlm)
    #     run_qwen(prompt_pipe)


    # elif args.lvlm == "llava":
    #     prompt_pipe = zero_shot(image_input, prompt, args.lvlm)
    #     run_llava(prompt_pipe)

    # elif args.lvlm == "idefics":
    #     prompt_pipe = zero_shot(image_input, prompt, args.lvlm)
    #     run_idefics(prompt_pipe, image_pil)

    print(f"TRUE LABEL: {input_dict['label']}")


if __name__ == "__main__":
    main()


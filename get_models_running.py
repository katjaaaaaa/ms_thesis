# Docs used for examples: 2, 5004, 78, 5065


import os
import argparse
# Change the cache location
os.environ['TRANSFORMERS_CACHE'] = "/scratch/s4790383/.cache"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image, ImageFeatureExtractionMixin
from datasets import load_dataset
import pandas
import torch
import sys
# torch.cuda.empty_cache()
import base64
import json
from datetime import datetime
import re


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # GENERAL STUFF
    parser.add_argument("-l", "--lvlm", type=str, choices=["qwen", "llava", "idefics", "all"], default="qwen",
                        help="Choose which LVLM system to run: Qwen2, LLaVa-1.6, InstructBLIP or all of them")
    parser.add_argument("-p", "--prompt", type=str, choices=["zero-shot", "few-shot", "both"], default="one-shot",
                        help="Choose which prompt type to apply: zero-shot, few-shot, or both")
    parser.add_argument("-pf", "--prompt_file", type=str, default="prompt01", help="Inserd desired prompt file (Must be .txt)")
    parser.add_argument("-ex", "--examples", type=int, nargs="+", default=[5004, 2],
                        help="Determines the example entries for the few-shot prompt. Choose from: [5004, 2, 5065, 78]")
    parser.add_argument("-n", "--n_samples", type=int, default=1, help="Determines the number of randomly taken samples from the training data")
    parser.add_argument("-ds", "--data_split", type=str, default="train", choices=["train","validate","test"], help="Chooses a data split to work with")
    parser.add_argument("-ap", "--answer_pipe", type=str, default="answer_pipe", help="Assign the answer pipe used in the few-shot prompt (Must be .json)")
    parser.add_argument("-plo", "--prompt_label_only", type=bool, default=False, help="Decide whether apply the 'label only' prompt")

    args = parser.parse_args()
    return args


def load_sample(args, sample, data, index, promptfile, is_bytes=False):

    # Retrieve the label
    if sample.iloc[1] == 0: misinfo_label = 0
    else: misinfo_label = 1

    # Prepare the prompt
    with open(f"{promptfile}.txt") as f:
        prompt_base = f.read()
    text_sample = f"Text: {sample.iloc[2]}"
    prompt = prompt_base + "\n\n" + text_sample

    # Prepare the image in bytes
    if is_bytes:
        image_bytes = sample.iloc[0]["bytes"]
        # Converting the image to base64 for Qwen input (and maybe for LLaVa?)
        image_base64 = base64.b64encode(image_bytes).decode()
        image_input = f"data:image;base64,{image_base64}"

    # Image as PIL (load via the path)
    else:
        image_pil_path = data[index]["image"]
        image_input = load_image(image_pil_path)
        processor = ImageFeatureExtractionMixin()
        image_resized = processor.resize(image=image_input, size=450, default_to_square=False, max_size=600) # if height > width, then image will be rescaled to (size * height / width, size)

    return {"data_index" : str(index + 1), # Restore the entry position based on its position in the dataset
            "caption": sample.iloc[2],
            "image": image_resized,
            "prompt": prompt, 
            "label": misinfo_label
            }


def zero_shot(prompt):

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


def few_shot(args, fewshot_list, input_prompt):

    # Load the prepared few-shot answers TODO: write a check for that
    with open(f"{args.answer_pipe}.json") as f:
        answer_dict = json.load(f)

    messages = []

    for entry_dict in fewshot_list:
        # Build a prompt part of the pipe
        messages.append(
          {
              "role": "user",
              "content": [
                  {"type": "image"},
                  {"type": "text", "text": entry_dict["prompt"]},
              ]
          })

        # Build example answer part of the pipe
        messages.append(
          {
              "role": "assistant",
              "content": [ # 2
                  {"type": "text", "text": answer_dict[entry_dict["data_index"]]["text"]}
              ]
          })


    # Lastly, add the input prompt to finish the pipe
    messages.append(
          {
              "role": "user",
              "content": [
                  {"type": "image"},
                  {"type": "text", "text": input_prompt},
              ]
          })

    return messages


def choose_prompt(args, fewshot_list, input_dict):

    '''
    Takes few-shot and input entries as dictionaries, converts them to
    a specific prompt and returns as lists
    '''

    # Create two types of prompts
    zeroshot_pipe = zero_shot(input_dict["prompt"])

    fewshot_pipe = few_shot(args, fewshot_list,
                            input_dict["prompt"])

    # Adjust the image input
    if args.prompt == "zero-shot":
        prompt_pipe = [zeroshot_pipe]
        image_input = [input_dict["image"]]

    elif args.prompt == "few-shot":
        prompt_pipe = [fewshot_pipe]
        # image_input = [[sample1_dict["image"], sample2_dict["image"], input_dict["image"]]]
        image_input = [[sample_dict["image"] for sample_dict in fewshot_list] + [input_dict["image"]]]

    elif args.prompt == "both":
        prompt_pipe = [zeroshot_pipe, fewshot_pipe]
        image_input = [input_dict["image"], [sample_dict["image"] for sample_dict in fewshot_list] + [input_dict["image"]]]
        

    return prompt_pipe, image_input


def run_lvlm(model, args, fewshot_list, input_list, date_time):
    match model:
        case "qwen":
            model_name = "Qwen/Qwen2-VL-7B-Instruct"
            separator = "\nassistant\n"
        case "llava":
            model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
            separator = "[/INST]"
        case "idefics":
            model_name = "HuggingFaceM4/Idefics3-8B-Llama3"
            separator = "\nAssistant:"
        case _: sys.exit("No valid LVLM name provided, aborting the program")

    # Initialize the model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name,
                                                   torch_dtype="auto",
                                                   device_map="cuda",
                                                   low_cpu_mem_usage=True
                                                   ).to("cuda:0")

    # Loop through a list of sample dictionaries
    for input_dict in input_list:

        # Open an output file if exists 
        try:
            with open(f"{args.prompt}_{date_time}.json") as f:
                output_dict = json.load(f)
        except FileNotFoundError:
            output_dict = dict()
            output_dict["args"] = output_dict.get("args", dict())
            # { for key, value in vars(args).items() if value}
            for key, value in vars(args).items():
                if value:
                    output_dict["args"][key] = value

        prompt_list, image_list = choose_prompt(args, fewshot_list, input_dict)

        # Loop through the prompt types
        for i, prompt_pipe in enumerate(prompt_list):

            # Prepare the input
            prompt = processor.apply_chat_template(prompt_pipe, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image_list[i], return_tensors="pt")
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
            # Generate the output
            generated_ids = model.generate(**inputs,
                                           max_new_tokens=180,
                                           temperature=0.01, # 0.01 is the built-in temperature for qwen2
                                           do_sample=True)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
            # Store output in a dictionary
            index_str =input_dict["data_index"]
            output_dict[index_str] = output_dict.get(index_str, dict())
            output_dict[index_str]["caption"] = input_dict['caption']
            output_dict[index_str]["true_label"] = input_dict['label']

            if len(prompt_list) == 1: prompt_type = [args.prompt]
            else: prompt_type = ["zero-shot", "few-shot"]

            output_dict[index_str][prompt_type[i]] = output_dict[index_str].get(prompt_type[i], dict())
            output_dict[index_str][prompt_type[i]][model_name] = extract_output(generated_texts, separator)

            torch.cuda.empty_cache()

        # Save output as a JSON
        with open(f"{args.prompt}_{date_time}.json", "w") as f:
            json.dump(output_dict, f, indent=3)


def extract_output(text, separator):

    output_str = text[0].split(separator)[-1]
    try:
        #output_str = output_str.replace("\"model_label\"", "label")
        #output_str = output_str.replace("model_label", "explanation")
    
        output_str = output_str.replace("'model_label'", "\"model_label\"")
        output_str = output_str.replace("'model_explanation'", "\"model_explanation\"")

        #match = re.search(r'"model_explanation"\s*:\s*"(.+?)"\s*(?:,|\})', output_str).group(1)
        #corrected = match.replace("\"", "'")

        #output_str = re.sub(r'"model_explanation"\s*:\s*"(.+?)"\s*(?:,|\})',
                            #f'"model_explanation": "{corrected}"', output_str)
        
        
        output_dict = json.loads(output_str)

    except json.decoder.JSONDecodeError:
        output_dict = output_str

    return output_dict


def main():
    args = create_arg_parser() # TODO: write a check of the arguments
    model_list = ["qwen", "llava", "idefics"]

    # Load the separate data splits
    data = load_dataset("anson-huang/mirage-news", split="train")
    valid_data = load_dataset("anson-huang/mirage-news", split="validation")
    #test_data  = load_dataset("anson-huang/mirage-news", split="test")

    #data = load_dataset("anson-huang/mirage-news")
    #df_train = data["train"].to_pandas()
    df_train = data.to_pandas()

    # Insert the list of indexes of the few-shot examples TODO: add an argument for that
    #example_index_list = [2, 5004, 78, 5065] # Example order in the prompt: Fake - True - Fake - True
    example_index_list = args.examples

    # TODO: CHANGE THE WITH OPEN() FROM ARGS TO A VAR AND DETERMINE IT WITH A ARGS BOOLEAN
    promptfile = args.prompt_file
    promptfile_nolabel = "prompt01_labelonly"
    if args.prompt_label_only:
        fewshot_list = [load_sample(args, df_train.iloc[i - 1], data, i - 1, promptfile_nolabel) for i in example_index_list]
    else:
        fewshot_list = [load_sample(args, df_train.iloc[i - 1], data, i - 1, promptfile) for i in example_index_list]

    input_list = []

    # Prepare N samples from the train set as input
    if args.data_split == "train":
        samples = df_train.sample(n=args.n_samples, # Sample of N entries from df_train
                                  random_state=2)
        for index, row in samples.iterrows():
            input_list.append(load_sample(args, row, data, index, promptfile))

    # Prepare the whole validation set as input
    elif args.data_split == "validate":
        for index, row in valid_data.to_pandas().iterrows():
            input_list.append(load_sample(args, row, valid_data, index, promptfile))

    # print(f"INPUT LENGTH LIST: {len(input_list)}, OUTPUT: {input_list}")
    print(f"INPUT LENGTH LIST: {len(input_list)}")
    print(f"FEWSHOT LENGTH LIST: {len(fewshot_list)}, OUTPUT: {fewshot_list}")

    now = datetime.now() # current date and time for the output file name
    date_time = now.strftime("%H%M%S_%d_%m_%Y")
    i_date_time = "_".join([str(i) for i in example_index_list]) + f"_{date_time}"

    # Create the prompt and run the models
    if args.lvlm == "all":
        for model in model_list:
            # run_lvlm(model, args, sample1_dict, sample2_dict, input_list, date_time)
            run_lvlm(model, args, fewshot_list, input_list, i_date_time)

    else:
        # run_lvlm(args.lvlm, args, sample1_dict, sample2_dict, input_list, date_time)
        run_lvlm(args.lvlm, args, fewshot_list, input_list, i_date_time)


if __name__ == "__main__":
    main()


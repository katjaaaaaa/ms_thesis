# Change the cache location
import os
os.environ['TRANSFORMERS_CACHE'] = "/scratch/s4790383/.cache"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

import argparse
import json
import spacy
from collections import defaultdict
import re
from statistics import fmean
from bert_score import BERTScorer
from sacrebleu import corpus_bleu
from sklearn.metrics import classification_report
import numpy as np
from rouge_score import rouge_scorer
import sys
from copy import deepcopy
from sklearn.metrics import confusion_matrix


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # Input data arguments
    parser.add_argument("-of", "--output_file", type=str, nargs="+", help="Choose which specific output files to process")
    parser.add_argument("-p", "--prompt", type=str, choices=["zero-shot","two-shot", "four-shot", "six-shot"], help="Insert the type of the prompt that must be checked")

    # Final test analysis related arguments
    parser.add_argument("-t", "--test", action="store_true", help="Prepare the provided output file for test analysis")
    parser.add_argument("-tm", "--test_model", type=str, choices=["qwen", "llava", "idefics"], help="Provide the model necessary for the test analysis")

    # Types of analysises
    parser.add_argument("-r", "--render_check", action="store_true", 
                        help="Indicate whether the output json must be checked on the subject of the answer finished rendering")
    parser.add_argument("-c", "--concreteness", action="store_true", help="Calculates the concreteness level per output")
    parser.add_argument("-n", "--negation", action="store_true", help="Calculates the average amount of negation used per output")
    parser.add_argument("-s", "--similarity", action="store_true", help="Calculates the BLEU and BERTscores of the model output")

    args = parser.parse_args()
    return args


def nested_dict():
    return defaultdict(nested_dict)


def text_preprocess(s, nlp):
    '''Takes a string of text, removes punctuation, 
    lowercases the rest and returns as a new strings'''
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)

    # UNCOMMENT TO USE BLEU AND ROUGE-L W/O STOP-WORDS

    # s_new = []
    # stopwords = nlp.Defaults.stop_words

    # for token in s.split():
    #     if token not in stopwords:
    #         s_new.append(token)

    # return " ".join(s)
    return s


def calculate_negation(data_dict, nlp):

    '''
    Takes data dictionary and spacy processor,
    counts negation dependency per model explanation and
    saves them in the data dictionary, then returns it
    '''

    neg_words = {"never", "no", "nothing", "nowhere", "noone", "none"}
    neg_avg = []

    for _, data in data_dict.items():
        for k in ("zero-shot", "few-shot"):
            if k not in data:
                continue
            for model, answers in data[k].copy().items():
                if isinstance(answers, dict):
                    neg_counter = 0
                    model_answer = answers["model_explanation"]
                    doc = nlp(model_answer)
                    for token in doc:
                        if token.dep_ == "neg" or token in neg_words:
                            neg_counter += 1
                    neg_avg_doc = neg_counter / len(doc)
                    answers["avg_negation"] = neg_avg_doc
                    neg_avg.append(neg_avg_doc)

    print(f"NEG AVERAGE: {np.mean(neg_avg)}")

    return data_dict


def sim_prompt(candidate, reference, bert, scores_list, nlp):

    # Remove punctuation and lower-case
    cand_processed = text_preprocess(candidate, nlp)
    ref_processed = text_preprocess(reference, nlp)

    # Calculate ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouge_scores = scorer.score(ref_processed, cand_processed)
    _, _, fmeasure = rouge_scores['rougeL']

    # Calculate BLEU
    bleu_scores = corpus_bleu([cand_processed], [[ref_processed]])

    # Calculate BERTScore
    _, _, f1 = bert.score([candidate], [reference])

    # print(f"ROUGE-L: {rouge_scores}\nCORPUS BLEU: {bleu_scores.score}\nBEERTSCORE: {f1}")

    # Add the scores
    scores_list[0].append(fmeasure)
    scores_list[1].append(bleu_scores.score)
    scores_list[2].append(float(f1))
    return fmeasure, bleu_scores.score, float(f1), scores_list


def calculate_similarity(args, data_dict, bertscorer, nlp):

    prompt_scores_list, caption_scores_list  = [[],[],[]], [[],[],[]]
    data_args = data_dict["args"]

    # Load the few-shot examples
    if data_args["prompt"] == "few-shot" and "prompt_label_only" not in data_args:
        examples_id = data_args["examples"]
        fewshot_scores_list = [[list() for g in range(3)] for i in range(len(examples_id))]
        with open("answer_pipe.json") as f:
            answer_dict = json.load(f)

    # Load the prompt
    with open("prompt01.txt") as f:
        prompt_base = f.read()

    for entry, data in data_dict.items():
        #print(entry)
        for k in ("zero-shot", "few-shot"):
            if k not in data:
                continue
            caption = data["caption"]
            for model, answers in data[k].copy().items():
                if isinstance(answers, dict) and model == "llava-hf/llava-v1.6-mistral-7b-hf":
                    model_expl = answers["model_explanation"]
                    
                    rp, blp, bep, prompt_scores_list = sim_prompt(model_expl, prompt_base, bertscorer, prompt_scores_list, nlp)
                    rc, blc, bec, caption_scores_list = sim_prompt(model_expl, caption, bertscorer, caption_scores_list, nlp)

                    answers["prompt_sim_metrics"] = (rp, blp, bep)
                    answers["caption_sim_metrics"] = (rc, blc, bec)

                    if data_args["prompt"] == "few-shot" and "prompt_label_only" not in data_args:
                        answers["fewshot_sim_metrics"] = answers.get("fewshot_sim_metrics", [])
                        for i, id_num in enumerate(examples_id):
                            fewshot_expl = answer_dict[str(id_num)]["text"]
                            rf, blf, bef, fewshot_scores_list[i] = sim_prompt(model_expl, fewshot_expl, bertscorer, fewshot_scores_list[i], nlp)
                        answers["fewshot_sim_metrics"].append((rf, blf, bef))

    metrics = dict()
    metrics["prompt"] = prompt_scores_list
    metrics["caption"] = caption_scores_list
    if data_args["prompt"] == "few-shot" and "prompt_label_only" not in data_args:
        metrics["fewshot"] = fewshot_scores_list

    for i, score in enumerate(["ROUGE-L", "CORPUS-BLEU", "BERTSCORE"]):
        print()
        print(f"PROMPT {score} MEAN: {np.mean(prompt_scores_list[i])} MAX: {np.max(prompt_scores_list[i])}")
        print(f"CAPTION {score} MEAN: {np.mean(caption_scores_list[i])} MAX: {np.max(caption_scores_list[i])}")
        if data_args["prompt"] == "few-shot" and "prompt_label_only" not in data_args:
            for g, entry in enumerate(fewshot_scores_list):
                print(f"FEWSHOT ENTRY {examples_id[g]} {score} MEAN: {np.mean(entry[i])} MAX: {np.max(entry[i])}")

    return data_dict, metrics

def calculate_concreteness(data_dict, nlp):

    # accepted_tags = ["R", "J", "V", "N"]
    accepted_tags = {"R" : "Adverb",
                     "J" : "Adjective",
                     "V" : "Verb",
                     "N" : "Noun" }

    print("Start the search...")
    for _, data in data_dict.items():
        for k in ("zero-shot", "few-shot"):
            if k not in data:
                continue
            for _, answers in data[k].copy().items():
                if not isinstance(answers, dict):
                    # data[k]["concreteness_words"] = None
                    continue
                answers["concreteness_words"] = []
                model_answer = answers["model_explanation"]
                doc = nlp(model_answer) # Parse the model output thru spaCy
                for token in doc:
                    # Store only the words with accepted tags
                    if token.tag_[0] in accepted_tags.keys(): # or token.tag_ == "IN": (we do not use 2-word phrases so IN is not needed)
                        if token.tag_[:3] != "NNP":
                            token_data = [0] * 9 # Generate a list of 9 entries to match the MDT40K training data format
                            token_data[0] = token.lemma_
                            token_data[8] = accepted_tags[token.tag_[0]]
                            # Ensure the words do not get repeated
                            if token_data not in answers["concreteness_words"]:
                                # feature_data = get_concreteness_feature(token_data)
                                answers["concreteness_words"].append(token_data)
                # print(answers)

    print("Search completed! Saving in the new file...")

    return data_dict


def count_faulty_output(args, out_dict):

    # Open an output file if exists 
    try:
        with open(f"faulty_output.json") as f:
            error_dict = json.load(f)
    except FileNotFoundError:
        error_dict = nested_dict()


    error_dict = nested_dict() # Dict for storing the errors in output and classification
    f1_dict = nested_dict() # Dict for storing classification results

    f1_best = nested_dict()
    f1_best_all = 0
    model_best = None
    # model_best_all = nested_dict()

    error_counter = dict()
    for filename, data_dict in out_dict.items():
        print(f"FILENAME: {filename}")
        print(f"ENTRIES IN TOTAL: {len(data_dict) - 1}") # Exclude the dict entry with input arguments
        error_counter[filename] = nested_dict()

        for entry, data in data_dict.items():
            for k in ("zero-shot", "few-shot"):
                if k not in data:
                    continue
                true_label = data["true_label"]

                for model, answers in data[k].items():
                    # Count how many entries the model failed to output correctly
                    if not isinstance(answers, dict):
                        # determine error message
                        error_counter[filename][k][model]["render_error"] = error_counter[filename][k][model].get("render_error", 0)
                        error_counter[filename][k][model]["exceed_error"] = error_counter[filename][k][model].get("exceed_error", 0)

                        if "}" in answers:
                            # TODO: DELETE IF IT WORKS IN THE OG CODE
                            # pattern = r'\s*{\s*"model_label"\s*:\s*(\d+).*?"model_explanation":\s*"(.*?)".*?\s*}\s*'
                            pattern = r'{\s*"model_label"\s*:\s*(\d+)\s*,\s*"model_explanation"\s*:\s*"((?:[^"\\]|\\.|")*?)"[^\w]*}'
                            match = re.search(pattern, answers)
                            if match:
                                model_label = int(match.group(1))
                                model_explanation = match.group(2)
                                result = {
                                    "model_label": model_label,
                                    "model_explanation": model_explanation
                                }
                                #print(json.dumps(result, indent=2))
                                data[k] = result
                                # print(data[k])
                            else:
                                print(f"ERROR: {answers}")
                                error_msg = "JSON rendering error"
                                error_counter[filename][k][model]["render_error"] += 1
                        else:
                            error_msg = "Max number of generated tokens exceeded"
                            error_counter[filename][k][model]["exceed_error"] += 1

                        if k == "zero-shot":
                            error_dict[model][entry][k][filename]["error"] = error_msg
                            error_dict[model][entry][k][filename]["model_output"] = answers
                        elif k == "few-shot":
                            examples = "_".join([str(i) for i in data_dict["args"]["examples"]]) # Get example entries used in few-shot
                            error_dict[model][entry][k][examples][filename]["error"] = error_msg
                            error_dict[model][entry][k][examples][filename]["model_output"] = answers

                    # Count how many entries the model got right
                    else:
                        error_counter[filename][k][model]["y_pred"] = error_counter[filename][k][model].get("y_pred", [])
                        error_counter[filename][k][model]["y_true"] = error_counter[filename][k][model].get("y_true", [])

                        # Collect true and predicted values
                        error_counter[filename][k][model]["y_pred"].append(int(answers["model_label"]))
                        error_counter[filename][k][model]["y_true"].append(int(true_label))


        for prompt, data in error_counter[filename].items():
            print(f"PROMPT TYPE: {prompt}")
            for model, val in data.items():

                report = classification_report(val["y_true"], val["y_pred"], output_dict=True)

                # Save classification results in a dictionary
                f1_dict[filename][k][model] = [report["macro avg"]["precision"],
                                               report["macro avg"]["recall"],
                                               report["macro avg"]["f1-score"],
                                               report["accuracy"],
                                               report["macro avg"]["support"]
                                               ]

                f1_best[model] = f1_best.get(model, [0, ""])
                # Search for the highest f1-score
                if report["macro avg"]["f1-score"] > f1_best[model][0]:
                    if f1_best_all < report["macro avg"]["f1-score"]:
                        f1_best_all = report["macro avg"]["f1-score"]
                        model_best = { filename : { model : f1_dict[filename][k][model]}}
                    # Search for the model best
                    f1_best[model] = [report["macro avg"]["f1-score"], filename]

                # Print the results
                print(f" MODEL: {model}")
                print(classification_report(val["y_true"], val["y_pred"])) # get the full report 
                print(f"    JSON RENDERING ERROR: {val['render_error']} \n    TOKENS EXCEEDED ERROR: {val['exceed_error']}")
                print("----------------------------")
                print()

    # Save the results in a JSON file
    with open(f"./test/f1_results_{args.prompt}.json", "w") as f:
        json.dump(f1_dict, f, indent=2)

    # Save the best result in a different JSON file
    try:
        with open(f"./test/best_scores.json") as f:
            best_dict = json.load(f)
    except FileNotFoundError:
        best_dict = dict()

    best_dict[args.prompt] = model_best
    with open(f"./test/best_scores.json", "w") as f:
        json.dump(best_dict, f, indent=2)

    # Print best results per model
    for model, data in f1_best.items():
        print(f"MODEL {model} ({data[1]}) HIGHEST F1: {data[0]}")

    # Print best results per prompt type
    print("----------------------------")
    print(f"BEST RESULTS: {model_best}")
    print()

    return out_dict


def split_data_confuse(args, output_dict):

    '''
    Takes a data dictionary, splits the entries based on the
    true and generated label similarity and returns a new
    dictionary with keys as confusion matrix labels
    '''

    # Create a new dictionary sorted by label correctness
    for fname, data_dict in output_dict.items():
        splitted_dict = dict()
        pos_neg_labels = ["true_negative", "false_positive", "false_negative", "true_positive"]
        for val in pos_neg_labels:
            splitted_dict[val] = nested_dict()
            splitted_dict[val]["args"] = data_dict["args"]

        for entry, data in data_dict.items():
            for k in ("zero-shot", "few-shot"):
                if k not in data:
                    continue
                for model, answers in data[k].copy().items():
                    if not isinstance(answers, dict) or args.test_model not in model:
                        continue
                    true_label = data["true_label"]
                    tn, fp, fn, tp = confusion_matrix([true_label], [answers["model_label"]], labels=[1, 0]).ravel().tolist()
                    for i, val in enumerate([tn, fp, fn, tp]):
                        if val == 1:
                            splitted_dict[pos_neg_labels[i]][entry]["caption"] = data["caption"]
                            splitted_dict[pos_neg_labels[i]][entry]["true_label"] = true_label
                            splitted_dict[pos_neg_labels[i]][entry][k][model] = answers

        output_dict = deepcopy(splitted_dict)
        with open(f"aaaaaaaaaaaaa.json", "w") as f:
            json.dump(output_dict, f, indent=3)

    return output_dict


def get_data(args):
    output_dict = {}

    if args.output_file is not None:
        for fname in args.output_file:
            with open(f"{fname}") as f:
                data_dict = json.load(f)
                output_dict[fname] = data_dict
    
    elif args.prompt is not None:
        for filename in os.listdir(f"./test/{args.prompt}"):
            with open(os.path.join(f"./test/{args.prompt}", filename)) as f:
                data_dict = json.load(f)
                output_dict[filename] = data_dict

    else:
        sys.exit("No valid input files provided, refer to -h to see the possible options. Aborting the program...")

    return output_dict


def main():
    args = create_arg_parser()
    output_dict = get_data(args)
    fname = None
    # Load spacy parser
    nlp = spacy.load("en_core_web_trf")

    output_name = f"./test/test_output/final/"

    if args.test:
        #os.chdir(f"{os.getcwd()}/test")
        if args.test_model == None: sys.exit("Insert the model name to analyze. Aborting the program...")
        output_dict = split_data_confuse(args, output_dict)

    if args.render_check:
        output_dict = count_faulty_output(args, output_dict)
    

    if args.similarity:
        bertscorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')
        output_name += "similarity_"
        for fname, data_dict in output_dict.items():
            print(fname)
            data_dict, metrics = calculate_similarity(args, data_dict, bertscorer, nlp)
            output_dict[fname] = data_dict
            with open(f"./test/test_output/metrics_{fname}.json", "w") as f:
                json.dump(metrics, f, indent=2)

            with open(f"{output_name}{fname}.json", "w") as f:
                json.dump(data_dict, f, indent=2)
            print("*********************************")
    if args.negation:
        output_name += "negation_"
        for fname, data_dict in output_dict.items():
            print("Calculating negation...")
            print(f"FILENAME: {fname}")
            data_dict = calculate_negation(data_dict, nlp)
            output_dict[fname] = data_dict
            print("Calculation completed! Saving in the new file...")
            with open(f"{output_name}{fname}.json", "w") as f:
                json.dump(data_dict, f, indent=2)
    if args.concreteness:
        output_name += "features_"
        for fname, data_dict in output_dict.copy().items():
            data_dict = calculate_concreteness(data_dict, nlp)
            output_dict[fname] = data_dict
            with open(f"{output_name}{fname}.json", "w") as f:
                json.dump(data_dict, f, indent=2)

    # for fname, data_dict in output_dict.items():
    #     with open(f"./test/test_output/{fname}_updated.json", "w") as f:
    #         json.dump(data_dict, f, indent=2)

if __name__ == "__main__":
    main()
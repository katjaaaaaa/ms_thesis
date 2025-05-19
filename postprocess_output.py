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
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def TODO():
    # 1. CHECK IF THE OUTPUT HAS FINISHED RENDERING
        # - yes but fucked up formatting -> convert it back to normal (DO IT IN THE RUNNING SCRIPT)
        # - no finishing } present -> store it per MODEL then example
    # 2. CHECK CHANGE OF LABELS IN FEWSHOT BASED ON EXAMPLES
        # - track the examples used in the filename
        # - if first label was 0 are there more 0s than in "1 first" fi
    # 3. CODE FOR OUTPUT SIMILARITIES WITH BLEU
        # - similarity caption <-> output (zeroshot-fewshot)
        # - similarity output1 <-> output2 based on examples (fewshot only)
    # 3. CODE FOR CONCRETENESS
        # - load the model
        # - understand how this shit works
    # 4. CODE FOR NEGBERT 
        # - ???
    pass


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # GENERAL STUFF
    parser.add_argument("-of", "--output_file", type=str, nargs="+", help="Choose which output files to process")
    parser.add_argument("-r", "--render_check", action="store_true", 
                        help="Indicate whether the output json must be checked on the subject of the answer finished rendering")
    parser.add_argument("-lt", "--label_tracking", action="store_true", 
                        help="Indicate whether a label tracking must be performed on the few-shot answers")
    parser.add_argument("-c", "--concreteness", action="store_true", help="Calculates the concreteness level per output")
    parser.add_argument("-s", "--specificity", action="store_true", help="Calculates the BLEU and BERTscores of the model output")

    args = parser.parse_args()
    return args


def nested_dict():
    return defaultdict(nested_dict)


def find_2w_phrases(text, nlp, words_list):

    accepted_tags = ["R", "J", "V", "N"]

    for token in text:
        if token.tag_[0] != "N" or token.tag_[0] != "V":
            continue

        # Search for the children of noun and verb phrases
        for child in token.children:
            if child.tag_[0] in accepted_tags or child.tag_ == "IN":
                if token.tag_[0] == "V":
                    v_lemma = token.lemma_


def bleu_preprocess(s):
    '''Takes a string of text, removes punctuation, 
    lowercases the rest and returns as a list of strings'''
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)

    return s.split()


def calculate_specificty(data_dict):
    bertscorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')
    # bleuscorer = ...

    scores_counter = nested_dict()

    for _, data in data_dict.items():
        for k in ("zero-shot", "few-shot"):
            if k not in data:
                continue
            caption = data["caption"]
            for model, answers in data[k].copy().items():
                if isinstance(answers, dict):
                    model_expl = answers["model_explanation"]

                    ref_bleu = bleu_preprocess(caption)
                    bleu_score_list = list()
                    bertscore_list = list()

                    for candidate in model_expl.split(". "):
                        # Calculate the BLEU score per sentence
                        candidate_bleu = bleu_preprocess(candidate)
                        bleu_score_list.append(sentence_bleu(ref_bleu, candidate_bleu))

                        # Calculate the BERTscore per sentence
                        p, r, f1 = bertscorer.score([candidate], [caption])
                        bertscore_list.append([p.mean(), r.mean(), f1.mean()])

                        # print(f"BLEU SENTENCE: {sentence_bleu(ref_bleu, candidate_bleu)}")
                        # print(f"BERT SENTENCE F1: {f1}")

                    # Take the mean of all scores to represent the output scores
                    bleu_final = fmean(bleu_score_list)
                    bert_final = np.mean(bertscore_list, axis=0)
                    # print(f"BLEU FINAL: {bleu_final}")
                    # print(f"BERTSCORE FINAL: {bert_final}")

                    # Add the scores to the output dictionary
                    answers["BLEU"] = bleu_final
                    answers["BERTScore"] = list(bert_final)

                    # Store the scores for the model statistics
                    scores_counter[model][k]["BLEU"] = scores_counter[model][k].get("BLEU", [])
                    scores_counter[model][k]["BLEU"].append(bleu_final)
                    scores_counter[model][k]["BERTscore"] = scores_counter[model][k].get("BERTscore", [])
                    scores_counter[model][k]["BERTscore"].append(bert_final)

    for model, data in scores_counter.items():
        print(f"MODEL: {model}")
        for k, scores in data.items():
            print(f"{k.upper()}")
            for metric, v in scores.items():
                print()
                print(f"    {metric} MEAN: {np.mean(v, axis=0)}") # TODO: BERT [p,r,f1] needs a diff mean than BLEU
                print(f"    {metric} MAX: {np.max(v, axis=0)}")
                print(f"    {metric} MIN: {np.min(v, axis=0)}")
        print("#########################")


def calculate_concreteness(data_dict):

    # Load spacy parser
    nlp = spacy.load("en_core_web_trf")
    accepted_tags = ["R", "J", "V", "N"]

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
                    if token.tag_[0] in accepted_tags: # or token.tag_ == "IN": (we do not use 2-word phrases so IN is not needed)
                        if token.tag_[:3] != "NNP":
                            token_data = [0] * 9 # Generate a list of 9 entries to match the MDT40K training data
                            token_data[0] = token.lemma_
                            # Ensure the words do not get repeated
                            if token_data not in answers["concreteness_words"]:
                                # feature_data = get_concreteness_feature(token_data)
                                answers["concreteness_words"].append(token_data)
                print(answers)

    return data_dict


def count_faulty_output(out_dict):

    # Open an output file if exists 
    try:
        with open(f"faulty_output.json") as f:
            error_dict = json.load(f)
    except FileNotFoundError:
        error_dict = nested_dict()
    error_dict = nested_dict()

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
                                #print(answers)
                                model_label = int(match.group(1))
                                model_explanation = match.group(2)
                                result = {
                                    "model_label": model_label,
                                    "model_explanation": model_explanation
                                }
                                #print(json.dumps(result, indent=2))
                                data[k] = result
                                print(data[k])
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
                        error_counter[filename][k][model]["entries_correct"] = error_counter[filename][k][model].get("entries_correct", 0)
                        error_counter[filename][k][model]["entries_wrong"] = error_counter[filename][k][model].get("entries_wrong", 0)
                        if int(true_label) == int(answers["model_label"]):
                            error_counter[filename][k][model]["entries_correct"] += 1
                        else:
                            error_counter[filename][k][model]["entries_wrong"] += 1


        for prompt, data in error_counter[filename].items():
            print(f"PROMPT TYPE: {prompt}")
            for model, val in data.items():
                print(f" MODEL: {model}")
                print(f"    CORRECT PREDICTIONS: {val['entries_correct']} \n    WRONG PREDICTIONS: {val['entries_wrong']}\n")
                print(f"    JSON RENDERING ERROR: {val['render_error']} \n    TOKENS EXCEEDED ERROR: {val['exceed_error']}")
                print("----------------------------")

        # with open(f"faulty_output_{filename}.json", "w") as f:
        #         json.dump(error_dict, f, indent=4)

    return out_dict


def get_data(args):
    output_dict = {}

    for fname in args.output_file:
        with open(f"{fname}.json") as f:
            data_dict = json.load(f)
            output_dict[fname] = data_dict

    return output_dict


def main():
    args = create_arg_parser()
    output_dict = get_data(args)
    print(args)
    if args.render_check:
        output_dict = count_faulty_output(output_dict)
    if args.concreteness:
        for fname, data_dict in output_dict.items():
            data_dict = calculate_concreteness(data_dict)
            with open(f"{args.fname}_features.json", "w") as f:
                json.dump(data_dict, f, indent=3)
    if args.specificity:
        for fname, data_dict in output_dict.items():
            data_dict = calculate_specificty(data_dict)


if __name__ == "__main__":
    main()
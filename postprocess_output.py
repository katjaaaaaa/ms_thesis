# Filename: postprocess_output.py
# Date: 19/07/2025
# Author: Katja Kamyshanova
# The program takes a single or multiple model output files and utilizes several types of postprocess analyses, such as classification results evaluation, 
# input similarity, calculation of negation and preparing the output for concreteness calculation. The program operates with the argument parser

# Change the cache location
import os
os.environ['TRANSFORMERS_CACHE'] = "/scratch/s4790383/.cache" # Used instead of HF_HOME as HF_HOME did not work
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

import argparse
import json
import spacy
from collections import defaultdict
import re
from bert_score import BERTScorer
from sacrebleu import sentence_bleu
from sklearn.metrics import classification_report
import numpy as np
from nltk import word_tokenize
from rouge_score import rouge_scorer
import sys
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import random
import pandas as pd
import string


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # Input data arguments
    parser.add_argument("-of", "--output_file", type=str, nargs="+", help="Choose which specific output files to process")
    parser.add_argument("-p", "--prompt", type=str, choices=["zero-shot","two-shot", "four-shot", "six-shot"], help="Insert the type of the prompt that must be checked")

    # Final test analysis related arguments
    parser.add_argument("-t", "--test", action="store_true", help="Prepare the provided output file for test analysis")
    parser.add_argument("-tm", "--test_model", type=str, choices=["Qwen", "llava", "Idefics"], help="Provide the model necessary for the test analysis")

    # Types of analysises
    parser.add_argument("-r", "--render_check", action="store_true",help="Indicate whether the output json must be checked on the subject of the answer finished rendering")
    parser.add_argument("-c", "--concreteness", action="store_true", help="Calculates the concreteness level per output")
    parser.add_argument("-n", "--negation", action="store_true", help="Calculates the average amount of negation used per output")
    parser.add_argument("-s", "--similarity", action="store_true", help="Calculates the BLEU and BERTscores of the model output")
    parser.add_argument("-sw", "--stop_words", action="store_true", help="Define whether remove stop-words during calculation of text similarity")

    # Arguments for additional functions
    parser.add_argument("-ml", "--mean_length", action="store_true", help="Calculates the mean length of the generated explanation per requested model output")
    parser.add_argument("-ra", "--random_entry", action="store_true", help="Provides 5 random entries from each document for further error analysis")

    args = parser.parse_args()
    return args


def nested_dict():
    return defaultdict(nested_dict)


def text_preprocess(args, s, nlp):

    '''Takes a string of text, removes punctuation, 
    lowercases the rest and returns as a new strings'''

    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)

    # Remove stop-words
    if args.stop_words:
        s_new = []
        stopwords = nlp.Defaults.stop_words

        for token in s.split():
            if token not in stopwords:
                s_new.append(token)


        return " ".join(s_new)

    return s


def calculate_negation(args, data_dict, nlp):

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
                if isinstance(answers, dict) and args.test_model in model:
                    neg_counter = 0
                    model_answer = answers["model_explanation"]
                    doc = nlp(model_answer)
                    for token in doc:
                        if token.dep_ == "neg" or token in neg_words:
                            neg_counter += 1
                    model_answer = model_answer.translate(str.maketrans('', '',string.punctuation))
                    neg_avg_doc = neg_counter / len(word_tokenize(model_answer))
                    answers["avg_negation"] = neg_avg_doc
                    neg_avg.append(neg_avg_doc)

    print(f"NEG AVERAGE: {np.mean(neg_avg)}")

    return data_dict


def get_sim_score(args, candidate, reference, bert, scores_list, nlp):

    '''
    Takes a candidate and a reference as strings, calculates the 
    SBLEU, ROUGE-L and BertScore for the given strings, adds them
    to the list and returns it alongside with individual float scores
    '''

    # Remove punctuation and lower-case
    cand_processed = text_preprocess(args, candidate, nlp)
    ref_processed = text_preprocess(args, reference, nlp)

    # Calculate ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouge_scores = scorer.score(ref_processed, cand_processed)
    _, _, fmeasure = rouge_scores['rougeL']

    # Calculate Sentence BLEU
    bleu_scores = sentence_bleu(cand_processed, [ref_processed])

    # Calculate BERTScore
    if args.stop_words:
        _, _, f1 = bert.score([cand_processed], [ref_processed])
    else:
        _, _, f1 = bert.score([candidate], [reference])

    # Add the scores
    scores_list[0].append(bleu_scores.score)
    scores_list[1].append(fmeasure)
    scores_list[2].append(float(f1))
    return fmeasure, bleu_scores.score, float(f1), scores_list


def calculate_similarity(args, data_dict, bertscorer, nlp):

    '''
    Loops through the model output dictionary, calculates the
    similarity between the generated explanations and input
    dependent on the type of model experiment (PLO/PLE),
    then prints the average similarity scores
    '''

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

    # Calculate similarity
    for entry, data in data_dict.items():
        for k in ("zero-shot", "few-shot"):
            if k not in data:
                continue
            caption = data["caption"]
            for model, answers in data[k].copy().items():
                if isinstance(answers, dict) and args.test_model in model:
                    model_expl = answers["model_explanation"]
                    
                    rp, blp, bep, prompt_scores_list = get_sim_score(args, model_expl, prompt_base, bertscorer, prompt_scores_list, nlp)
                    rc, blc, bec, caption_scores_list = get_sim_score(args, model_expl, caption, bertscorer, caption_scores_list, nlp)

                    answers["prompt_sim_metrics"] = (blp, rp, bep)
                    answers["caption_sim_metrics"] = (blc, rc, bec)

                    if data_args["prompt"] == "few-shot" and "prompt_label_only" not in data_args:
                        answers["fewshot_sim_metrics"] = answers.get("fewshot_sim_metrics", [])
                        for i, id_num in enumerate(examples_id):
                            fewshot_expl = answer_dict[str(id_num)]["text"]
                            rf, blf, bef, fewshot_scores_list[i] = get_sim_score(args, model_expl, fewshot_expl, bertscorer, fewshot_scores_list[i], nlp)
                        answers["fewshot_sim_metrics"].append((blf, rf, bef))

    metrics = dict()
    metrics["prompt"] = prompt_scores_list
    metrics["caption"] = caption_scores_list
    if data_args["prompt"] == "few-shot" and "prompt_label_only" not in data_args:
        metrics["fewshot"] = fewshot_scores_list

    # Print the average similarities per input type
    for i, score in enumerate(["SBLEU", "ROUGE-L", "BERTSCORE"]):
        print()
        print(f"PROMPT {score} MEAN: {round(np.mean(prompt_scores_list[i]), 4)} MIN: {round(np.min(prompt_scores_list[i]), 4)} MAX: {round(np.max(prompt_scores_list[i]), 4)} SD: {round(np.std(prompt_scores_list[i]), 4)}")
        print(f"CAPTION {score} MEAN: {round(np.mean(caption_scores_list[i]), 4)} MIN: {round(np.min(caption_scores_list[i]), 4)} MAX: {round(np.max(caption_scores_list[i]), 4)} SD: {round(np.std(caption_scores_list[i]), 4)}")
        if data_args["prompt"] == "few-shot" and "prompt_label_only" not in data_args:
            for g, entry in enumerate(fewshot_scores_list):
                print(f"FEWSHOT ENTRY {examples_id[g]} {score} MEAN: {round(np.mean(entry[i]), 4)} MIN: {round(np.min(entry[i]), 4)} MAX: {round(np.max(entry[i]), 4)} SD: {round(np.std(entry[i]), 4)}")

    return data_dict, metrics


def calculate_concreteness(data_dict, nlp):

    '''
    Loops through the model output dictionary, extracts with SpaCy
    words from explanations with requested POS tags,
    stores them in a list as a new dictionary pair,
    and returns the updated dictionary
    '''

    accepted_tags = {"J" : "Adjective",
                     "V" : "Verb",
                     "N" : "Noun" ,
                     # "R" : "Adverb", While adverbs can be classified by the concreteness model, we omitted them in current research
                     }

    print("Start the search...")
    for _, data in data_dict.items():
        for k in ("zero-shot", "few-shot"):
            if k not in data:
                continue
            for _, answers in data[k].copy().items():
                if not isinstance(answers, dict):
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
                                answers["concreteness_words"].append(token_data)

    print("Search completed! Saving in the new file...")

    return data_dict



def get_analysis(filename, analysis_data, examples, model, val, report):

    '''
    Gets the data necessary for the classification
    results analysis per experiment, stores it in
    a dictionary and appends to a list, then returns it
    '''


    # Determine the type of prompt
    if "plo" in filename:
        prompt_setting = "PLO"
    else:
        prompt_setting = "PLE"

    # Determine the order of few-shot examples
    if examples[0] >= 5000:
        shots_order = "true-false"
    else:
        shots_order = "false-true"

    # Store all data
    analysis_data.append({
            'file_name': filename,
            'prompt_setting': prompt_setting,
            'order': shots_order,
            'model': model.split("/")[1],
            'precision_0': report["0"]['precision'],
            'recall_0': report["0"]['recall'],
            'precision_1': report["1"]['precision'],
            'recall_1': report["1"]['recall'],
            'f1': report["macro avg"]['f1-score'],
            'exceed_error': val,
        })

    return analysis_data


def get_classification_results(args, out_dict):

    '''
    Loops through the model output dictionary,
    counts the occured generation errors, then
    prints the classification result analyses
    '''

    analysis_data = [] # List for storing the Error Analysis data
    error_dict = nested_dict() # Dict for storing the errors in output and classification
    f1_dict = nested_dict() # Dict for storing classification results

    f1_best = nested_dict()
    all_scores = []
    error_counter = dict()

    for filename, data_dict in out_dict.items():
        print(f"FILENAME: {filename}")
        print(f"ENTRIES IN TOTAL: {len(data_dict) - 1}") # Exclude the dict entry with input arguments
        error_counter[filename] = nested_dict()
        examples = data_dict["args"]["examples"]
        for entry, data in data_dict.items():
            for k in ("zero-shot", "few-shot"):
                if k not in data:
                    continue
                true_label = data["true_label"]

                for model, answers in data[k].items():
                    # Count how many entries the model failed to output correctly
                    error_counter[filename][k][model]["render_error"] = error_counter[filename][k][model].get("render_error", 0)
                    error_counter[filename][k][model]["exceed_error"] = error_counter[filename][k][model].get("exceed_error", 0)
                    if not isinstance(answers, dict):
                        if "}" in answers:
                            pattern = r'{\s*"model_label"\s*:\s*(\d+)\s*,\s*"model_explanation"\s*:\s*"((?:[^"\\]|\\.|")*?)"[^\w]*}'
                            match = re.search(pattern, answers)
                            if match:
                                continue
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
                            examples_str = "_".join([str(i) for i in examples]) # Get example entries used in few-shot
                            error_dict[model][entry][k][examples_str][filename]["error"] = error_msg
                            error_dict[model][entry][k][examples_str][filename]["model_output"] = answers

                    # Count how many entries the model got right
                    else:
                        error_counter[filename][k][model]["y_pred"] = error_counter[filename][k][model].get("y_pred", [])
                        error_counter[filename][k][model]["y_true"] = error_counter[filename][k][model].get("y_true", [])

                        # Collect true and predicted values
                        error_counter[filename][k][model]["y_pred"].append(int(answers["model_label"]))
                        error_counter[filename][k][model]["y_true"].append(int(true_label))

        # Get the classification results
        for _, data in error_counter[filename].items():
            for model, val in data.items():

                report = classification_report(val["y_true"], val["y_pred"], output_dict=True)
                analysis_data = get_analysis(filename, analysis_data, examples, model, val['exceed_error'], report)

                # Save classification results in a dictionary
                f1_dict[filename][k][model] = [report["macro avg"]["precision"],
                                               report["macro avg"]["recall"],
                                               report["macro avg"]["f1-score"],
                                               report["accuracy"],
                                               report["macro avg"]["support"]
                                               ]


                # Append score to the all_scores list
                all_scores.append((
                    report["macro avg"]["f1-score"], 
                    filename, 
                    model, 
                    f1_dict[filename][k][model]
                ))

                f1_best[model] = f1_best.get(model, [0, ""])

                # Search for the highest f1-score
                if report["macro avg"]["f1-score"] > f1_best[model][0]:
                    f1_best[model] = [report["macro avg"]["f1-score"], report["macro avg"]["precision"], report["macro avg"]["recall"], filename]

                # Print the results
                print(f" MODEL: {model}")
                print(f"    F1 {round(report['macro avg']['f1-score'], 3)}")
                #print(classification_report(val["y_true"], val["y_pred"])) # get the full report 
                print(f"    JSON RENDERING ERROR: {val['render_error']} \n    TOKENS EXCEEDED ERROR: {val['exceed_error']}")
                print("----------------------------")
                print()

    # Sort all scores by F1-score in descending order and get top 3
    top_3_overall = sorted(all_scores, key=lambda x: x[0], reverse=True)[:3]

    # Print best overal results
    for i, (score, fname, mdl, metrics) in enumerate(top_3_overall, 1):
        print(f"TOP {i} F1: {round(score, 4)}, FILE: {fname}, MODEL: {mdl}, ALL METRICS: {metrics}")
    print()
    # Print best results per model
    for model, data in f1_best.items():
        print(f"MODEL {model} ({data[3]}) HIGHEST F1: {round(data[0], 4)} PRECISION: {round(data[1], 4)} RECALL {round(data[2], 4)}")

    # Group the classification results by the prompt setting
    print()
    print("----------------------------")
    print("EFFECT OF PROMPT TYPE")
    analysis_df = pd.DataFrame(analysis_data)
    by_prompt = analysis_df.groupby(['prompt_setting', 'model']).agg({
                                                            'precision_0': 'mean',
                                                            'recall_0': 'mean',
                                                            'precision_1': 'mean',
                                                            'recall_1': 'mean',
                                                            'f1': 'mean',
                                                            'exceed_error': 'mean',
                                                            }).round(2)
    print(by_prompt)

    # Group the classification results by the example order
    print()
    print("----------------------------")
    print("EFFECT OF FEW-SHOT EXAMPLE ORDER")
    by_order = analysis_df.groupby(['order', 'model']).agg({
                                                            'precision_0': 'mean',
                                                            'recall_0': 'mean',
                                                            'precision_1': 'mean',
                                                            'recall_1': 'mean',
                                                            'f1': 'mean',
                                                            'exceed_error': 'mean',
                                                            }).round(2)
    print(by_order)
    if args.prompt is not None:
        by_prompt.reset_index().to_csv(f"./test_final/error-analysis/{args.prompt}_by_prompt.csv", sep=",", index=False)
        by_order.reset_index().to_csv(f"./test_final/error-analysis/{args.prompt}_by_order.csv", sep=",", index=False)

    return out_dict


def split_data_confuse(args, output_dict):

    '''
    Takes a data dictionary, splits the entries based on the
    true and generated label similarity and returns a new
    dictionary with keys as confusion matrix labels
    '''

    # Create a new dictionary sorted by label correctness
    for _, data_dict in output_dict.items():
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
                    if isinstance(answers, dict) and args.test_model in model:
                        true_label = data["true_label"]
                        tn, fp, fn, tp = confusion_matrix([true_label], [answers["model_label"]], labels=[1, 0]).ravel().tolist()
                        for i, val in enumerate([tn, fp, fn, tp]):
                            if val == 1:
                                splitted_dict[pos_neg_labels[i]][entry]["caption"] = data["caption"]
                                splitted_dict[pos_neg_labels[i]][entry]["true_label"] = true_label
                                splitted_dict[pos_neg_labels[i]][entry][k][model] = answers

        output_dict = deepcopy(splitted_dict)


    return output_dict


def count_expl_length(args, output_dict):

    '''
    Takes a dictionary of generated explanations by specific
    model, tokenizes each explanation and takes its
    length, then prints the average explanation length
    across the whole dictionary
    '''

    for filename, data_dict in output_dict.items():
        print(filename)
        slength_counter = []
        for entry, data in data_dict.items():
            for k in ("zero-shot", "few-shot"):
                if k not in data:
                    continue
                for model, answers in data[k].items():
                    if isinstance(answers, dict) and args.test_model in model:
                        answer_nopunct = answers["model_explanation"].translate(str.maketrans('', '',string.punctuation))
                        slength_counter.append(len(word_tokenize(answer_nopunct)))

        print(f"AVERAGE ANSWER LENGTH: {round(np.mean(slength_counter), 4)}")


def get_random_entries(args, output_dict):

    '''
    Takes the dictionaries and gets
    five random keys from each of them
    '''

    random_dict = dict()

    for name, data_dict in output_dict.items():
        random_list = random.sample(list(data_dict.items()), 5)
        print(f"FILE NAME: {name}")
        random_dict[name] = dict()
        for k, v in dict(random_list).items():
            k_list = k.split("_")
            entry_id = k_list.pop(0)
            dataset_name = "_".join(k_list)
            random_dict[name][dataset_name] = random_dict[name].get(dataset_name, dict())
            random_dict[name][dataset_name][entry_id] = dict()
            random_dict[name][dataset_name][entry_id]["caption"] = v["caption"]
            random_dict[name][dataset_name][entry_id]["true_label"] = v["true_label"]
            for k in ("zero-shot", "few-shot"):
                if k not in v:
                    continue
                for model, answers in v[k].items():
                    if args.test_model in model:
                        random_dict[name][dataset_name][entry_id]["model_label"] = answers["model_label"]
                        random_dict[name][dataset_name][entry_id]["model_explanation"] = answers["model_explanation"]

    with open(f"./test_final/error-analysis/random_entries.json", "w") as f:
                json.dump(random_dict, f, indent=2)
    print("Entries saved to JSON file")


def get_data(args):

    '''
    Loads either a single model output file or
    multiple files based on the type of shots,
    then returns as a dictionary
    '''

    output_dict = {}

    if args.output_file is not None:
        for fname in args.output_file:
            with open(f"{fname}") as f:
                name_only = fname.split("/")[-1].removesuffix(".json")
                data_dict = json.load(f)
                output_dict[name_only] = data_dict
    
    elif args.prompt is not None:
        for filename in os.listdir(f"./test_final/{args.prompt}"):
            with open(os.path.join(f"./test_final/{args.prompt}", filename)) as f:
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

    output_name = f"./test_final/best-model-final/"
    print(args)
    if args.test:
        if args.test_model == None: sys.exit("Insert the model name to analyze. Aborting the program...")
        output_dict = split_data_confuse(args, output_dict)
        
        if args.random_entry:
            get_random_entries(args, output_dict)

    if args.mean_length:
        count_expl_length(args, output_dict)

    if args.render_check:
        output_dict = get_classification_results(args, output_dict)

    if args.negation:
        output_name += "negation_"
        for fname, data_dict in output_dict.items():
            print("Calculating negation...")
            print(f"FILENAME: {fname}")
            data_dict = calculate_negation(args, data_dict, nlp)
            output_dict[fname] = data_dict
            print("Calculation completed! Saving in the new file...")
            with open(f"{output_name}{fname}.json", "w") as f:
                json.dump(data_dict, f, indent=2)
        print("***************************")

    if args.concreteness:
        print("Calculating concreteness...")
        output_name += "features_"
        for fname, data_dict in output_dict.copy().items():
            print(f"FILENAME: {fname}")
            data_dict = calculate_concreteness(data_dict, nlp)
            output_dict[fname] = data_dict
            print("Calculation completed! Saving in the new file...")
            with open(f"{output_name}{fname}.json", "w") as f:
                json.dump(data_dict, f, indent=2)
        print("***************************")

    if args.similarity:
        print("Calculating similarity...")
        # Load BertScorer
        bertscorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')
        output_name += "similarity_"
        for fname, data_dict in output_dict.items():
            print(fname)
            data_dict, metrics = calculate_similarity(args, data_dict, bertscorer, nlp)
            output_dict[fname] = data_dict

            # Save the similarity output
            with open(f"{output_name}metrics_{fname}.json", "w") as f:
                json.dump(metrics, f, indent=2)

            sys.stdout.flush()


if __name__ == "__main__":
    main()
# A Picture Worth a Thousand Lies: Challenging Visual Language Models to Detect and Explain Multi-modal Misinformation

This repository contains the code and data used in the experiment made for the Information Science programme master's thesis at the Groningen University. The study explores the ability of instruction-tuned Visual Language Models (VLM) to detect and explain multi-modal misinformation. Please refer to the paper for a full explanation of the approach.

Below, we elaborate on the contents of the repository, the order in which to run the experiment, and the input arguments required. 

## Contents

The repository contains the following components, grouped by the order of usage in the thesis:
- `get_models_running.py`: used to run the experiments with the VLMs and get each model's predicted labels and explanations
  - `./prompt/`: contains the prompts used as model input (PLO and PLE)
  - `./pre-made_answers/`: contains the pre-made answers used in the few-shot prompts (PLO and PLE)
- `postprocess_output.py`: used to run the postprocess analyses of the VLM output, including splitting the output file into the subsets, the classification results analysis, input similarity, negation, and preparing the input for the concreteness and specificity analyses. Additionally, it includes several other functions utilized in the thesis. Refer to the `create_arg_parser()` for further info
- `./test_final/`: contains the VLM and postprocessing output. Refer to the README file in that directory for further info
- `./concreteness/`: contains the Jupyter notebook, the regression model, and the input data necessary to calculate the concreteness of the model-generated explanations
- `./specificity/`: contains the Jupyter notebook and the input data necessary to calculate the specificity of the model-generated explanations

The winning experiment that was analysed in the thesis (Qwen2-VL six-shot PLO) is located in `./test_final/six-shot/few-shot_5065_154_7879_2_5004_78_071436_28_06_2025_plo_test.json`

## Running the experiments

We ran our experiment in a Python 3.11.13 virtual environment. Before running the programs, install the necessary packages with this command:

```
pip install -r requirements.txt
```
### 1. Generate the VLM output

We used Qwen2-VL, LLaVA-1.6, and Idefics3 to generate the misinformation label and explanations for the multi-modal input. The code allows for adjusting the settings of experiments, choosing between the data splits (train/validate/test), the models, number of examples, types of prompts (zero-shot/few-shot/both), settings (PLO/PLE), etc. For instance, the current command runs only Idefics3 with both zero-shot and few-shot PLE prompts, default examples (#5004 and #2), default pre-made answers file (`answer_pipe.json`) on 50 entries chosen randomly from the training data split:

```
python3 get_models_running.py -p both -ds train -n 50 -l idefics
```
Use the following command to generate the output on the test split for the experiment containing the winning misinformation detector, Qwen2-VL: 

```
python3 get_models_running.py -l all -p few-shot -ex 5065 154 7879 2 5004 78 -ds test -ap answer_pipe_labelonly -plo True
```
### 2. Postprocess analysis

All VLM outputs are stored in the `test_final` directory. From there, we can choose how to further work with the output: whether we want to analyze a single file, split it into subsets first (TP/FP/FN/TN), or analyze a group of experiments at once (zero-shot/two-shot/four-shot/six-shot). We used the groups to get insights into the classification results. For example, to get the classification result analysis for all six-shot experiments, use the following command:

```
python3 postprocess_output.py -p six-shot -r
```
After we identified the best-performing misinformation detector, we zoomed in on that specific model to perform further analyses, such as negation and input similarity. The following command calculates the negation and input similarity scores averaged across the whole document for Qwen2-VL-generated explanations only: 
```
python3 postprocess_output.py -of ./test_final/six-shot/few-shot_5065_154_7879_2_5004_78_071436_28_06_2025_plo_test.json -tm Qwen -n -s
```
To split the document into subsets based on the correctly predicted misinformation label and then perform the postprocess analyses, add the `-t` command: 
```
python3 postprocess_output.py -of ./test_final/six-shot/few-shot_5065_154_7879_2_5004_78_071436_28_06_2025_plo_test.json -tm Qwen -t -n -s
```
While the above-mentioned results per explanation are stored in the given input file, the averaged results are printed directly to the console. Therefore, consider printing them in a separate file to not lose the results. 

### 3. Calculate the concreteness
Because the initial concreteness code was provided in a Jupyter notebook, we decided to keep it separate from the overall postprocess analysis as well to provide the extra clarity on which code does what. 
1. Extract all nouns, adjectives, and verbs from each explanation and present them in a format suited for the regression model with the command below (add `-t` to do it per subset). The output filenames will begin with "features" and will be stored in `./test_final/best-model-final/`:
```
python3 postprocess_output.py -of ./test_final/six-shot/few-shot_5065_154_7879_2_5004_78_071436_28_06_2025_plo_test.json -tm Qwen -c
```
2. Transfer the output to the `./concreteness/data/` directory or use the data that we utilized in our experiment, which is already stored in the data directory
3. Extract the regression model stored in `model.zip` (We received this model file by training it in the notebook provided by Charbonnier and Wartena (2019). [Link to the notebook](https://colab.research.google.com/drive/14341vvHfrEX1W-JTUlUKKNatfkI8aH_W?usp=sharing))
4. Run the `calculate_concreteness.ipynb` notebook
> Note: We ran the concreteness and specificity notebooks using the Google Colab environment and stored all the data on the Google Drive. If you prefer to run the notebook elsewhere, consider changing the `HEAD_DIR` variable to a different path

> Note 2: The Charbonnier and Wartena (2019) notebook requires specific versions of `numpy` and `gensim`. If you are working with Google Colab and have installed the packages, it may request that you restart the environment to apply the changes to the packages. Some already installed packages may not support the dependency with the older versions of numpy. However, it does not have any effect on the packages necessary to run the notebook

### 4. Calculate the specificity
Similarly to concreteness, the specificity code is placed separately for the overall clarity of the code. The output of the concreteness code is required to calculate the specificity scores. 
1. Transfer the concreteness output files to `./specificity/data/` directory or use the data that we utilized in our experiment, which is already stored in the data directory
2. Run the `calculate_specificity.ipynb` notebook


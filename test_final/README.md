# VLM Generated Output

This directory contains the output from all 37 experiments and the further analyses performed on the best experiment. The experiments are sorted by the number of few-shot examples. Below, we included all the few-shot combinations used in the experiments. The letters correspond to the letters used in Table 3 in the thesis. The few-shot example combination that scored the highest on the classification task (Qwen2-VL PLO) __in bold__:

|Letter| Two-shot   | Four-shot | Six-shot |
|:---|:---:|:---:|:---:|
|A|5065 78|5065 2 7879 78|__5065 154 7879 2 5004 78__|
|B|7879 154|7879 154 5004 2	|7879 2 5004 154 5065 78|
|C|5004 2 |5004 78 7879 154|5004 78 5065 154 7879 2|
|D|78 5065 |78 5004 2 7879|78 5004 2 5065 154 7879|
|E|2 5004 |2 5065 154 7879|2 5065 154 7879 78 5004|
|F|154 7879 |154 5004 78 5065|154 7879 2 5004 78 5065|

Further, this directory contains the output of the post-process analyses made in `postprocess_output.py`: 
- `best-model-final` includes the output of the input similarity, negation, and the input made for the concreteness notebook.
- `error-analysis` contains the output of the classification analyses (grouped results by example order & by prompt setting), input similarity results, and 20 randomly chosen entries used in the qualitative analysis.

Lastly, the `to-merge` directory contains model outputs that could not be completed in a single run and, therefore, required multiple runs followed by merging. The final merged results are included in the designated few-shot directories, while the contents of `to-merge` are retained for administrative reference.

# CaselawQA

CaselawQA is a benchmark comprising legal classification tasks derived from the Supreme Court and Songer Court of Appeals databases.
The majority of its 10,000 questions are multiple-choice, with 5,000 sourced from each database. 
The questions are randomly selected from the test sets of the [Lawma tasks](https://huggingface.co/datasets/ricdomolm/lawma-tasks), and only cases with Court opinions of at least 2,000 characters are included.

CaselawQA also includes two additional subsets: CaselawQA Tiny and CaselawQA Hard. 
CaselawQA Tiny consists of 49 Lawma tasks with fewer than 150 training examples. 
CaselawQA Hard comprises tasks where [Lawma 70B](https://huggingface.co/ricdomolm/lawma-70b) achieves less than 70% accuracy.

To ensure input length limits, Court opinions are shortened so that the total input prompt (instruction, opinion, question, and answer choices) does not exceed 8,000 Llama 3 tokens.

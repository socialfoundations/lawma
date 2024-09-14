# CaselawQA

CaselawQA is a benchmark comprising legal classification tasks derived from the Supreme Court and Songer Court of Appeals databases.
The majority of its 10,000 questions are multiple-choice, with 5,000 sourced from each database. 
The questions are randomly selected from the test sets of the [Lawma tasks](https://huggingface.co/datasets/ricdomolm/lawma-tasks), and only cases with Court opinions of at least 2,000 characters are included.

To ensure input length limits, Court opinions are shortened so that the total input prompt (instruction, opinion, question, and answer choices) does not exceed 8,000 Llama 3 tokens.

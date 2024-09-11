# CaselawQA

CaselawQA is a benchmark based on legal classification tasks derived from the Supreme Court and Songer
Court of Appeals databases. Most of the questions in CaselawQA are framed as multiple choice questions. 
CaselawQA comprises 10,000 questions, 5,000 of them derived from the Supreme Court 
database and 5,000 of them from the Songer Court of Appeals database. Questions are sampled uniformly at 
random from the test sets of the [Lawma tasks](https://huggingface.co/datasets/ricdomolm/lawma-tasks). We 
only include cases whose Court opinion has at least 2,000 characters.

CaselawQA contains two additional subsets: CaselawQA Tiny and CaselawQA Hard. Caselaw QA Tiny comprises 
the 49 Lawma tasks that have fewer than 150 training examples. CaselawQA hard comprises those tasks for 
which [Lawma 70B](https://huggingface.co/ricdomolm/lawma-70b) attains less than 70% accuracy.

We shorten the Court opinions such that the overall input prompt (instruction + opinion + question + answer choices) does not exceed 8,000 Llama 3 tokens.

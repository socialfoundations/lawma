# Lawma: the power of specialization for legal tasks

[Lawma 8B](https://huggingface.co/ricdomolm/lawma-8b) and [Lawma 70B](https://huggingface.co/ricdomolm/lawma-70b) are language models fine-tuned on 260 legal classification tasks derived from the Supreme Court and Songer Court of Appeals legal databases. The Lawma models substantially outperform GPT-4 on 95\% of these legal classification tasks, on average by over 17 accuracy points.

* **The fine-tuning dataset**: our [fine-tuning dataset](https://huggingface.co/datasets/ricdomolm/lawma-all-tasks) contains a diverse set of 260 legal classification tasks, with around 500k task examples and 2 billion tokens. We fine-tune the Llama 3 Instruct models.
* **The legal classification tasks**: they comprise almost all of the variables of the [Supreme Court](http://scdb.wustl.edu/data.php) and [Songer Court of Appeals](www.songerproject.org/us-courts-of-appeals-databases.html) databases.
* **The details**: see our [arXiv preprint](https://arxiv.org/abs/2407.16615) for more details and a number of experiments on the scaling behaviour of fine-tuning, its sample efficiency, its generalization to unseen tasks and Courts, and the merits of single task specialization.

**What are the Lawma models useful for?** We recommend using the Lawma models only for the legal classification tasks that they models were fine-tuned on.
The model has been fine-tuned on multiple-choice questions, not on general instructions. Therefore, the model only outputs multiple choice letters (e.g., A, B, C, etc) or numbers. 
The main take-away of our paper is that specializing models leads to large improvements in performance. Therefore, we strongly recommend practitioners to further fine-tune Lawma on the actual tasks that the models will be used for. Relatively few examples --i.e, dozens or hundreds-- may already lead to large gains in performance.

## The legal classification tasks

Our reasons to study legal classification tasks are both technical and substantive. From a technical machine learning perspective, these tasks provide highly non-trivial classification problems where even the best models leave much room for improvement. From a substantive legal perspective, efficient solutions to such classification problems have rich and important applications in legal research.

You can find these legal classification tasks in [`ricdomolm/lawma-tasks`](https://huggingface.co/datasets/ricdomolm/lawma-tasks). For example, the following retrieves the train split of the `sc_issuearea` task:

```python
import pandas
import datasets

task_data = datasets.load_dataset('ricdomolm/lawma-tasks', 'sc_issuearea')
task_data = pandas.DataFrame(task_data['train'])
```

The datasets contain the following fields: `opinion` (the Court's opinion), the task's `instruction` and `question` (derived from the SC and Songer documnetation), `choices` (the possible answer chocies, if applicable), and `answer` (indexes of choices if choices if non-empty).

## Evaluation

To evaluate language models on each of the 260 legal tasks, please refer to the [evaluation](evaluation/) folder, and in particular [hf_eval.py](evaluation/hf_eval.py). You must first download the task files from [here](), or generate them yourself by following the instructions in the [data_generation](data_generation/) folder. We evaluated a range of language models:

| Model   | All tasks | Supreme Court tasks | Court of Appeals tasks |
|---------|:---------:|:-------------:|:----------------:|
| Lawma 70B | **81.9** | **84.1** | **81.5** |
| Lawma 8B | 80.3 | 82.4 | 79.9 |
| GPT4 | 62.9 | 59.8 | 63.4 |
| Llama 3 70B Inst | 58.4 | 47.1 | 60.3 |
| Mixtral 8x7B Inst | 43.2 | 24.4 | 46.4 |
| Llama 3 8B Inst | 42.6 | 32.8 | 44.2 |
| Majority classifier | 41.7 | 31.5 | 43.5 |
| Mistral 7B Inst | 39.9 | 19.5 | 43.4 |
| Saul 7B Inst | 34.4 | 20.2 | 36.8 |
| LegalBert | 24.6 | 13.6 | 26.4 |

The Lawma models substantially outperform all other models tested, and in particular GPT-4. Note that, while Lawma 70B generally outperforms Lawma 8B, the difference in performance is typically rather small. Therefore, practitioners may prefer to use Lawma 8B for its significantly cheaper inference and fine-tuning, with little cost in terms of model performance.

Note: evaluating models on all 260 classification tasks is reasonably compute intensive. However, for the purposes of language model benchmarking we may be mostly interested in aggregate performance. We are currently working on making aggregate evaluations less resource intensive by only considering a limited number of examples per task.

## Fine-tuning on our dataset

We fine-tune Lawma using the [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) library. Please refer to the README in the [fine-tune](fine-tune/) folder for the training scripts and configuration files that we used to fine-tune Lawma.

To fine-tune on our dataset of legal classification tasks, simply indicate so in your `config.yml` file:

```yaml
datasets:
  - path: ricdomolm/lawma-all-tasks
    type: alpaca
```

and then train using axolotl as usual

```bash
accelerate launch -m axolotl.cli.train config.yml
```

Fine-tuning Lawma 8B on 7xH100 GPUs required a total of 600 H100 hours (3 epochs), whereas fine-tuning Lawma 70B on 8 H100 nodes of 8 GPUs each required around 1600 H100 hours (1 epoch).

## Reproducing the experiments and figures of the paper

* The directory [data_generation](data_generation/) contains code used to create the legal classification tasks and the fine-tuning dataset.
* The directory [evaluation](evaluation/) contains code used to evaluate various models on the classification tasks.
* The directory [fine-tune](fine-tune/) contains code to fine-tune Lawma, as well as the for the additional fine-tuning experiments included in the paper.
* The directory [notebooks](notebooks/) contains ipynb files to generate the plots and tables of the paper.

See the README.md files in the subdirectories for additional documentation.

## Citation

Please cite as:

```
@misc{dominguezolmedo2024lawmapowerspecializationlegal,
      title={Lawma: The Power of Specialization for Legal Tasks}, 
      author={Ricardo Dominguez-Olmedo and Vedant Nanda and Rediet Abebe and Stefan Bechtold and Christoph Engel and Jens Frankenreiter and Krishna Gummadi and Moritz Hardt and Michael Livermore},
      year={2024},
      eprint={2407.16615},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16615}, 
}
```

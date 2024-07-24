# Lawma: The power of specialization for legal tasks

This is the primary code base for the project:

*Lawma: The Power of Specizalization for Legal Tasks*

*Ricardo Dominguez-Olmedo and Vedant Nanda and Rediet Abebe and Stefan Bechtold and Christoph Engel and Jens Frankenreiter and Krishna Gummadi and Moritz Hardt and Michael Livermore*

*2024*

To reproduce the results of the paper, take the following steps:

1. Go to `data_generation` for all code to create the classification tasks and the fine-tuning dataset.
2. The directory `evaluation` contains code used to evaluate various models on the classification tasks.
3. The directory `fine-tune` contains code to fine-tune Lawma, as well as the for the additional fine-tuning experiments included in the paper.
4. The directory `notebooks` contains ipynb files to generate the plots of the paper.

See the `README.md` files in the subdirectories for additional documentation.

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

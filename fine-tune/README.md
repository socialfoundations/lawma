# Fine-tuning experiments

To train the Lawma models, first follow the steps in the `data_generation` folder in order to generate the fine-tuning dataset, which we assume will be stored in `../instructions/tok_llama-3-8k_8192/`. Then, to train Lawma 8B on a single node with 7 GPUs, we use

```bash
accelerate launch --config_file 7gpu_zero2.yaml ft.py lawma-8b.yml --output_dir ../models/lawma-8b
```

For Lawma 70B, we use 8 nodes with 8 GPUs each. Modify `hostfile` to match your system. Then, train Lawma 70B using

```bash
NCCL_SOCKET_IFNAME=enp52s0np0 accelerate launch --config_file 8nodes_zero3.yaml ft.py lawma-70b.yml --output_dir ../models/lawma-70b
```

### Additional fine-tuning experiments

For our experiments, we use use an internal cluster with `htcondor`. You can see the specific job files in the `jobs/` folder, in particular:
    * `jobs_scaling.py` for the scaling experiments in Section 3.1
    * `jobs_eff.py` for the sample effiency experiments in Section 3.2
    * `jobs_specialized.py` for the specialization experiments in Section 3.3

For the generalization experiment in Section 3.4, we run

```bash
accelerate launch --config_file 7gpu_zero2.yaml ft.py lawma-8b.yml --output_dir ../models/ft-songer --prefix songer_ --num_epochs 1
```

in order to only train on Songer Court of Appeals tasks.
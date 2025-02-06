# ESM2 generator
This repository contains scripts for generating ESM2 embeddings. If your machine has a GPU, the script will automatically utilize it for faster embedding generation. If you're running the script on a cluster with SLURM, a pre-configured bash script is provided for this purpose.

## Choose model
You can select the desired ESM2 model by uncommenting the appropriate line in the `compute-esm.py` file:
```
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
```

## How to run
To generate embeddings, run the `compute-esm.py` script using the following command:
```
python3 compute-esm.py --input /path/to/input --output /path/to/output
```
Here, `--input` specifies the path to the text files containing sequences (refer to the example format in the `data` folder), and `--output` defines the path where the generated embeddings will be saved.

### Running on a SLURM cluster
To run the script on a SLURM-managed cluster, use the provided bash script:
```
sbatch --gpus=1 run-sbatch.sh
```
Before executing, make sure to update the `--input` and `--output` parameters inside the `run-sbatch.sh` script.

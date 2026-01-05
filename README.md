# Installation

This project uses `pipenv` for package management with `python 3.10.11`. Ensure that pipenv is installed and then run the following command:

```sh
pipenv install
```

A CUDA-enabled GPU with at least 12gb of VRAM is recommended for running the code in this repository. The experiments included in the paper used an Nvidia RTX 4090 with a runtime of around 48 hours to fully train the evolutionary algorithm.

# Datasets
The audition and somatosensation datasets are provided by `Freesound` and `MIT` respectively and both require additional preprocessing before they can be used. The olfaction and hippocampus datasets are generated using scripts provided in this repository.

## Audition Dataset

Download the following files from [https://zenodo.org/records/4060432](https://zenodo.org/records/4060432) for the `FSD50k` dataset:

```sh
FSD50K.dev_audio.z01
FSD50K.dev_audio.z02
FSD50K.dev_audio.z03
FSD50K.dev_audio.z04
FSD50K.dev_audio.z05
FSD50K.dev_audio.zip
FSD50K.doc.zip
FSD50K.eval_audio.z01
FSD50K.eval_audio.zip
FSD50K.ground_truth.zip
FSD50K.metadata.zip
```

Move those files to a new directory at `datasets/fsd50k_zipped/` located at the root of this repository, then run the following script:

```sh
./dual_computational_systems/data/scripts/unzip_fsd50k.sh
```

After the files are unzipped, run the next script to create a cochleagram dataset using the extracted `wav` files:

```sh
pipenv run python dual_computational_systems/data/generation/create_cochleagram_dataset.py
```

Note that cochleagram creation can be GPU-accelerated by adding a `--device cuda` flag.


## Somatosensation Dataset
Visit [https://stag.csail.mit.edu/](https://stag.csail.mit.edu/) and download the classification dataset. Place the downloaded `zip` file in a new directory at `datasets/tactile_glove_zipped` then run the script below from the root of this repository.

```sh
./dual_computational_systems/data/scripts/unzip_tactile_glove.sh
```

We use a subset of this dataset so as to keep the size of each dataset similar for each modality. After unzipping the data run the script below to create the subset:

```sh
pipenv run python dual_computational_systems/data/generation/subset_tactile_glove.py
```

## Hippocampus Dataset
Generate the hippocampal dataset with the following script:

```sh
pipenv run python dual_computational_systems/data/generation/create_hippocampus_dataset.py
```

## Olfactorion Dataset
Generate the olfaction dataset with the following script:

```sh
pipenv run python dual_computational_systems/data/generation/create_olfaction_dataset.py
```

# Evolutionary Algorithm
The evolutionary algorithm runs in two stages, first we train all mutations of the neural networks and cache the results of each. The number of epochs that each mutation is trained on can be modified with the `--n-epochs` command line flag and the total number of units to be divided amongst the different modalities can be controled with the `--model-base-channels` flag. We recommend running this portion of the algorithm with a CUDA-enabled GPU and the `--device cuda` flag.

```sh
pipenv run python dual_computational_systems/experiments/evolution.py
```

After training, the results of the training run will be saved to the cache directory and printed to the command line for reference. The next stage of the algorithm requires that one enter the path of that cache file as a command line argument. `--mutations-cache-path`. Doing so will skip the training phase and run the evolutionary phase to create visualizations like those in the paper. These visualizations are stored in the `output` directory at the root of the project. The first time the evolutionary phase is run, a mutations_pct cache file will also be created. To speed up subsequent runs of the evolutionary phase, provide the path to that cache file as a command line argument as well via the `--mutation-pcts-cache-path` flag. We provide the cache files used for the visualizations in the paper along with this code, see below for an example command which uses those files to run the evolutionary phase. Using pre-cached data does not require downloading/preprocessing any of the datasets mentioned above.

```sh
pipenv run python dual_computational_systems/experiments/evolution.py \
    --mutations-cache-path cache/mutations_cache_paper.json \
    --mutation-pcts-cache-path cache/mutation_pcts_paper.json \
    --max-survivors-scatter 10 \
    --allowed-mutation-drift 25 \
    --n-generations 10 \
    --population-size 50 \
    --mutation_sample_width 8 \
    --k-fit 25
```

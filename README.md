# Few-Shot Protein Fitness Prediction
Supported PLMs: **ESM-1b, ESM-1v, ESM-2, and SaProt**

## Requirements
```
cudatoolkit 11.8.0
learn2learn 0.2.0
pandas 1.5.3
peft 0.4.0
python 3.10
pytorch 2.0.1
scipy 1.10.1
scikit-learn 1.3.0
tqdm 4.65.0
transformers 4.29.2
```

## Config file
The config file `fsfp/config.json` defines the paths of model checkpoints, input and output.

## Data preprocessing
The datasets of ProteinGym should be put under `data/substitutions/`. Run `python preprocess.py -s` to preprocess the raw datasets and pack them to `data/merged.pkl`.

## Search for similar datasets for meta-training
- Run `python retrieval.py -m vectorize -md esm2` to compute and cache the embedding vectors of the proteins in ProteinGym, using ESM-2 for example.
- Run `python retrieval.py -m retrieve -md esm2 -b 16 -k 71 -mt cosine -cpu` to measure and save the similarities between proteins from the cached vectors.

## Training and inference
See `main.py` for detailed descriptions of each hyper-parameter. The default hyper-parameters are not optimal. It is highly recommended to perform hyper-parameter search for each protein via cross-validation.
### Use LTR and LoRA to train PLMs for specific protein (SYUA_HUMAN for example) without meta-learning
Run `python main.py -md esm2 -m finetune -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -p SYUA_HUMAN`. The trained model is saved to `checkpoints/`.
### Test the trained model, print results, and save predictions
Run `python main.py -md esm2 -m finetune -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -p SYUA_HUMAN -t`. The predictions are saved to `predictions/`.
### Meta-train PLMs on the auxiliary tasks
Run `python main.py -md esm2 -m meta -ts 40 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 1e-3 -as 4 -a GEMME -p SYUA_HUMAN`
### Transfer the meta-trained model to the target task
Run `python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 1e-3 -as 4 -a GEMME -p SYUA_HUMAN`
### Test the trained model, print results, and save predictions
Run `python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 1e-3 -as 4 -a GEMME -p SYUA_HUMAN -t`

## Before run SaProt
In order to build the structure-aware sequences for SaProt, it is necessary to first generate the 3Di sequences for all proteins by FoldSeek. Please refer to [Saprot](https://github.com/westlake-repl/SaProt) and [FoldSeek](https://github.com/steineggerlab/foldseek) for detailed information, and specify the `struc_seq_path` in `fsfp/config.json` for the generated sequences.

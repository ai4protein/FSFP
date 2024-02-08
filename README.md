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

## Search for similar datasets for meta-training (not necessary if using LTR only)
- Run `python retrieval.py -m vectorize -md esm2` to compute and cache the embedding vectors of the proteins in ProteinGym, using ESM-2 for example.
- Run `python retrieval.py -m retrieve -md esm2 -b 16 -k 71 -mt cosine -cpu` to measure and save the similarities between proteins from the cached vectors.

## Training and inference
Run `main.py` for model training and inference. The default hyper-parameters may not be optimal, so it is recommended to perform hyper-parameter search for each protein via cross-validation.
Important hyper-parmeters are listed as follows (abbreviations in parentheses):
- --mode (-m): perform LTR finetuning, meta-learning or transfer learning using the mear-learned model
- --test (-t): whether to load the trained models from checkpoints and test them
- --model (-md): name of the PLM to train
- --protein (-p): name of the target protein (UniProt ID)
- --train_size (-ts): few-shot training set size, can be a float number less than 1 to indicate a proportion
- --train_batch (-tb): batch size for training (outer batch size in the case of meta-learning)
- --eval_batch (-eb): batch size for evaluation
- --lora_r (-r): hyper-parameter r of LORA
- --optimizer (-o): optimizer for training (outer loop optimization in the case of meta-learning)
- --learning_rate (-lr): learning rate
- --epochs (-e): maximum training epochs
- --max_grad_norm (-gn): maximum gradient norm to clip to
- --list_size (-ls): list size for ranking
- --max_iter (-mi): maximum number of iterations per training epoch, useless during meta-training
- --eval_metric (-em): evaluation metric
- --augment (-a): specify one or more models to use their zero-shot scores for data augmentation
- --meta_tasks (-mt): number of tasks used for meta-training
- --meta_train_batch (-mtb): inner batch size for meta-training
- --meta_eval_batch (-meb): inner batch size for meta-testing
- --adapt_lr (-alr): learning rate for inner loop during meta-learning
- --patience (-pt): number of epochs to wait until the validation score improves
- --cross_validation (-cv): number of splits for cross validation (shuffle & split) on the training set
- --force_cpu (-cpu): use cpu for training and evaluation even if gpu is available

### Examples
- Use LTR and LoRA to train PLMs for specific protein (SYUA_HUMAN for example) without meta-learning: <br>
`python main.py -md esm2 -m finetune -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -p SYUA_HUMAN`. The trained model is saved to `checkpoints/`.
- Test the trained model, print results, and save predictions: <br>
`python main.py -md esm2 -m finetune -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -p SYUA_HUMAN -t`. The predictions are saved to `predictions/`.
- Meta-train PLMs on the auxiliary tasks: <br>
`python main.py -md esm2 -m meta -ts 40 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 1e-3 -as 4 -a GEMME -p SYUA_HUMAN`
- Transfer the meta-trained model to the target task: <br>
`python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 1e-3 -as 4 -a GEMME -p SYUA_HUMAN`
- Test the trained model, print results, and save predictions: <br>
`python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 1e-3 -as 4 -a GEMME -p SYUA_HUMAN -t`

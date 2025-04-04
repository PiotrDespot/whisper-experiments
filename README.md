# whisper-experiments
This repository contains a script that has an implementation of several training methods that were used to run experiments on the Whisper model.

## Prepare the data 
In order to prepare the data you need to run the `prepare_data.py` script in a proper configuration:
```bash
python prepare_data.py --dataset <dataset_name> --subset <dataset_subset> --split <dataset_split> --language <dataset_language> --perturbation_level <perturbation_level> --save_path <save_path>
```
The available arguments are:
- `dataset` - specifies the dataset name in huggingface repository. Only CommonVoice and LibriSpeech datasets are available.
- `subset` - specifies the subset of the downloaded dataset that will be processed
- `split` - specifies the split of the dataset that will be used for processing
- `language` - specifies the language of the dataset
- `perturbation_level` - specifies the magnitude of added noise to the dataset. Minimum level is 1 and maximum is 4. If you do not want to do any perturbations, do not set this argument.
- `save_path` - specifies where the processed dataset will be saved

## Run the experiments
You can run the training by filling the properties.json file with the proper configuration. In the `datasets` section fill the proper directory where the datasets were saved, names of the datasets, their splits and aliases. Additionaly, aliases should be put in the proper list in the `splits` section. After that, run:
```bash
python train_whisper.py --config properties.json
```

## Notable experiment configurations

### Experience Replay
In order to run the experiment with experience replay, set the `er_enabled` parameter in the properties file to `true` and set the proper `er_mix_percent` variable that will specify the size of the memory buffer.

### Learning without Forgetting
In order to run the experiment with LwF loss enabled, set the `lwf_enabled` parameter in the properties file to `true` and fill the `lwf_params` list in the `training` section with two variables. The first variable specifies the weight of the cross-entropy loss and the second specifies the weight of the kl-divergence loss. Can be conbined with experience replay.

### Learning without Forgetting with EMA as a teacher
In order to run the experiment with EMA as a teacher, set the `ema_enabled` parameter in the properties file to `true` and the `lwf_enabled` parameter to `false`. Then fill the `lwf_params` list in the `ema` section with two variables. The first variable specifies the weight of the cross-entropy loss and the second specifies the weight of the kl-divergence loss. Can be combined with experience replay.

### Weight averaging
In order to do the weight averaging at the end of the training, set the `merging_enabled` parameter in the properties_file to `true` and the `ema_enabled` parameter to `false`. Then set the `merging_alpha` parameter to specify the weight of the trained model during merging. Can be conbined with expereince raplay and Learning without Forgetting.

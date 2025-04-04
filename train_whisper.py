from dataclasses import dataclass
import torch
import os
import json
import datasets
import logging
import random
import numpy as np
import transformers
import torch.nn as nn
import time
from math import ceil
from tqdm import tqdm
from argparse import ArgumentParser
import evaluate
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_scheduler
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from typing import Any, Dict, List, Union
from datasets import load_from_disk, DatasetDict, concatenate_datasets, IterableDataset
from pydantic import BaseModel
from typing import List, Union
from copy import deepcopy
from enum import Enum


class DatasetPathsConfig(BaseModel):
    name: str
    subset: str
    split: str
    alias: str
    language: str
    remove_cols: list
    rename_cols: dict


class DatasetSplitsConfig(BaseModel):
    train: list
    eval: list
    test: list
    replay: list
 

class DatasetConfig(BaseModel):
    save_dir: str
    load_from_disk: bool
    paths: List[DatasetPathsConfig]
    splits: DatasetSplitsConfig


class WhisperTrainingArguments(BaseModel):
    cache_dir: str
    teacher_model: str
    student_model: str
    lwf_enabled: bool 
    er_enabled: bool
    gradient_checkpointing_enabled: bool
    do_eval: bool
    per_device_batch_size: int
    gradient_accumulation_steps: int
    dtype: str
    output_dir: str
    device: str
    num_epochs: float
    eval_steps: int
    log_steps: int
    save_steps: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    learning_rate: float
    warmup_steps: Union[int, float]
    lr_scheduler_type: str
    er_mix_percent: float
    lwf_params: list
    generate_max_length: int

class WandbConfig(BaseModel):
    project: str
    log_model: str
    run_name: str
    cache_dir: str

class SpecAugmentConfig(BaseModel):
    time_warp_enabled: bool
    frequency_masking_enabled: bool
    time_masking_enabled: bool
    time_warping_param: int
    frequency_masking_param: int
    time_masking_param: int
    frequency_mask_num: int
    time_mask_num: int
    augment_slice: int

class EmaConfig(BaseModel):
    ema_enabled: bool
    decay_enabled: bool
    alpha: float
    lwf_params: list

class WeightAveragingConfig(BaseModel):
    merging_enabled: bool
    merging_alpha: float

class Config(BaseModel):
    datasets: DatasetConfig
    training: WhisperTrainingArguments
    wandb: WandbConfig
    spec_augment: SpecAugmentConfig
    ema: EmaConfig
    weight_averaging: WeightAveragingConfig

class WhisperSpecAugment(nn.Module):

    def __init__(self, args: SpecAugmentConfig):
        super().__init__()
        self.time_warp_enabled = args.time_warp_enabled
        self.frequency_masking_enabled = args.frequency_masking_enabled
        self.time_masking_enabled = args.time_masking_enabled
        self.time_warping_param = args.time_warping_param
        self.frequency_masking_param = args.frequency_masking_param
        self.time_masking_param = args.time_masking_param
        self.frequency_mask_num = args.frequency_mask_num
        self.time_mask_num = args.time_mask_num
        self.augment_slice = args.augment_slice


    @staticmethod
    def __h_poly(t):
        tt = t.unsqueeze(-2)**torch.arange(4, device=t.device).view(-1,1)
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=t.dtype, device=t.device)
        return A @ tt
    
    @classmethod
    def __hspline_interpolate_1D(cls, x, y, xs):
        '''
        Input x and y must be of shape (batch, n) or (n)
        '''
        m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
        m = torch.cat([m[...,[0]], (m[...,1:] + m[...,:-1]) / 2, m[...,[-1]]], -1)
        idxs = torch.searchsorted(x[..., 1:], xs)
        dx = (x.gather(dim=-1, index=idxs+1) - x.gather(dim=-1, index=idxs))
        hh = cls.__h_poly((xs - x.gather(dim=-1, index=idxs)) / dx)
        return hh[...,0,:] * y.gather(dim=-1, index=idxs) \
            + hh[...,1,:] * m.gather(dim=-1, index=idxs) * dx \
            + hh[...,2,:] * y.gather(dim=-1, index=idxs+1) \
            + hh[...,3,:] * m.gather(dim=-1, index=idxs+1) * dx

    def _time_warp(self, mel_sectrogram):
        '''
        Timewarp augmentation

        param:
            specs: spectrogram of size (batch, channel, freq_bin, length)
            W: strength of warp
        '''
        device = mel_sectrogram.device
        batch_size, _, num_rows, spec_len = mel_sectrogram.shape

        warp_p = torch.randint(self.time_warping_param, spec_len - self.time_warping_param, (batch_size,), device=device)

        # Uniform distribution from (0,W) with chance to be up to W negative
        warp_d = torch.randint(-self.time_warping_param, self.time_warping_param, (batch_size,), device=device)
        
        x = torch.stack([torch.tensor([0], device=device).expand(batch_size),
                        warp_p, torch.tensor([spec_len-1], device=device).expand(batch_size)], 1)
        y = torch.stack([torch.tensor([-1.], device=device).expand(batch_size),
                        (warp_p-warp_d)*2/(spec_len-1.)-1., torch.tensor([1.], device=device).expand(batch_size)], 1)

        # Interpolate from 3 points to spec_len
        xs = torch.linspace(0, spec_len-1, spec_len, device=device).unsqueeze(0).expand(batch_size, -1)
        ys = self.__hspline_interpolate_1D(x, y, xs)

        grid = torch.cat(
            (ys.view(batch_size,1,-1,1).expand(-1,num_rows,-1,-1),
            torch.linspace(-1, 1, num_rows, device=device).view(-1,1,1).expand(batch_size,-1,spec_len,-1)), -1)
        
        return torch.nn.functional.grid_sample(mel_sectrogram, grid, align_corners=True)

    def forward(self, x):

        if not (self.time_warp_enabled and self.time_masking_enabled and self.frequency_masking_enabled):
            return x
        
        mel_spectrogram = x["input_features"]
        B, F, T = mel_spectrogram.shape
        mel_spectrogram = mel_spectrogram.view(B, 1, F, T)

        if self.augment_slice > B:
            raise ValueError("Can't augment slice that is bigger than batch size")
        
        if self.augment_slice > 0:
            augment_batch = torch.randint(0, B, (self.augment_slice, ))
            print(augment_batch)
            mel_spectrogram_slice = mel_spectrogram[augment_batch]
        else:
            mel_spectrogram_slice = mel_spectrogram

        # Step 1 : Time warping
        if self.time_warp_enabled:
            mel_spectrogram_slice = self._time_warp(mel_spectrogram_slice)

        # Step 2 : Frequency masking
        if self.frequency_masking_enabled:
            for _ in range(self.frequency_mask_num):
                f = np.random.uniform(low=0.0, high=self.frequency_masking_param)
                f = int(f)
                f0 = random.randint(0, F-f)
                mel_spectrogram_slice[:, :, f0:f0+f, :] = 0

        # Step 3 : Time masking
        if self.time_masking_enabled:
            for _ in range(self.time_mask_num):
                t = np.random.uniform(low=0.0, high=self.time_masking_param)
                t = int(t)
                t0 = random.randint(0, T-t)
                mel_spectrogram_slice[:, :, :, t0:t0+t] = 0

        if self.augment_slice > 0:
            mel_spectrogram[augment_batch] = mel_spectrogram_slice
        else:
            mel_spectrogram = mel_spectrogram_slice

        x["input_features"] = mel_spectrogram.view(B, F, T)

        return x
    

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class TrainingUtils:

    @staticmethod
    def load_data(config: DatasetConfig) -> DatasetDict:

        data = DatasetDict()

        for download_path in config.download_paths:
            load_path = os.path.join(config.save_dir, download_path.name)
            dataset = load_from_disk(load_path)

            if download_path.alias in config.splits.train:
                data[download_path.alias] = dataset
            
            if download_path.alias in config.splits.eval:
                data[download_path.alias] = dataset

            if download_path.alias in config.splits.test:
                data[download_path.alias] = dataset

            if download_path.alias in config.splits.replay:
                data[download_path.alias] = dataset

        return data

    @classmethod
    def get_parameter_names(cls, model, forbidden_layer_types, forbidden_module=None):
        """
        Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
        Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
        (e.g. if the module is frozen).
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in cls.get_parameter_names(child, forbidden_layer_types, forbidden_module)
                if not (
                    isinstance(child, tuple(forbidden_layer_types))
                    or (child in tuple(forbidden_module) if forbidden_module is not None else False)
                )
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result


    @staticmethod
    def log_metric(accelerator, metrics: Dict, train_time: float, step: int, epoch: int, learning_rate: float = None, prefix: str = "train"):
        log_metrics = {}

        for k, v in metrics.items():
            log_metrics[f"{prefix}/{k}"] = v

        log_metrics[f"{prefix}/time"] = train_time
        log_metrics[f"{prefix}/epoch"] = epoch

        if learning_rate is not None:
            log_metrics[f"{prefix}/learning_rate"] = learning_rate

        accelerator.log(log_metrics, step=step)

    @staticmethod
    def perform_weight_averaging(student_model, teacher_model, alpha):
        teacher_params = dict(teacher_model.named_parameters())
        for name, param in student_model.named_parameters():
            param.data.copy_((1 - alpha) * teacher_params[name].data + alpha * param.data)

        return student_model


class EmaModelUtils:

    def __init__(self, ema_args: EmaConfig):
        self.ema_args = ema_args

    def get_ema_model(self, model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
        if not self.ema_args.ema_enabled:
            return None
        
        new_ema_model = deepcopy(model)
        for param in new_ema_model.parameters():
            param.detach_()

        return new_ema_model
    
    def update_ema_variables(self, ema_model: WhisperForConditionalGeneration, model: WhisperForConditionalGeneration):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = self.ema_args.alpha * ema_param[:].data[:] + (1 - self.ema_args.alpha) * param[:].data[:]
        
        return ema_model    

class TrainerStage(Enum):
    TRAIN = "train"
    EVAL = "test"
    GENERATE = "generate"


class Trainer:

    def __init__(self, train_args: WhisperTrainingArguments, ema_args: EmaConfig, accelerator: Accelerator, tokenizer: WhisperTokenizer):
        self.train_args = train_args
        self.ema_args = ema_args
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer", cache_dir=train_args.cache_dir)

    def __call__(self, batch, student_model: WhisperForConditionalGeneration, teacher_model: WhisperForConditionalGeneration | None, stage: TrainerStage):
        match stage:
            case TrainerStage.TRAIN:
                return self.train_step(student_model, teacher_model, batch)
            case TrainerStage.EVAL:
                return self.eval_step(student_model, teacher_model, batch)
            case TrainerStage.GENERATE:
                return self.generate_step(student_model, batch)
        

    @staticmethod
    def __kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence

    def train_step(self, student_model, teacher_model, batch, temperature=2.0):
        student_model.train()
        teacher_model.eval()

        student_outputs = student_model(**batch)
        ce_loss = student_outputs.loss

        if self.train_args.lwf_enabled or self.ema_args.ema_enabled:
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
                student_distribution = nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1)
                kl_loss = self.__kl_divergence(teacher_distribution, student_distribution, batch["labels"]) * temperature**2
        else:
            kl_loss = 0.0

        if self.train_args.lwf_enabled:
            loss = self.train_args.lwf_params[0] * ce_loss + self.train_args.lwf_params[1] * kl_loss
        elif self.ema_args.ema_enabled:     
            loss = self.ema_args.lwf_params[0] * ce_loss + self.ema_args.lwf_params[1] * kl_loss
        else:
            loss = ce_loss

        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss} if self.train_args.lwf_enabled or self.ema_args.ema_enabled else {"loss": loss}

        return loss, metrics
    
    def eval_step(self, student_model, teacher_model, batch, temperature=1.0):
        student_model.eval()
        teacher_model.eval()

        with torch.no_grad():
            student_outputs = student_model(**batch)
            ce_loss = student_outputs.loss

            if self.train_args.lwf_enabled:
                teacher_outputs = teacher_model(**batch)
                teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
                student_distribution = nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1)
                kl_loss = self.__kl_divergence(teacher_distribution, student_distribution, batch["labels"]) * temperature**2
            else:
                kl_loss = 0.0

        if self.train_args.lwf_enabled:
            loss = self.train_args.lwf_params[0] * ce_loss + self.train_args.lwf_params[1] * kl_loss
        elif self.ema_args.ema_enabled:     
            loss = self.ema_args.lwf_params[0] * ce_loss + self.ema_args.lwf_params[1] * kl_loss
        else:
            loss = ce_loss

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss} if self.train_args.lwf_enabled or self.ema_args.ema_enabled else {"loss": loss}

    def generate_step(self, student_model: WhisperForConditionalGeneration, batch):
        student_model.eval()

        output_ids = self.accelerator.unwrap_model(student_model).generate(batch["input_features"], max_length=self.train_args.generate_max_length)
        output_ids = self.accelerator.pad_across_processes(output_ids, dim=1, pad_index=student_model.tokenizer.pad_token_id)

        return output_ids

    def compute_metrics(self, preds, labels):

        # replace -100 with the pad_token_id
        for idx in range(len(labels)):
            labels[idx][labels == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True, normalize=True)
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True, normalize=True)
       
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="properties.json")
    args = parser.parse_args()

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    with open(args.config, 'r') as f:
        config = Config(**json.load(f)) 

    train_args = config.training
    datasets_args = config.datasets
    wandb_args = config.wandb
    spec_augment_args = config.spec_augment
    ema_args = config.ema
    merging_args = config.weight_averaging
    

    if train_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16

    elif train_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32    

    accelerator = Accelerator(
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_dir=train_args.output_dir
    )

    logger.info(f"Found {accelerator.num_processes} processes")

    logger.info('Loading whisper tokenizer and processor')
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe", cache_dir=train_args.cache_dir)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe", cache_dir=train_args.cache_dir)

    logger.info('Loading student model')
    student_model = WhisperForConditionalGeneration.from_pretrained(train_args.student_model, cache_dir=train_args.cache_dir)
    student_model.config.forced_decoder_ids = None
    student_model.config.suppress_tokens = []

    logger.info('Loading teacher model')
    teacher_model = WhisperForConditionalGeneration.from_pretrained(train_args.teacher_model, torch_dtype=teacher_dtype, cache_dir=train_args.cache_dir)
    teacher_model.config.forced_decoder_ids = None
    teacher_model.config.suppress_tokens = []

    spec_augment = WhisperSpecAugment(spec_augment_args)

    os.environ["WANDB_PROJECT"] = wandb_args.project    
    os.environ["WANDB_LOG_MODEL"] = wandb_args.log_model
    os.environ["WANDB_NAME"] = wandb_args.run_name
    os.environ["WANDB_CACHE_DIR"] = wandb_args.cache_dir

    accelerator.init_trackers(os.environ["WANDB_PROJECT"])

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.wait_for_everyone()

    if student_model.config.decoder_start_token_id is None or teacher_model.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for both the "
            f"student and teacher model. Got {student_model.config.decoder_start_token_id} for the "
            f"student and {teacher_model.config.decoder_start_token_id} for the teacher."
        )

    if train_args.gradient_checkpointing_enabled:
        student_model.gradient_checkpointing_enable()

    accelerator.wait_for_everyone()


    logger.info("Loading datasets from disk")
    data = TrainingUtils.load_data(datasets_args)

    data["train"] = concatenate_datasets([data[split] for split in datasets_args.splits.train]).shuffle()

    if train_args.er_enabled:
        data["replay"] = concatenate_datasets([data[split] for split in datasets_args.splits.replay]).shuffle()


    train_batch_size = train_args.per_device_batch_size

    if train_args.er_enabled:
        required_replay_sample = int(len(data["train"]) * train_args.er_mix_percent)
        logger.info(f"Experience replay enabled. {required_replay_sample} samples from replay datasets will be mixed with training data.")
        data["train"] = concatenate_datasets([data["train"], data["replay"].select(range(required_replay_sample))]).shuffle()

    steps_per_epoch = len(data["train"]) // (train_args.per_device_batch_size * train_args.gradient_accumulation_steps * accelerator.num_processes)
    total_train_steps = int(steps_per_epoch * train_args.num_epochs)

    forbidden_module = [
        module for module, flag in [(student_model.model.encoder, train_args.freeze_encoder)] if flag
    ] or None if not train_args.peft_enabled else None


    decay_parameters = [name for name in TrainingUtils.get_parameter_names(student_model, [nn.LayerNorm], forbidden_module=forbidden_module) if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [param for name, param in student_model.named_parameters() if name in decay_parameters],
            "weight_decay": train_args.weight_decay
        },
        {
            "params": [param for name, param in student_model.named_parameters() if name not in decay_parameters],
            "weight_decay": 0.0
        }
    ]

    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_epsilon
    )

    if isinstance(train_args.warmup_steps, float):
        warmup_steps = int(train_args.warmup_steps * steps_per_epoch)
    else:
        warmup_steps = train_args.warmup_steps

    logger.info(f"Number of warmup steps: {warmup_steps}")

    lr_scheduler = get_scheduler(
        name=train_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps * accelerator.num_processes
    )

    student_model, teacher_model, optimizer, lr_scheduler, spec_augment = accelerator.prepare(
        student_model, teacher_model, optimizer, lr_scheduler, spec_augment
    )

    trainer = Trainer(train_args, ema_args, accelerator, tokenizer)
    ema_utils = EmaModelUtils(ema_args)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    ema_model = ema_utils.get_ema_model(student_model)
    
    logger.info("Training initialized")

    train_time = 0

    train_start = time.time()
    progress_bar = tqdm(
        range(total_train_steps),
        desc="Train steps ... ",
        position=0, 
        disable = not accelerator.is_local_main_process
    )

    continue_training = True
    epochs_trained = 0
    current_step = 0

    total_epochs = int(ceil(train_args.num_epochs))
    for epoch in range(epochs_trained, total_epochs):

        train_dataloader = DataLoader(
            data["train"],
            collate_fn=data_collator,
            batch_size=train_batch_size,
            num_workers=0, 
            pin_memory=False
        )

        train_dataloader = accelerator.prepare(train_dataloader)

        for batch in train_dataloader:
            batch = spec_augment(batch)
            with accelerator.accumulate(student_model):
                loss, train_metric = trainer(batch, student_model, teacher_model, TrainerStage.TRAIN) if ema_args.ema_enabled else trainer(batch, student_model, ema_model, TrainerStage.TRAIN)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if ema_args.ema_enabled:
                ema_model = ema_utils.update_ema_variables(ema_model, student_model)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                current_step += 1

                if current_step % train_args.log_steps == 0:
                    progress_bar.write(
                        f"Step... ({current_step} / {total_train_steps} | Loss: {train_metric['loss']}, Learning Rate: {lr_scheduler.get_last_lr()[0]})"
                    )

                    TrainingUtils.log_metric(
                        accelerator, 
                        metrics=train_metric, 
                        learning_rate=lr_scheduler.get_last_lr()[0], 
                        train_time=train_time + time.time() - train_start,
                        step=current_step, 
                        epoch=epoch, 
                        prefix="train"
                    )

                if current_step == total_train_steps and merging_args.merging_enabled:
                    student_model = TrainingUtils.perform_weight_averaging(student_model, teacher_model, merging_args.merging_alpha)
                
                if (train_args.save_steps > 0 and current_step % train_args.save_steps == 0) or current_step == total_train_steps:
                    checkpoint_dir = os.path.join(train_args.output_dir, "checkpoints", wandb_args.run_name, f"checkpoint-{current_step}-epoch-{epoch}")
                    accelerator.wait_for_everyone()

                    unwrapped_model = accelerator.unwrap_model(student_model)

                    unwrapped_model.save_pretrained(
                        checkpoint_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(student_model)    
                    )

                    accelerator.wait_for_everyone()

                if train_args.do_eval and ((train_args.eval_steps > 0 and current_step % train_args.eval_steps == 0) or current_step == total_train_steps):
                    train_time += time.time() - train_start
                    student_model.eval()

                    for eval_split in datasets_args.splits.eval:
                        eval_metrics = []
                        eval_preds = []
                        eval_labels = []
                        eval_start = time.time()

                        validation_data = DataLoader(
                            data[eval_split],
                            collate_fn=data_collator,
                            batch_size=train_args.per_device_batch_size,
                            drop_last=False,
                            num_workers=0,
                            pin_memory=False,
                            shuffle=False
                        )

                        validation_data = accelerator.prepare(validation_data)

                        accelerator.wait_for_everyone()

                        for batch in tqdm(validation_data, desc=f"Evaluating {eval_split}...", position=2, disable=not accelerator.is_local_main_process):
                            eval_metric = trainer(batch, student_model, teacher_model, TrainerStage.EVAL) if ema_args.ema_enabled else trainer(batch, student_model, ema_model, TrainerStage.EVAL)
                            eval_metric = accelerator.gather_for_metrics(eval_metric)
                            generated_ids = trainer(batch, student_model, None, TrainerStage.GENERATE)

                            generated_ids, labels =  accelerator.gather_for_metrics(
                                (generated_ids, batch["labels"])
                            )                            

                            eval_metrics.append(eval_metric)
                            eval_preds.extend(generated_ids)
                            eval_labels.extend(labels)

                        eval_time = time.time() - eval_start

                        eval_metrics = {
                            key: torch.mean(torch.stack([d[key] for d in eval_metrics])) for key in eval_metrics[0]
                        }

                        wer_desc = ""
                        wer_metric = trainer.compute_metrics(
                            eval_preds, eval_labels
                        )

                        eval_metrics.update(wer_metric)

                        wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
                    
                        progress_bar.write(f"Eval results for step ({current_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']}) | {wer_desc}")

                        TrainingUtils.log_metric(
                            accelerator,
                            metrics=eval_metrics,
                            train_time=eval_time,
                            step=current_step,
                            epoch=epoch,
                            prefix=eval_split
                        )

                    train_start = time.time()
            
                if current_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()
                                         

if __name__ == '__main__':
    main()
                                         

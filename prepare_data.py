from datasets import load_dataset, Audio
import torch
from argparse import ArgumentParser
import string
from multiprocessing import cpu_count
from transformers import WhisperProcessor


N_CPUS = cpu_count()
N_GPUS = torch.cuda.device_count()


class GaussianNoise(torch.nn.Module):
    PERTURBATION_LEVELS = [40, 30, 20, 10, 0]
    def __init__(self, perturbation_level) -> None:
        super().__init__()
        self.snr = self.PERTURBATION_LEVELS[perturbation_level]
    
    def __repr__(self):
        return f"GaussianNoise({self.snr} dB)"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        rng = torch.Generator(x.device)
        rng = rng.manual_seed(rng.seed())
        d = torch.empty_like(x).normal_(0, 1, generator=rng)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        return F.add_noise(x, d, snr)



def trim_text_to_charcount(text, charcount):
    if len(text) > charcount:
        new_text = text[:charcount]
        if text[charcount] in string.ascii_letters:
            new_text = new_text[:new_text.rfind(' ')]
        text = new_text
    return text

def transform_dataset(dataset, perturbation_level):
    noise = GaussianNoise(perturbation_level)
    def transform_(batch):
        for i, (audio, text) in enumerate(zip(batch['audio'], batch['sentence'])):
            audio['array'] = noise(audio['array'])
        return batch

    dataset = dataset.map(transform_, batched=True, batch_size=64, num_proc=4)

    dataset = dataset.with_format('np')
    return dataset

def process_dataset(data, tokenizer, feature_extractor):
    def map_function(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"].lower()).input_ids
        return batch
    
    return data.map(map_function, remove_columns=data.column_names, num_proc=1)


COMMON_VOICE_REMOVE_COLS = [
    "age", 
    "client_id", 
    "down_votes", 
    "gender", 
    "locale", 
    "path", 
    "segment", 
    "up_votes"
]
LIBRI_SPEECH_REMOVE_COLS = [
    "file", 
    "speaker_id", 
    "chapter_id", 
    "id"
]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="librispeech_asr", help='Name for dataset to load from huggingface hub. Only common voice and librispeech datasets are supported default: librispeech_asr.')
    parser.add_argument('--subset', default="clean", help='Subset of the dataset to use. default: clean')
    parser.add_argument('--split', default='test', help='Split of the dataset to use. default: test')
    parser.add_argument("--language", type=str, default="English", help="Language in which the transcripts in the datasets were written. default: English")
    parser.add_argument('--perturbation_level', type=int, help='Augmentation to apply to the dataset. 1-4 default: None')
    parser.add_argument('--save_path', type=str, required=True, help="Path where processed dataset is to be saved")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.filter(lambda x: not x['id'].startswith('inter_segment_gap'))

    if "librispeech" in args.dataset:
        dataset = dataset.remove_columns(LIBRI_SPEECH_REMOVE_COLS).rename_column("transcript", "sentence")
    elif "common_voice" in args.dataset:    
        dataset = dataset.remove_columns(COMMON_VOICE_REMOVE_COLS)
    else:
        raise ValueError("Unrecognised dataset. Only common_voice and librispeech datasets are supported")
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language=args.language, task="transcribe")

    if args.perturbation_level is not None:
        dataset = transform_dataset(dataset, args.perturbation_level)

    dataset = process_dataset(dataset, processor.tokenizer, processor.feature_extractor)

    dataset.save_to_disk(args.save_path)

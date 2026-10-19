#!/usr/bin/env python
# coding: utf-8
"""
run script with these:
    runs the model and sets its name -> python experiments.py --run baseline
    set some augmentartions -> python experiments.py --run gauss_only --set p_gauss=0.5 p_gain=0.0 p_shift=0.0
    run from a saved configuration -> python experiments.py --run loo_mixup --set p_mixup=0.0 --from full_stack.json
"""

import argparse
import ast
import copy
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, asdict, fields

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True' #makes debugging gpu issues easier

import matplotlib
matplotlib.use('Agg')  # no display over ssh
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import librosa  # audio preprocessing
import librosa.display
import soundfile as sf  # faster audio preprocessing
import kornia.augmentation as Kornia  # for colour jiggle
import audiomentations as A
from tqdm import tqdm 

import torch
import torchaudio
import torch.nn as nn
from torch.distributions import Beta #for mixup
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
import sklearn.metrics


# configs

@dataclass
class Config:
    # --- run identity 
    run_name: str = 'baseline'
    seed: int = 42
    notes: str = ''

    #--- file paths
    audio_dir: str = 'train_audio'
    metadata_csv: str = 'train_metadata.csv'
    work_dir: str = 'work'
    cache_dir: str = 'cache'
    results_csv: str = 'results/runs.csv'
    init_checkpoint: str = ''         #pretrained weights. empty string is for imagenet

    #--- audio params
    sr: int = 32_000
    chunk_secs: float = 5.0           #match the 5 s test window for kaggle submission
    train_overlap: float = 0.5        #overlap %
    num_classes: int = 264

    #--- optimisation stuff
    backbone: str = 'resnet18'        #resnet18,resnet50,efficientnet_b0
    batch_size: int = 64
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.0
    dropout: float = 0.0
    use_amp: bool = True
    freeze_strategy: str = 'none'     # none,head_only,warmup
    freeze_epochs: int = 5            # only used by warmup
    loss_type: str = 'ce'             #cross entropy

    #---spectrogram params
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    f_min: int = 40
    f_max: int = 15_000
    top_db: float = 100.0
    img_size: int = 224

    #---soft augs, p=0 disables 
    p_gauss: float = 0.0
    gauss_amp: tuple = (0.001, 0.015)
    p_colour_noise: float = 0.0       # pink / brown
    colour_snr_db: tuple = (5, 25)
    p_shift: float = 0.0              # cyclic shift
    shift_frac: float = 0.5
    p_gain: float = 0.0
    gain_db: tuple = (-6, 6)

    #--- hard augs
    p_mixup: float = 0.0
    mix_beta: float = 0.4
    p_stretch: float = 0.0            #time stretching
    stretch_range: tuple = (0.9, 1.1)
    p_mask: float = 0.0               # specaugment
    freq_mask: int = 24               #num mels being masked
    time_mask: int = 24
    n_masks: int = 2
    p_pitch: float = 0.0              #pitch shifting
    p_jiggle: float = 0.0
    jiggle_brightness: float = 0.2
    jiggle_contrast: float = 0.2

    #--- early stopping + cpu workers
    patience: int = 5
    num_workers: int = 4


def coerce(value, target_type):
    """turn a --set string into the type the dataclass field declares"""
    if target_type is bool:
        return str(value).lower() in ('1', 'true', 'yes')
    if target_type is tuple:
        return tuple(ast.literal_eval(value))
    return target_type(value)


def build_config():
    """takes args used in cmd to make the config files to keep track of run details and what not"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None, help='run name')
    parser.add_argument('--from', dest='from_json', default=None,
                        help='start from a previous run config json')
    parser.add_argument('--set', nargs='*', default=[],
                        help='key=value overrides, applied last')
    args = parser.parse_args()

    cfg = Config()
    if args.from_json:
        with open(args.from_json) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, tuple(v) if isinstance(v, list) else v)

    types = {f.name: f.type for f in fields(Config)}
    for item in args.set:
        key, _, value = item.partition('=')
        if key not in types:
            raise KeyError(f'unknown config field: {key}')
        setattr(cfg, key, coerce(value, types[key]))

    if args.run:
        cfg.run_name = args.run
    return cfg


#sets the seed for stuff
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)



# audiomentations stuff
def build_waveform_augment(cfg):
    """compose a set of augmentations using audiomentatoons and apply them in order"""
    transforms = []
    if cfg.p_gauss > 0:
        transforms.append(A.AddGaussianNoise(min_amplitude=cfg.gauss_amp[0],
                                             max_amplitude=cfg.gauss_amp[1],
                                             p=cfg.p_gauss))
                                             
    if cfg.p_colour_noise > 0:
        transforms.append(A.OneOf([
            A.AddColorNoise(min_snr_db=cfg.colour_snr_db[0],
                            max_snr_db=cfg.colour_snr_db[1],
                            min_f_decay=-3.01, max_f_decay=-3.01, p=1.0),  #pink
            A.AddColorNoise(min_snr_db=cfg.colour_snr_db[0],
                            max_snr_db=cfg.colour_snr_db[1],
                            min_f_decay=-6.02, max_f_decay=-6.02, p=1.0),  #brown
        ], p=cfg.p_colour_noise))
        
    if cfg.p_shift > 0:
        transforms.append(A.Shift(min_shift=-cfg.shift_frac, max_shift=cfg.shift_frac,
                                  shift_unit='fraction', rollover=True, p=cfg.p_shift))
                                  
    if cfg.p_gain > 0:
        transforms.append(A.Gain(min_gain_db=cfg.gain_db[0],
                                 max_gain_db=cfg.gain_db[1], p=cfg.p_gain))
    return A.Compose(transforms) if transforms else None



#---dataset handling, index logic
def index_chunks(paths, labels, sr, chunk_secs, stride_frac, cache_path):
    """build the (path, start, end, label) index, or laod it from memory using pickle"""
    if cache_path and os.path.exists(cache_path):
        print(f'loading cached index from {cache_path}...')
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)
        print(f'loaded {len(samples)} chunks.')
        return samples

    samples = []
    chunk_len = int(chunk_secs * sr)
    stride = max(1, int(chunk_len * stride_frac))
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        try:
            info = sf.info(path)  #reads the header only
            n_frames = int(info.frames * (sr / info.samplerate))
            num_chunks = max(1, (n_frames - chunk_len) // stride + 1)
            for i in range(num_chunks):
                start = i * stride
                end = min(start + chunk_len, n_frames)
                samples.append((path, start, end, label))
        except Exception as e:
            print(f'skipping {path}: {e}')

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f)
    print(f'total chunks: {len(samples)}')
    return samples


class ChunkDataset(Dataset):
    """builds the train and val datasets. augmentations and overlap are applied to the train, but not the val """

    def __init__(self, samples, cfg, augment=None):
        self.samples = samples
        self.cfg = cfg
        self.augment = augment
        self.target_len = int(cfg.chunk_secs * cfg.sr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """open the audio recording fromn the index, jump to its start, then read its exact chunk length"""
        path, start_frame, end_frame, label = self.samples[index]

        try:
            #if the file doesnt load or an augmentation fails to apply, the chunks are replaced with zeros
            with sf.SoundFile(path) as f:
                f.seek(start_frame)  # jump to the start frame
                chunk = f.read(frames=end_frame - start_frame, dtype='float32')

            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
                
            if len(chunk) < self.target_len:  #pad short chunks with zeros
                padded = np.zeros(self.target_len, dtype='float32')
                padded[:len(chunk)] = chunk
                chunk = padded

            if self.augment is not None:
                chunk = self.augment(samples=chunk, sample_rate=self.cfg.sr) 

        except Exception as e:  #write all the corrupt files to a log
            os.makedirs(self.cfg.work_dir, exist_ok=True)
            with open(os.path.join(self.cfg.work_dir, 'bad_files.txt'), 'a') as log:
                log.write(f'{path}\t{e}\n')
            chunk = np.zeros(self.target_len, dtype='float32')

        return torch.tensor(chunk, dtype=torch.float32), label


def link_labels(audio_dir, df, label2idx):
    """match each file path to its label index. drops missing files"""
    paths, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(audio_dir, row['filename'])
        if os.path.exists(path) and row['primary_label'] in label2idx:
            paths.append(path)
            labels.append(label2idx[row['primary_label']])
    return paths, labels




def padded_cmap(solution, submission, padding_factor=5):
    """metric function provided by birdclef"""
    solution = solution.drop(['row_id'], axis=1, errors='ignore')
    submission = submission.drop(['row_id'], axis=1, errors='ignore')
    new_rows = [[1 for _ in range(len(solution.columns))]
                for _ in range(padding_factor)]
    new_rows = pd.DataFrame(new_rows, columns=solution.columns)
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True)
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True)
    return sklearn.metrics.average_precision_score(
        padded_solution.values, padded_submission.values, average='macro')


def display_spectrogram(spec, sr, hop_length, title='Spectrogram', out_path=None):
    """visualise a spec using librosa + matplotlib"""
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)  
    fig = plt.figure(figsize=(10, 5))
    librosa.display.specshow(spec, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    if out_path:
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        print(f'saved {out_path}')
    plt.close(fig)



# gpu-side augmentations
class Mixup(nn.Module):
    """blends two signals and their labels using per-sample coefficients"""

    def __init__(self, mix_beta=0.4):
        super().__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y):
        bs = X.shape[0]
        perm = torch.randperm(bs, device=X.device)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        shape = [-1] + [1] * (X.dim() - 1)
        X = coeffs.view(*shape) * X + (1 - coeffs.view(*shape)) * X[perm]
        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
        return X, Y


class RationalPitchShift(nn.Module):
    """a rational ratio keeps the resample kernel smol. 
        torchaudio's PitchShift derives an irrational ratio from n_steps, resulting in insufficient memory"""

    def __init__(self, num, den, n_fft=512, hop_length=128):
        super().__init__()
        self.num, self.den = num, den
        self.n_fft, self.hop = n_fft, hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        self.stretch = torchaudio.transforms.TimeStretch(
            hop_length=hop_length, n_freq=n_fft // 2 + 1)

    def forward(self, wav):
        n = wav.shape[-1]
        z = torch.stft(wav, self.n_fft, self.hop,
                       window=self.window, return_complex=True)
        z = self.stretch(z, self.den / self.num)                   # stretch the pitch
        y = torch.istft(z, self.n_fft, self.hop, window=self.window)
        y = torchaudio.functional.resample(y, self.num, self.den)  # squash the pitch
        if y.shape[-1] > n:
            y = y[..., :n]
        elif y.shape[-1] < n:
            y = nn.functional.pad(y, (0, n - y.shape[-1]))
        return y


def build_pitch_shifters(dev, chunk_len):
    shifters = {-2: RationalPitchShift(8, 9).to(dev),
                2: RationalPitchShift(9, 8).to(dev)}
    warm = torch.zeros(8, chunk_len, device=dev)
    for n, s in shifters.items():
        _ = s(warm)
        print(f'pitch shifter {n:+d} ready')
    del warm
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return shifters


def apply_pitch_shift(wav, shifters, p):
    """each chunk independently has a chance p of being shifted"""
    if p <= 0 or not shifters:
        return wav
        
    mask = torch.rand(wav.shape[0], device=wav.device) < p
    if not mask.any():
        return wav
    out = wav.clone()
    idx = mask.nonzero(as_tuple=True)[0]
    idx = idx[torch.randperm(idx.numel(), device=idx.device)]  # split evenly
    halves = idx.chunk(len(shifters))
    for shifter, rows in zip(shifters.values(), halves):
        if rows.numel():
            out[rows] = shifter(wav[rows])
    return out


class MelSpecTransform(nn.Module):
    """audio -> normalised 3 channel spec. also applies time stretching and specaug"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=cfg.n_fft, win_length=cfg.n_fft,
            hop_length=cfg.hop_length, power=None)
            
        self.stretch = torchaudio.transforms.TimeStretch(
            hop_length=cfg.hop_length, n_freq=cfg.n_fft // 2 + 1)
            
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=cfg.n_mels, sample_rate=cfg.sr,
            f_min=cfg.f_min, f_max=cfg.f_max, n_stft=cfg.n_fft // 2 + 1)
            
        self.n_frames = int(cfg.chunk_secs * cfg.sr) // cfg.hop_length + 1
        self.freq_mask = torchaudio.transforms.FrequencyMasking(cfg.freq_mask)
        self.time_mask = torchaudio.transforms.TimeMasking(cfg.time_mask)

    def _to_db(self, power):
        """per-sample power-to-dB with a per-sample top_db floor"""
        db = 10.0 * torch.log10(power.clamp(min=1e-10))
        if self.cfg.top_db is not None:
            peak = db.amax(dim=(-2, -1), keepdim=True)
            db = torch.maximum(db, peak - self.cfg.top_db)
        return db

    @torch.no_grad()
    def forward(self, audio, training=False):
        cfg = self.cfg
        z = self.spec(audio)     #(B, F, T)

        if training and cfg.p_stretch > 0 and np.random.rand() < cfg.p_stretch:
            rate = float(np.random.uniform(*cfg.stretch_range))
            z = self.stretch(z, rate)

        # fixed frame count - before the resize, or the resize undoes the stretch
        if z.shape[-1] > self.n_frames:
            z = z[..., :self.n_frames]
        elif z.shape[-1] < self.n_frames:
            z = nn.functional.pad(z, (0, self.n_frames - z.shape[-1]))

        mel = self._to_db(self.mel_scale(z.abs().pow(2)))
        mel = mel.unsqueeze(1)   #(B, 1, mel, T)

        mel = nn.functional.interpolate(
            mel, size=(cfg.img_size, cfg.img_size),
            mode='bilinear', align_corners=False)

        #per sample norm
        mean = mel.mean(dim=(1, 2, 3), keepdim=True)
        std = mel.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        mel = (mel - mean) / std

        #masking with the spec's mean
        if training and cfg.p_mask > 0 and np.random.rand() < cfg.p_mask:
            for _ in range(cfg.n_masks):
                mel = self.freq_mask(mel, mask_value=0.0)
                mel = self.time_mask(mel, mask_value=0.0)

        return mel.repeat(1, 3, 1, 1)



# models
def build_model(cfg):
    if cfg.backbone == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=cfg.dropout),
                                 nn.Linear(in_features, cfg.num_classes))
    elif cfg.backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=cfg.dropout),
                                 nn.Linear(in_features, cfg.num_classes))
    elif cfg.backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=cfg.dropout),
                                         nn.Linear(in_features, cfg.num_classes))
    else:
        raise ValueError(f'unknown backbone: {cfg.backbone}')

    if cfg.init_checkpoint:
        # pretrained backbone from a previous corpus; the head is re-initialised
        state = torch.load(cfg.init_checkpoint, map_location='cpu',
                           weights_only=False)
        state = state.get('model_state_dict', state)
        state = {k: v for k, v in state.items()
                 if not k.startswith(('fc.', 'classifier.'))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'loaded {cfg.init_checkpoint}: {len(missing)} missing, '
              f'{len(unexpected)} unexpected keys')
    return model


def set_backbone_trainable(model, trainable):
    head_prefixes = ('fc.', 'classifier.')
    for name, param in model.named_parameters():
        if not name.startswith(head_prefixes):
            param.requires_grad = trainable


def freeze_backbone_bn(model):
    """put every BatchNorm layer into eval mode"""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()



# training and evaluation
@torch.no_grad()
def evaluate(model, loader, cfg, mel_transform, device, desc='Validation'):
    """switch off dropout and disable augmentations"""
    model.eval()
    all_targets, all_probs = [], []
    running_loss = 0.0

    for chunks, targets in tqdm(loader, desc=desc):
        chunks = chunks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        mel = mel_transform(chunks, training=False)
        with torch.autocast('cuda', enabled=cfg.use_amp and device.type == 'cuda'):
            outputs = model(mel)


        hard = targets if targets.dim() == 1 else targets.argmax(1)
        loss = nn.functional.cross_entropy(outputs.float(), hard)
        probs = torch.softmax(outputs.float(), dim=1)
        hard = nn.functional.one_hot(hard, cfg.num_classes).float()

        running_loss += loss.item() * chunks.size(0)
        all_targets.append(hard.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    targets_np = np.concatenate(all_targets)
    probs_np = np.concatenate(all_probs)
    cols = [str(i) for i in range(cfg.num_classes)]
    cmap = padded_cmap(pd.DataFrame(targets_np, columns=cols),
                       pd.DataFrame(probs_np, columns=cols))
    return running_loss / len(loader.dataset), cmap


def train(cfg, model, load_train, load_val, optimizer, scheduler, mel_transform, colour_jiggle, pitch_shifters, device):
    """train on the augmented batches, validate, add a checkpoint for an improved epoch, step the lr schedule, and finally return a chkpt with the best epoch"""  
    
   
    mixup = Mixup(mix_beta=cfg.mix_beta)    #mixup is applied per batch 
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp and device.type == 'cuda')    
    best_metric, wait, best_state, history = 0.0, 0, None, []   #keep track of some values for early stopping
    
    #freezes layers based on what strategy is chosen
    if cfg.freeze_strategy in ('head_only', 'warmup'):
        set_backbone_trainable(model, False)
    
    #unfreeze layers and train from the same lr and epoch counbt
    for epoch in range(cfg.num_epochs):
        if cfg.freeze_strategy == 'warmup' and epoch == cfg.freeze_epochs:
            print('unfreezing backbone')
            set_backbone_trainable(model, True)
        
       
        model.train()   #sets dropout and batch normalisation on

        if (cfg.freeze_strategy == 'head_only'
                or (cfg.freeze_strategy == 'warmup' and epoch < cfg.freeze_epochs)):
            freeze_backbone_bn(model)

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (chunks, targets) in enumerate(
                tqdm(load_train, desc=f'Epoch {epoch+1}/{cfg.num_epochs} - Training')):
                    
            #send audio chunks and labels to the gpu
            chunks = chunks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            #one hot encode the labels
            if targets.dim() == 1:
                targets = nn.functional.one_hot(targets, cfg.num_classes).float()
            
            #apply some augmentations if theyre set
            if cfg.p_mixup > 0 and np.random.rand() < cfg.p_mixup:
                chunks, targets = mixup(chunks, targets)

            chunks = apply_pitch_shift(chunks, pitch_shifters, cfg.p_pitch)

            mel = mel_transform(chunks, training=True)
            if cfg.p_jiggle > 0:
                mel = colour_jiggle(mel)
            
            #apply label smoothing if it's set on, and calculate the loss values using the smoothed labels
            with torch.autocast('cuda', enabled=cfg.use_amp and device.type == 'cuda'):
                outputs = model(mel)
                smoothed = targets
                if cfg.label_smoothing > 0:
                    smoothed = (targets * (1 - cfg.label_smoothing)
                                + cfg.label_smoothing / cfg.num_classes)
                loss = torch.mean(torch.sum(
                    -smoothed * nn.functional.log_softmax(outputs.float(), dim=1),
                    dim=1))
            
            scaler.scale(loss).backward() #backward pass through the gradient scaler
            scaler.step(optimizer) #unscale the gradients
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * chunks.size(0)

        train_loss = running_loss / len(load_train.dataset)
        val_loss, val_cmap = evaluate(model, load_val, cfg, mel_transform, device,
                                      desc=f'Epoch {epoch+1} - Validation')
        
        #save checkpoints if there is an improvement in val cmap
        if val_cmap > best_metric + 1e-4:
            best_metric, wait = val_cmap, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch, 'cfg': asdict(cfg),
                        'model_state_dict': model.state_dict(),
                        'val_cmap': val_cmap},
                       os.path.join(cfg.work_dir, f'{cfg.run_name}_best.pth'))
            torch.save({'cfg': asdict(cfg), 'model_state_dict': model.state_dict()},
                       os.path.join(cfg.work_dir, f'{cfg.run_name}_weights.pth'))
        else:
            wait += 1
        
        #cosinne annealing is used anyways
        if scheduler is not None:
            scheduler.step()
        
        #print training progress and save epoch history to plot graphs
        print(f'Epoch [{epoch+1}/{cfg.num_epochs}] train_loss: {train_loss:.4f}, '
              f'val_loss: {val_loss:.4f}, val_cmap: {val_cmap:.4f}, '
              f'lr: {optimizer.param_groups[0]["lr"]:.2e}')

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_cmap': val_cmap})
        pd.DataFrame(history).to_csv(
            os.path.join(cfg.work_dir, f'{cfg.run_name}_history.csv'), index=False)

        if wait >= cfg.patience:
            print('Early stopping!')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best = max(history, key=lambda h: h['val_cmap'])
    print(f'best val_cmap: {best["val_cmap"]:.4f} (epoch {best["epoch"]})')
    return model, best


def log_run(cfg, best):
    """append one row per run, every config field plus the metrics."""
    os.makedirs(os.path.dirname(cfg.results_csv) or '.', exist_ok=True)
    row = asdict(cfg)
    row.update({k: v for k, v in best.items()})
    row.update({'finished': time.strftime('%Y-%m-%d %H:%M'),
                'pvt_cmap': '', 'pub_cmap': ''})

    frame = pd.DataFrame([row])
    if os.path.exists(cfg.results_csv):
        existing = pd.read_csv(cfg.results_csv)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(cfg.results_csv, index=False)
    print(f'logged run to {cfg.results_csv} ({len(frame)} runs)')



# main
def main():
    cfg = build_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)
    os.makedirs(cfg.work_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    print(f'run: {cfg.run_name}, device: {device}')
    with open(os.path.join(cfg.work_dir, f'{cfg.run_name}_config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- labels + splitting
    metadata = pd.read_csv(cfg.metadata_csv)
    species = sorted(metadata['primary_label'].unique())
    label2idx = {label: idx for idx, label in enumerate(species)}
    
    #link the audio chunks to the labels assigned to the parent recordings
    every_path, every_label = link_labels(cfg.audio_dir, metadata, label2idx)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        every_path, every_label, test_size=0.2, random_state=42)  # fixed, not cfg.seed

    tag = f'{int(cfg.chunk_secs)}s'
    train_samples = index_chunks(train_paths, train_labels, cfg.sr, cfg.chunk_secs,
                                 cfg.train_overlap,
                                 os.path.join(cfg.cache_dir, f'train_{tag}.pkl'))
                                 
    val_samples = index_chunks(val_paths, val_labels, cfg.sr, cfg.chunk_secs, 1.0,
                               os.path.join(cfg.cache_dir, f'val_{tag}.pkl'))

    train_dataset = ChunkDataset(train_samples, cfg,
                                 augment=build_waveform_augment(cfg))
    val_dataset = ChunkDataset(val_samples, cfg)  # augment=None

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                         pin_memory=True, worker_init_fn=worker_init)
    load_train = DataLoader(train_dataset, shuffle=True, drop_last=True,
                            prefetch_factor=4, **loader_kwargs)
    load_val = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    #--- sanity checks
    img, label = next(iter(load_train))
    print(f'waveform max: {img.max():.6f}, mean: {img.mean():.6f}')
    print(f'num zero chunks: {(img.abs().sum(dim=1) == 0).sum().item()} / {len(img)}')

    mel_transform = MelSpecTransform(cfg).to(device)
    colour_jiggle = Kornia.ColorJiggle(brightness=cfg.jiggle_brightness,
                                       contrast=cfg.jiggle_contrast, p=cfg.p_jiggle)
    pitch_shifters = (build_pitch_shifters(device, int(cfg.chunk_secs * cfg.sr))
                      if cfg.p_pitch > 0 else None)

    mel = mel_transform(img.to(device), training=False)
    print(f'batch shape: {tuple(mel.shape)}, range: [{mel.min():.3f}, {mel.max():.3f}]')
    
    per_sample_means = mel.mean(dim=(1, 2, 3))
    print(f'per-sample mean spread: {per_sample_means.std():.2e} (should be close to 0)')
    
    display_spectrogram(mel[0, 0].cpu().numpy(), cfg.sr, cfg.hop_length,
                        title=f'{cfg.run_name}',
                        out_path=os.path.join(cfg.work_dir, f'{cfg.run_name}_spec.png'))
    del mel, img

    #train the model 
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    model, best = train(cfg, model, load_train, load_val, optimizer, scheduler,
                        mel_transform, colour_jiggle, pitch_shifters, device)
    log_run(cfg, best)


if __name__ == '__main__':
    main()

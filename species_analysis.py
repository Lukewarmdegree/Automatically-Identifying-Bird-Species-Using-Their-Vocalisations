#!/usr/bin/env python
# coding: utf-8
"""
per-class average precision for a finished checkpoint.
runs the validation split once, decomposes the cmAP into its per-species
components, and joins them against the number of training recordings per
species. writes a csv and, with --plot, a scatter of AP against sample count.
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn.metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments import (Config, ChunkDataset, MelSpecTransform, build_model,
                       index_chunks, link_labels, set_seed, worker_init)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default=None,
                        help='output csv (default: work/<run>_per_class.csv)')
    parser.add_argument('--plot', action='store_true',
                        help='also write a scatter of AP against sample count')
    parser.add_argument('--padding', type=int, default=5,
                        help='padding factor; 5 matches the competition metric')
    return parser.parse_args()


def load_config(checkpoint):
    cfg = Config()
    for key, value in checkpoint.get('cfg', {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
    return cfg


def padded_ap_per_class(targets, probs, padding_factor=5):
    """per-species AP with the same padding the competition metric applies.

    padded cmAP adds artificial positives padded by a factor scored 1.0 to every
    class before averaging, which caps how badly a rare species can score. the
    mean of what this returns is exactly the padded cmAP.
    """
    n_classes = targets.shape[1]
    pad_t = np.ones((padding_factor, n_classes))
    pad_p = np.ones((padding_factor, n_classes))
    padded_targets = np.concatenate([targets, pad_t], axis=0)
    padded_probs = np.concatenate([probs, pad_p], axis=0)
    return sklearn.metrics.average_precision_score(
        padded_targets, padded_probs, average=None)


@torch.no_grad()
def collect_predictions(model, loader, mel_transform, cfg, device):
    model.eval()
    all_labels, all_probs = [], []
    for chunks, labels in tqdm(loader, desc='Validation'):
        chunks = chunks.to(device, non_blocking=True)
        mel = mel_transform(chunks, training=False)
        with torch.autocast('cuda', enabled=cfg.use_amp and device.type == 'cuda'):
            logits = model(mel)
        probs = (torch.sigmoid(logits.float()) if cfg.loss_type == 'bce'
                 else torch.softmax(logits.float(), dim=1))
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(checkpoint)
    set_seed(cfg.seed)
    run_name = cfg.run_name
    print(f'{run_name}: backbone={cfg.backbone} loss={cfg.loss_type}')

    #the same split every run used
    metadata = pd.read_csv(cfg.metadata_csv)
    species = sorted(metadata['primary_label'].unique())
    label2idx = {label: idx for idx, label in enumerate(species)}

    every_path, every_label = link_labels(cfg.audio_dir, metadata, label2idx)
    _, val_paths, _, val_labels = train_test_split(
        every_path, every_label, test_size=0.2, random_state=42)

    tag = f'{int(cfg.chunk_secs)}s'
    val_samples = index_chunks(val_paths, val_labels, cfg.sr, cfg.chunk_secs, 1.0,
                               os.path.join(cfg.cache_dir, f'val_{tag}.pkl'))
    loader = DataLoader(ChunkDataset(val_samples, cfg), batch_size=cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, worker_init_fn=worker_init)

    #loading the model
    model = build_model(cfg)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)

    labels, probs = collect_predictions(model, loader, mel_transform=
                                        MelSpecTransform(cfg).to(device),
                                        cfg=cfg, device=device)
    targets = np.eye(cfg.num_classes)[labels.astype(int)]

    #--- per-class metrics
    ap_padded = padded_ap_per_class(targets, probs, args.padding)
    ap_raw = sklearn.metrics.average_precision_score(targets, probs, average=None)

    train_counts = metadata['primary_label'].value_counts()
    val_chunks = pd.Series(labels).value_counts()

    table = pd.DataFrame({
        'species': species,
        'padded_ap': ap_padded,
        'raw_ap': ap_raw,
        'train_recordings': [train_counts.get(s, 0) for s in species],
        'val_chunks': [int(val_chunks.get(i, 0)) for i in range(cfg.num_classes)],
    }).sort_values('padded_ap')

    out_path = args.out or os.path.join(cfg.work_dir, f'{run_name}_per_class.csv')
    table.to_csv(out_path, index=False)

    #--- summary
    print(f'\npadded cmAP (mean of padded_ap): {table.padded_ap.mean():.4f}')
    print(f'species with no validation chunks: {(table.val_chunks == 0).sum()}')

    scored = table[table.val_chunks > 0]
    rho = scored.padded_ap.corr(scored.train_recordings, method='spearman')
    pearson = scored.padded_ap.corr(np.log10(scored.train_recordings + 1))
    print(f'\nAP vs training recordings, over {len(scored)} species:')
    print(f'  spearman rho          : {rho:+.3f}')
    print(f'  pearson (log10 counts): {pearson:+.3f}')

    print('\n10 worst species:')
    print(scored.head(10).to_string(index=False))
    print('\n10 best species:')
    print(scored.tail(10).to_string(index=False))

if __name__ == '__main__':
    main()

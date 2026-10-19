#!/usr/bin/env python
# coding: utf-8
"""
continue training a finished checkpoint on the held-out 20% validation split.

"python retrain_on_val.py --ckpt work/r50_best.pth --run r50_valft --epochs 2"

loads the config stored in the checkpoint, rebuilds the model and the exact
augmentation stack that run used, then trains on the validation recordings only.
no validation is performed during these epochs, so there is no early stopping and
no checkpoint selection - the weights after the final epoch are what gets saved.
"""

import argparse
import copy
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

#import yhe variables and what not from experiments.py
from experiments import (Config, ChunkDataset, MelSpecTransform, Mixup,
                       build_model, build_pitch_shifters, build_waveform_augment,
                       apply_pitch_shift, index_chunks, link_labels, set_seed,
                       worker_init, freeze_backbone_bn, Kornia)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='checkpoint to continue from')
    parser.add_argument('--run', required=True, help='name for the fine-tuned run')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=None,
                        help='defaults to one tenth of the original lr')
    return parser.parse_args()


def load_config(checkpoint):
    stored = checkpoint.get('cfg', {})
    cfg = Config()
    for key, value in stored.items():
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
    return cfg


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(checkpoint)
    original_run = cfg.run_name
    cfg.run_name = args.run
    set_seed(cfg.seed)

    #retraining continues from converged weights, so the original lr would undo them
    lr = args.lr if args.lr is not None else cfg.lr / 10

    print(f'continuing {original_run} -> {args.run}')
    print(f'backbone={cfg.backbone} epochs={args.epochs} lr={lr:.2e}')

    os.makedirs(cfg.work_dir, exist_ok=True)
    with open(os.path.join(cfg.work_dir, f'{args.run}_config.json'), 'w') as f:
        json.dump({**asdict(cfg), 'finetuned_from': args.ckpt,
                   'finetune_epochs': args.epochs, 'finetune_lr': lr}, f, indent=2)

    #the same split as every other run
    metadata = pd.read_csv(cfg.metadata_csv)
    species = sorted(metadata['primary_label'].unique())
    label2idx = {label: idx for idx, label in enumerate(species)}
    every_path, every_label = link_labels(cfg.audio_dir, metadata, label2idx)
    _, val_paths, _, val_labels = train_test_split(
        every_path, every_label, test_size=0.2, random_state=42)
    print(f'{len(val_paths)} recordings in the held-out split')

    #chunked with the TRAINING overlap, not the validation stride, because these recordings are now training data
    tag = f'{int(cfg.chunk_secs)}s_ft'
    samples = index_chunks(val_paths, val_labels, cfg.sr, cfg.chunk_secs,
                           cfg.train_overlap,
                           os.path.join(cfg.cache_dir, f'valft_{tag}.pkl'))

    dataset = ChunkDataset(samples, cfg, augment=build_waveform_augment(cfg))
    
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        drop_last=True, num_workers=cfg.num_workers,
                        pin_memory=True, prefetch_factor=4,
                        worker_init_fn=worker_init)

    #load the model's 
    model = build_model(cfg)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)

    mel_transform = MelSpecTransform(cfg).to(device)
    mixup = Mixup(mix_beta=cfg.mix_beta)
    pitch_shifters = (build_pitch_shifters(device, int(cfg.chunk_secs * cfg.sr))
                      if cfg.p_pitch > 0 else None)
    colour_jiggle = None
    if cfg.p_jiggle > 0:
        colour_jiggle = Kornia.ColorJiggle(brightness=cfg.jiggle_brightness,
                                           contrast=cfg.jiggle_contrast,
                                           p=cfg.p_jiggle)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda',
                                  enabled=cfg.use_amp and device.type == 'cuda')

    #trainthe model, same as in the exoeriments
    history = []
    for epoch in range(args.epochs):
        model.train()
        if cfg.freeze_strategy == 'head_only':
            freeze_backbone_bn(model)

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for chunks, targets in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            chunks = chunks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets = nn.functional.one_hot(targets, cfg.num_classes).float()

            if cfg.p_mixup > 0 and np.random.rand() < cfg.p_mixup:
                chunks, targets = mixup(chunks, targets)
            chunks = apply_pitch_shift(chunks, pitch_shifters, cfg.p_pitch)

            mel = mel_transform(chunks, training=True)
            if colour_jiggle is not None:
                mel = colour_jiggle(mel)

            with torch.autocast('cuda', enabled=cfg.use_amp and device.type == 'cuda'):
                outputs = model(mel)
                
                smoothed = targets
                if cfg.label_smoothing > 0:
                    smoothed = (targets * (1 - cfg.label_smoothing)
                                + cfg.label_smoothing / cfg.num_classes)
                loss = torch.mean(torch.sum(
                    -smoothed * nn.functional.log_softmax(outputs.float(), dim=1),
                    dim=1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * chunks.size(0)

        train_loss = running_loss / len(loader.dataset)
        print(f'Epoch [{epoch+1}/{args.epochs}] train_loss: {train_loss:.4f}')
        history.append({'epoch': epoch + 1, 'train_loss': train_loss})

        # saved both epochs, but only the 2nd one will be used 
        torch.save({'cfg': asdict(cfg), 'model_state_dict': model.state_dict(),
                    'finetuned_from': args.ckpt, 'finetune_epoch': epoch + 1},
                   os.path.join(cfg.work_dir, f'{args.run}_e{epoch+1}_weights.pth'))

    pd.DataFrame(history).to_csv(
        os.path.join(cfg.work_dir, f'{args.run}_history.csv'), index=False)
    print(f'done. submit work/{args.run}_e{args.epochs}_weights.pth')


if __name__ == '__main__':
    main()

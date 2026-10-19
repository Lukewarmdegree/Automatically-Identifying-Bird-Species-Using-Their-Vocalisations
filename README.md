# Automatically-Identifying-Bird-Species-Using-Their-Vocalisations
This is the training pipeline for classifying the 264 bird species of the BirdCLEF 2023 Kaggle competition from 5-second audio chunks, using ImageNet-pretrained CNNs on log-mel spectrograms

Recordings are split 80/20 at the file level (fixed seed, independent of the run seed), then sliced into 5s chunks with 50% overlap for training, none for validation. Chunk offsets are indexed once and cached, and the audio is read lazily by seeking into each file using SoundFile.

Each training step:

1. CPU operations: Augmentations are applied to the audio using Audiomentations, including Gaussian noise, pink/brown noise, cyclic shift, and random gain adjustment.
2. GPU operations: Mixup and ±2 semitone pitch shift with a rational ratio, which keeps the resampling kernel tiny to avoid running out of memory.
3. Spectrogram generation: Short Time Foruier Transform → time stretch → 128-band log mels (40 Hz–15 kHz) → resize spectrogram to 224×224 → per-sample normalisation → SpecAugment masking using TorchAudio → 3 channels for the model.
4. Colour jiggle is applied using Kornia, then the backbone: ResNet18, ResNet50 or EfficientNet-B0.

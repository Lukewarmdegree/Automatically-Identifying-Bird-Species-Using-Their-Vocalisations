# Automatically-Identifying-Bird-Species-Using-Their-Vocalisations
This is the training pipeline for classifying the 264 bird species of the BirdCLEF 2023 Kaggle competition from 5-second audio chunks, using ImageNet-pretrained CNNs on log-mel spectrograms.

Recordings are split 80/20 at the file level (fixed seed, independent of the run seed), then sliced into 5s chunks with 50% overlap for training, none for validation. Chunk offsets are indexed once and cached, and the audio is read lazily by seeking into each file using SoundFile.

Each training step:
1. CPU operations: Augmentations are applied to the audio using Audiomentations, including Gaussian noise, pink/brown noise, cyclic shift, and random gain adjustment.
2. GPU operations: Mixup and ±2 semitone pitch shift with a rational ratio, which keeps the resampling kernel tiny to avoid running out of memory.
3. Spectrogram generation: Short Time Foruier Transform → time stretch → 128-band log mels (40 Hz–15 kHz) → resize spectrogram to 224×224 → per-sample normalisation → SpecAugment masking using TorchAudio → 3 channels for the model.
4. Colour jiggle is applied using Kornia, then the backbone: ResNet18, ResNet50 or EfficientNet-B0.

NB: The dataset is not included in the repo and must be downloaded from BirdCLEF's official website!

https://www.kaggle.com/competitions/birdclef-2023/data

Running Experiments:

All experimental configurations are saved in JSON files to make it easier to build on other experiments and modify configurations.

python experiments.py --run baseline (this runs a ResNet-18 without any augmentations or regularisation set. --run sets the name for the experiment)

python experiments.py --run gauss_only --set p_gauss=0.5 p_gain=0.0 p_shift=0.0 (--set is used to set a value for the specified parameter. p=0 means the augmentation is turned off. Any other number indicates the probability of that augmentation being applied, and hence switches it on for that experiment.)

python experiments.py --run loo_mixup --from work/full_stack_config.json --set p_mixup=0.0 (--from starts an experiment from a configuration saved as a JSON file. This makes it easier to stack augmentations on top of each other or remove augmentations from a stack.)

Parameters that can be set:

| Group | Fields |
|---|---|
| Audio | `sr`, `chunk_secs`, `train_overlap` (stride as a fraction of chunk length) |
| Optimisation | `backbone`, `batch_size`, `num_epochs`, `lr`, `weight_decay`, `label_smoothing`, `dropout`, `loss_type` (`ce`/`bce`), `freeze_strategy` (`none`/`head_only`/`warmup`), `patience` |
| Spectrogram | `n_mels`, `n_fft`, `hop_length`, `f_min`, `f_max`, `top_db`, `img_size` |
| Augmentation | `p_gauss`, `p_colour_noise`, `p_shift`, `p_gain`, `p_mixup`, `p_stretch`, `p_mask`, `p_pitch`, `p_jiggle`, plus their ranges |
| Initialisation | `init_checkpoint`: backbone weights from a previous run. The head is re-initialised |

Outputs:

| Path | Contents |
|---|---|
| `work/<run>_config.json` | Full config, reusable with `--from` |
| `work/<run>_best.pth` | Best-epoch checkpoint with epoch, config and val cmAP |
| `work/<run>_weights.pth` | Best-epoch weights and config |
| `work/<run>_history.csv` | Per-epoch train loss, val loss and val cmAP |
| `work/<run>_spec.png` | Sanity-check spectrogram from the first batch |
| `work/bad_files.txt` | Unreadable audio files (replaced with silence) |
| `cache/train_5s.pkl`, `cache/val_5s.pkl` | Cached chunk index |
| `results/runs.csv` | One row per run; `pvt_cmap`/`pub_cmap` are left blank for leaderboard scores |


Submitting Trained Models to Kaggle:

My submission history with all the trained models is available on my Kaggle notebook

https://www.kaggle.com/code/kylenkomo/thesis-model-submission

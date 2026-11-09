# UniCue: A Unified Benchmark for Multi-Cue Target Speaker Extraction

Official code for *UniCue: A Unified Benchmark for Multi-Cue Target Speaker Extraction with a Shared Conditioning Interface* (ACM MM '26).

👉 [[Demo](https://uni-cue.github.io/UniCue-demo/)] · [[DOI](https://doi.org/10.1145/3767308.3835611)]

UniCue is a unified benchmark for multi-cue target speaker extraction (TSE), with cues from four modalities — **video**, **pre-enrollment speech**, **text**, and **spatial**. It instantiates **UniCueNet**, a shared conditioning interface on a SepFormer-style separator that projects heterogeneous cues into a common latent space and organizes them into a unified cue memory.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Uni-cue/UniCue.git
cd UniCue
```

2. Install PyTorch first, matching your CUDA (https://pytorch.org), then the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation

UniCue is a speech-in-video benchmark: two-speaker mixtures are simulated with 4-channel RIRs (gpuRIR) and paired with four cue modalities — visual (face/body), self-enrollment speech, text (transcript, emotion, gender, language), and spatial (azimuth, distance) — with speaker-disjoint train/val/test splits. A small example mixture is provided in `data/example/`.

See [data/README.md](data/README.md) for the full construction pipeline.

## Training

1. Set `train_scp` / `eval_scp` in `configs/all_sepformer.yaml` to your data root.
2. Run:

```bash
python train.py run --config configs/all_sepformer.yaml
```



## Acknowledgements

**Paper.** This work was supported by the National Natural Science Foundation of China under Grant No. 62306029, the Beijing Natural Science Foundation under Grant No. L233032, the Young Elite Scientists Sponsorship Program of the Beijing High Innovation Plan and the Shenzhen Loop Area Institute under Grant No. FPF10120250012.

**Code.** This code builds on [SpeechBrain](https://github.com/speechbrain/speechbrain), [LAION-CLAP](https://github.com/LAION-AI/CLAP), [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR), and [yutto](https://github.com/yutto-dev/yutto). We thank their maintainers and contributors.

## License

The paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code released for research purposes.

> **Disclaimer:** This repository is for research purposes only. Users are responsible for complying with the video platform's Terms of Service and applicable local laws; downloaded videos must not be redistributed.

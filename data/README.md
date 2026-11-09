# Data Preparation

Build the UniCue dataset from raw videos. Requires `ffmpeg`/`ffprobe` and a GPU (`gpuRIR`).
The source videos are mainly from [Bilibili](https://www.bilibili.com/); `download_videos.py` uses [yutto](https://github.com/yutto-dev/yutto) to download them.

Run in order (edit the paths inside each script first):

1. **Prepare the URL list.** Create `video_all.txt` with one video per line — `<video_url> <gender> <lang>` — and set its path in `URLS_FILE` (in `download_videos.py`).
2. **Download videos.** `python download_videos.py`
3. **Split into segments.** `python split_videos.py`
4. **Filter segments (face / hand / speech).** `python video_filter.py`
5. **Extract 16 kHz mono audio.** `python extract_wav.py`
6. **Label emotion + ASR transcript.** `python labels.py`
7. **Generate 4-channel mixtures.** `python create_mix.py`

See `data/example/` for a sample mixture produced by step 7.

Face/body AV embeddings (`av_embedding/`) come from a separate extractor

Final data root expected by `train.py`:

```
data_root/
├── multi_mixtures/   # train/ val/ test/ -> mix_XXXXXX/
├── audio_clean/
└── av_embedding/
```

> **Disclaimer:** This repository is for research purposes only. Users are responsible for complying with the video platform's Terms of Service and applicable local laws; downloaded videos must not be redistributed.

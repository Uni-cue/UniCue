#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import sys

def extract_wav_from_mp4(video_path: str, audio_path: str, sampling_rate: int) -> None:
    """
    使用 ffmpeg 把 mp4 中的音频提取为 16kHz、单声道、16-bit PCM WAV。
    如果目标文件已存在且采样率匹配，直接返回；否则重新提取/重采样。
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    # 检查输入文件
    if not os.path.exists(video_path):
        raise RuntimeError(f"Input video not found: {video_path}")

    # 目标采样率
    target_sr = sampling_rate

    # 如果已有 wav，检查采样率
    need_resample = False
    if os.path.exists(audio_path):
        try:
            # 使用 ffmpeg 读取采样率（不加载完整音频）
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate", "-of", "csv=p=0", audio_path],
                capture_output=True, text=True, check=True
            )
            current_sr = int(probe.stdout.strip())
            if current_sr != target_sr:
                need_resample = True
        except Exception:
            need_resample = True
    else:
        need_resample = True

    # 提取或重采样
    if need_resample:
        tmp_path = audio_path + ".tmp.wav"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn",                     # 不包含视频
            "-ar", str(target_sr),     # 目标采样率
            "-ac", "1",                # 单声道
            "-acodec", "pcm_s16le",    # 16-bit PCM
            tmp_path
        ]
        try:
            subprocess.run(cmd, check=True)
            os.replace(tmp_path, audio_path)
        except subprocess.CalledProcessError as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise RuntimeError(f"ffmpeg failed: {e}")

def main(args):
    root = args.data_direc
    audio_root = args.audio_data_direc

    print(f"Scanning {root} for .mp4 files...")
    # os.walk 会自动递归遍历新的目录结构
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".mp4"):
                src = os.path.join(dirpath, fname)
                # 保持原相对路径结构 (例如: subdir_name/video.mp4)
                rel = os.path.relpath(src, root)
                dst = os.path.join(audio_root, os.path.splitext(rel)[0] + ".wav")

                try:
                    extract_wav_from_mp4(src, dst, args.sampling_rate)
                    print(f"Extracted: {dst}")
                except Exception as e:
                    print(f"Failed {src}: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从指定目录递归提取所有 mp4 的音频为 16kHz WAV"
    )
    
    # ==== 路径调整 ====
    # [修改] 输入目录改为 filtered_output_all/video_filtered
    parser.add_argument("--data_direc", type=str, 
                        default="/path/to/your/yutto/filtered_output_all/filtered_videos/",
                        help="根目录（包含各个视频子目录）")
    
    # [修改] 输出目录改为 filtered_output_all/audio_clean
    parser.add_argument("--audio_data_direc", type=str,
                        default="/path/to/your/yutto/filtered_output_all/audio_clean/",
                        help="提取的 wav 存放目录（会自动创建子目录）")
    
    parser.add_argument(
        "--sampling_rate", type=int, default=16000,
        help="目标采样率（默认 16000）"
    )
    args = parser.parse_args()
    main(args)
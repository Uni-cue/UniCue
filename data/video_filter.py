#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_segment_filter.py

对已切片的视频（每个原视频一个子目录，子目录内是 segment_XXXX.mp4）做自动筛选，
输出 filtered_output/{original_dir}/segment_XXXX.mp4

筛选条件（启发式）：
 - 单人（大多数采样帧检测到恰好 1 张人脸）
 - 人脸清晰（人脸 bbox 面积相对帧面积不太小；Laplace 方差用作清晰度）
 - 有手（检测到手的帧比例达到阈值 => 有手势）
 - 有语音（VAD 检测到语音的比例超过阈值）

修改说明：
 - 移除了 train/test 划分，所有结果直接输出到目标文件夹下的对应子目录。
 - [优化] 将 MediaPipe 模型初始化移至循环外，避免重复加载导致日志刷屏和性能下降。

要求：
 pip install mediapipe opencv-python moviepy webrtcvad tqdm numpy

用法示例：
 python video_segment_filter.py \
   --input_dir split_output \
   --output_dir /path/to/your/yutto/filtered_output_all/video_filtered \
   --sample_fps 2 \
   --min_single_face_frac 0.85 \
   --min_speech_frac 0.08

"""

import argparse
import tempfile
import shutil
import random
import os
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
# from moviepy.video.io.VideoFileClip import VideoFileClip # 原代码引用，保持保留
import wave
import webrtcvad
from tqdm import tqdm

# -------------------------- 辅助函数 (保持不变) --------------------------

def compute_vad_speech_ratio(wav_path: Path, aggressiveness: int = 2, frame_ms: int = 30) -> float:
    """使用 webrtcvad 计算有语音的帧比例（0.0 - 1.0）
    wav_path: 16-bit PCM WAV 文件，采样率建议 16000
    """
    wav_path = str(wav_path)
    wf = wave.open(wav_path, 'rb')
    sample_rate = wf.getframerate()
    assert wf.getsampwidth() == 2, "webrtcvad 需要 16-bit PCM"
    assert wf.getnchannels() == 1, "需要单声道 WAV"

    vad = webrtcvad.Vad(aggressiveness)
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)  # bytes per frame (s16)

    voiced = 0
    total = 0
    while True:
        data = wf.readframes(int(sample_rate * (frame_ms / 1000.0)))
        if len(data) < frame_bytes:
            break
        is_speech = vad.is_speech(data, sample_rate)
        if is_speech:
            voiced += 1
        total += 1
    wf.close()
    if total == 0:
        return 0.0
    return voiced / total


def extract_audio_to_wav(video_path: Path, out_wav: Path, sr: int = 16000):
    """使用 ffmpeg 或 moviepy 将视频音频导出为单声道 16-bit PCM WAV（采样率 sr）。
    确保输出符合 webrtcvad 要求：单声道、16-bit PCM
    """
    import subprocess
    import shutil

    # 尝试使用系统 ffmpeg
    if shutil.which('ffmpeg') is not None:
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn',          # 不要视频
            '-ac', '1',     # 单声道
            '-ar', str(sr), # 采样率
            '-acodec', 'pcm_s16le',
            str(out_wav)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg 提取音频失败: {e.stderr.decode(errors='ignore')}") from e
    else:
        # ffmpeg 不可用，回退使用 moviepy
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_path))
        try:
            if clip.audio is None:
                raise RuntimeError("视频没有音频轨道")
            clip.audio.set_channels(1).write_audiofile(
                str(out_wav),
                fps=sr,
                nbytes=2,
                codec='pcm_s16le'
            )
        finally:
            clip.close()

    # 验证输出 WAV 格式
    import wave
    wf = wave.open(str(out_wav), 'rb')
    nch = wf.getnchannels()
    sw = wf.getsampwidth()
    wf.close()
    if nch != 1 or sw != 2:
        raise RuntimeError('输出 WAV 不是单声道 16-bit，请检查 ffmpeg 或 moviepy 设置')


# -------------------------- 核心逻辑 (已修改优化) --------------------------

def analyze_video_frames(video_path: Path, 
                         face_detector, 
                         hands_detector, 
                         sample_fps: float = 2.0, 
                         face_min_area_frac: float = 0.02,
                         face_sharpness_thresh: float = 80.0, 
                         hand_presence_thresh: float = 0.02):
    """采样分析视频：返回统计量字典
    [修改] 增加了 face_detector, hands_detector 参数，避免函数内重复初始化
    - single_face_frac: 采样帧中恰好 1 张脸的比例
    - avg_face_area_frac: 检测到脸时的平均人脸 bbox 面积占比
    - face_sharpness_mean: Laplacian 方差平均
    - hand_frac: 检测到手的帧比例
    - frames_sampled
    - face_counts_distribution
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    if duration <= 0:
        duration = 0

    # [修改] 移除了此处的 mp_face / mp_hands 初始化，直接使用传入的 detector

    single_face_frames = 0
    total_face_area = 0.0
    face_sharpness_vals = []
    hand_frames = 0
    frames_sampled = 0
    face_counts = []

    # [修改] 移除了此处的 FaceDetection / Hands 实例化代码

    # 生成采样时间戳
    if duration <= 0:
        timestamps = [0.0]
    else:
        n_samples = max(1, int(round(duration * sample_fps)))
        timestamps = np.linspace(0, max(0.001, duration - 1e-6), n_samples)

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ret, frame = cap.read()
        if not ret:
            continue
        frames_sampled += 1

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face
        # [修改] 直接调用传入的 detector
        results = face_detector.process(rgb)
        num_faces = 0
        max_face_area = 0.0
        if results.detections:
            for det in results.detections:
                num_faces += 1
                bbox = det.location_data.relative_bounding_box
                area = max(0.0, float(bbox.width) * float(bbox.height))
                if area > max_face_area:
                    max_face_area = area
            # 面积根据相对 bbox（比例）计算
            total_face_area += max_face_area
            # 清晰度（基于最大人脸区域）
            x = int(max(0, bbox.xmin * w))
            y = int(max(0, bbox.ymin * h))
            bw = int(max(1, bbox.width * w))
            bh = int(max(1, bbox.height * h))
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            try:
                face_roi = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(face_roi, cv2.CV_64F)
                sharpness = float(lap.var())
                face_sharpness_vals.append(sharpness)
            except Exception:
                pass

        face_counts.append(num_faces)
        if num_faces == 1:
            single_face_frames += 1

        # hand
        # [修改] 直接调用传入的 detector
        hand_res = hands_detector.process(rgb)
        if hand_res.multi_hand_landmarks:
            hand_frames += 1

    # [修改] 移除了 close() 调用，由外部统一管理
    cap.release()

    single_face_frac = single_face_frames / frames_sampled if frames_sampled > 0 else 0.0
    avg_face_area_frac = (total_face_area / single_face_frames) if single_face_frames > 0 else 0.0
    face_sharpness_mean = float(np.mean(face_sharpness_vals)) if len(face_sharpness_vals) > 0 else 0.0
    hand_frac = hand_frames / frames_sampled if frames_sampled > 0 else 0.0

    return {
        'single_face_frac': single_face_frac,
        'avg_face_area_frac': avg_face_area_frac,
        'face_sharpness_mean': face_sharpness_mean,
        'hand_frac': hand_frac,
        'frames_sampled': frames_sampled,
        'face_counts': face_counts,
        'duration': duration,
    }


# -------------------------- 主流程 --------------------------

def process_dataset(input_dir: Path, output_dir: Path, sample_fps: float = 2.0,
                    min_single_face_frac: float = 0.85, face_min_area_frac: float = 0.02,
                    face_sharpness_thresh: float = 80.0, min_speech_frac: float = 0.05,
                    min_hand_frac: float = 0.02, seed: int = 42,
                    copy_files: bool = True):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    assert input_dir.exists()
    random.seed(seed)

    # 获取所有子目录（每个子目录视为一个 source id）
    subdirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]
    random.shuffle(subdirs)
    
    # [修改] 移除 train/test 划分逻辑，直接处理所有子目录
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    tmpdir = Path(tempfile.mkdtemp(prefix='segfilter_'))
    print(f"临时目录: {tmpdir}")

    accepted = []
    rejected = []

    # [优化] 在循环外部初始化模型，只初始化一次
    print("正在初始化 MediaPipe 模型...")
    mp_face = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    
    # 建立模型实例
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        # 逐个子目录处理，每个子目录内的所有 segment
        for sd in tqdm(subdirs, desc='处理子目录'):
            segs = sorted(sd.glob('*.mp4'))
            if len(segs) == 0:
                continue
            for seg in segs:
                try:
                    # 1) 提取音频到临时 wav
                    wav_tmp = tmpdir / (seg.stem + '.wav')
                    extract_audio_to_wav(seg, wav_tmp, sr=16000)
                    speech_frac = compute_vad_speech_ratio(wav_tmp, aggressiveness=2, frame_ms=30)

                    # 2) 采样分析视频帧
                    # [优化] 传入已经初始化好的模型
                    stats = analyze_video_frames(seg, 
                                                 face_detector=face_detector,
                                                 hands_detector=hands_detector,
                                                 sample_fps=sample_fps,
                                                 face_min_area_frac=face_min_area_frac,
                                                 face_sharpness_thresh=face_sharpness_thresh,
                                                 hand_presence_thresh=min_hand_frac)

                    # 3) 决策规则（启发式）
                    ok = True
                    reasons = []
                    if stats['single_face_frac'] < min_single_face_frac:
                        ok = False
                        reasons.append(f"single_face_frac={stats['single_face_frac']:.2f}<min({min_single_face_frac})")
                    if stats['avg_face_area_frac'] < face_min_area_frac:
                        ok = False
                        reasons.append(f"avg_face_area_frac={stats['avg_face_area_frac']:.3f}<min({face_min_area_frac})")
                    if stats['face_sharpness_mean'] < face_sharpness_thresh:
                        ok = False
                        reasons.append(f"face_sharpness_mean={stats['face_sharpness_mean']:.1f}<min({face_sharpness_thresh})")
                    if speech_frac < min_speech_frac:
                        ok = False
                        reasons.append(f"speech_frac={speech_frac:.2f}<min({min_speech_frac})")
                    if stats['hand_frac'] < min_hand_frac:
                        ok = False
                        reasons.append(f"hand_frac={stats['hand_frac']:.2f}<min({min_hand_frac})")

                    if ok:
                        accepted.append((sd.name, seg.name))
                        # [修改] 直接输出到 output_dir / sd.name，不再区分 train/test
                        dst_dir = output_dir / sd.name
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        if copy_files:
                            shutil.copy2(seg, dst_dir / seg.name)
                        else:
                            shutil.move(seg, dst_dir / seg.name)
                    else:
                        rejected.append((sd.name, seg.name, reasons))

                    # 清理 wav
                    try:
                        wav_tmp.unlink()
                    except Exception:
                        pass

                except Exception as e:
                    rejected.append((sd.name, seg.name, [f"error: {e}"]))
    
    finally:
        # [优化] 循环结束后（或出错时）统一关闭模型
        print("正在关闭 MediaPipe 模型...")
        face_detector.close()
        hands_detector.close()
        # 清理临时目录
        shutil.rmtree(tmpdir, ignore_errors=True)


    # 报告
    print(f"总共通过: {len(accepted)}, 拒绝: {len(rejected)}")
    print("示例拒绝原因（最多 10 条）:")
    for item in rejected[:10]:
        print(item)


# -------------------------- 命令行接口 --------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', 
                   default='/path/to/your/yutto/split_output_all',
                   help='原始切片目录，目录下每个子目录为一个视频/说话人')
    # [修改] 更新默认输出路径
    p.add_argument('--output_dir',
                   default='/path/to/your/yutto/filtered_output_all/filtered_videos',
                   help='筛选后输出目录')
    p.add_argument('--sample_fps', type=float, default=2.0)
    p.add_argument('--min_single_face_frac', type=float, default=0.85)
    p.add_argument('--face_min_area_frac', type=float, default=0.02)
    p.add_argument('--face_sharpness_thresh', type=float, default=80.0)
    p.add_argument('--min_speech_frac', type=float, default=0.05)
    p.add_argument('--min_hand_frac', type=float, default=0.02)
    # [修改] 移除了 --split_ratio 参数
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_copy', action='store_true', help='不复制，改为移动文件')

    args = p.parse_args()
    process_dataset(args.input_dir, args.output_dir, sample_fps=args.sample_fps,
                    min_single_face_frac=args.min_single_face_frac,
                    face_min_area_frac=args.face_min_area_frac,
                    face_sharpness_thresh=args.face_sharpness_thresh,
                    min_speech_frac=args.min_speech_frac,
                    min_hand_frac=args.min_hand_frac,
                    seed=args.seed,
                    copy_files=(not args.no_copy))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动分层划分与多模态混合语音数据集生成脚本
结合了 create_mix_multi_all.py 的 RIR/文件保存逻辑
以及 "Scheme A" 的自然分布分层采样策略。

输入: audio_clean/ (扁平结构: BVxxx_Gender_Language/segments...)
输出: multi_mixtures/train/, multi_mixtures/test/
"""

import os
import argparse
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import gpuRIR
import torch
import csv
import librosa
import numpy.matlib
import torch.nn as nn
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================== 配置 ====================
# 保持原有的 CSV Header 格式
INFO_HEADERS = [
    "Speaker", "ID", "Utterance", "Gender", "Text", "Emotion", "Language", 
    "Pre", "Sector"
]

mic_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# ==================== RIR 模拟器 ====================
class RIRmodel(nn.Module):
    def __init__(self):
        super(RIRmodel, self).__init__()
        self.room_sz = [6, 6, 3]
        self.circle_radius = 0.05
        self.mic_positions = self._get_mic_positions()
        self.abs_weights = [0.9] * 5 + [0.5]
        self.T60 = 0.1
        self.fs = 16000
        self.beta = gpuRIR.beta_SabineEstimation(self.room_sz, self.T60, abs_weights=self.abs_weights)
        self.Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60)
        self.Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)
        self.nb_img = gpuRIR.t2n(self.Tdiff, self.room_sz)
        self.mic_pattern = "card"
        self.orV_rcv = np.matlib.repmat(np.array([0, 1, 0]), 4, 1)

    def _get_mic_positions(self):
        mic_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        mic_positions = np.array([
            [self.room_sz[0] / 2 + self.circle_radius * np.cos(a),
             self.room_sz[1] / 2 + self.circle_radius * np.sin(a),
             1.7] for a in mic_angles
        ])
        return mic_positions

    def compute_tpd_time_domain(self, mic_pairs, region, sound_speed=343.0):
        mic_positions = np.array(self.mic_positions)
        tpd_features = []
        for p1, p2 in mic_pairs:
            delta_p = mic_positions[p1] - mic_positions[p2]
            distance = np.linalg.norm(delta_p)
            tpd = distance / sound_speed
            tpd_samples = tpd * self.fs
            tpd_features.append(tpd_samples)
        tpd_features = np.array(tpd_features).flatten()
        return tpd_features

    def get_source_positions(self, flag):
        margin = 0.3
        min_height, max_height = 1.6, min(1.9, self.room_sz[2] - margin)
        
        main_sector = random.randint(0, 7)
        if flag == 0:
            possible_values = [i for i in range(8) if i != main_sector]
            second_sector = random.choice(possible_values)
        else:
            second_sector = random.randint(0, 7)

        sector_angles = np.array([0, -45, -90, -135, 180, 135, 90, 45])
        
        def polar_to_cartesian(angle):
            radius = np.random.uniform(0.5, min(self.room_sz[0], self.room_sz[1]) / 2 - margin)
            return [self.room_sz[0] / 2 + radius * np.cos(angle),
                    self.room_sz[1] / 2 + radius * np.sin(angle),
                    np.random.uniform(min_height, max_height)]

        base_angle = np.radians(sector_angles[main_sector] + random.uniform(-22.5, 22.5))
        second_angle = np.radians(sector_angles[second_sector] + random.uniform(-22.5, 22.5))

        source_positions = np.array([polar_to_cartesian(base_angle), polar_to_cartesian(second_angle)])
        return source_positions, [main_sector, second_sector]

    def forward(self, a_tgt, a_int, flag=0):
        source_positions, sectors = self.get_source_positions(flag)

        RIRs1 = gpuRIR.simulateRIR(
            self.room_sz, self.beta, source_positions[0].reshape(1, 3), self.mic_positions,
            self.nb_img, self.Tmax, self.fs, Tdiff=self.Tdiff, orV_rcv=self.orV_rcv,
            mic_pattern=self.mic_pattern
        )
        RIRs2 = gpuRIR.simulateRIR(
            self.room_sz, self.beta, source_positions[1].reshape(1, 3), self.mic_positions,
            self.nb_img, self.Tmax, self.fs, Tdiff=self.Tdiff, orV_rcv=self.orV_rcv,
            mic_pattern=self.mic_pattern
        )

        b1 = gpuRIR.gpuRIR_bind_simulator.gpu_conv(a_tgt[None, :], RIRs1).transpose(0, 2, 1)[0]
        b2 = gpuRIR.gpuRIR_bind_simulator.gpu_conv(a_int[None, :], RIRs2).transpose(0, 2, 1)[0]
        return b1, b2, sectors

# ==================== 数据解析与划分====================

def parse_dataset_structure(root_dir):
    """
    扫描目录结构，不依赖 metadata.json 即可建立索引
    """
    data = []
    root_path = Path(root_dir)
    video_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    print(f"Scanning {len(video_dirs)} folders in {root_dir}...")
    
    for v_dir in video_dirs:
        folder_name = v_dir.name
        parts = folder_name.split('_')
        
        # 容错处理: 假设格式为 ID_Gender_Language
        # 如果格式不同，可能需要调整这里
        if len(parts) >= 3:
            lang = parts[-1]
            gender = parts[-2]
            video_id = "_".join(parts[:-2])
        else:
            print(f"[Warning] Skipping malformed folder: {folder_name}")
            continue
        
        # 扫描该文件夹下的所有音频
        segments = sorted(list(v_dir.glob("*.mp4")) + list(v_dir.glob("*.wav")))
        segments = [str(s) for s in segments]
        
        # 尝试查找 metadata.json 以备后用 (为了提取 Text/Emotion)
        meta_path = v_dir / "metadata.json"
        has_meta = meta_path.exists()

        if len(segments) > 0:
            data.append({
                "video_id": video_id,
                "folder_name": folder_name,
                "folder_path": v_dir, # Path object
                "gender": gender,
                "language": lang,
                "stratify_label": f"{gender}_{lang}", # 用于分层抽样
                "segments": segments,
                "meta_path": str(meta_path) if has_meta else None
            })
            
    return pd.DataFrame(data)

def perform_stratified_split(df, test_size=0.1, seed=42):
    """使用 sklearn 进行分层划分"""
    counts = df['stratify_label'].value_counts()
    valid_labels = counts[counts >= 2].index
    
    df_valid = df[df['stratify_label'].isin(valid_labels)]
    df_rare = df[~df['stratify_label'].isin(valid_labels)]
    
    print(f"Splitting data: {len(df_valid)} valid for stratification, {len(df_rare)} rare items added to train.")
    
    train_df, test_df = train_test_split(
        df_valid,
        test_size=test_size,
        stratify=df_valid['stratify_label'],
        random_state=seed
    )
    
    train_df = pd.concat([train_df, df_rare])
    return train_df, test_df

def extract_info_safe(meta_path, target_utt, default_gender="unknown", default_lang="unknown"):
    """
    尝试从 json 读取详细信息，如果失败则回退到文件夹名提供的信息
    """
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as rf:
                data = json.load(rf)
            for rec in data:
                # 有些 json 里的 filename 可能带路径，也可能不带，做个模糊匹配
                if Path(rec.get('filename', '')).name == Path(target_utt).name:
                    return (rec.get('text', ""), 
                            rec.get('gender', default_gender), 
                            rec.get('emotion', "neutral"), 
                            rec.get('language', default_lang))
        except:
            pass
    # 回退值
    return "", default_gender, "neutral", default_lang

def write_wav(path, data, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, sr, subtype='PCM_16')

# ==================== 核心流程 ====================

def create_mixtures_from_pool(video_list_df, split_name, args, rir_model):
    """
    根据传入的 video_list_df (Train 或 Test) 生成混合数据
    """
    dst_dir = Path(args.output_dir) / split_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换为 List of Dicts 以便快速随机访问
    video_pool = video_list_df.to_dict('records')
    pool_size = len(video_pool)
    
    n_mixtures = args.train_samples if split_name == "train" else args.test_samples
    print(f"[{split_name.upper()}] Generating {n_mixtures} mixtures from {pool_size} unique videos...")
    
    count = 0
    pbar = tqdm(total=n_mixtures)

    while count < n_mixtures:
        # 1. 随机选取两个不同的视频 (Scheme A: 自然分布)
        # 只要池子里的分布是自然的，这里随机抽取就会保持自然分布
        vid1_meta = random.choice(video_pool)
        vid2_meta = random.choice(video_pool)

        # 确保不是同一个视频文件夹 (防止同源混合)
        if vid1_meta['video_id'] == vid2_meta['video_id']:
            continue

        # 2. 随机选取片段
        seg1_path = random.choice(vid1_meta['segments'])
        seg2_path = random.choice(vid2_meta['segments'])
        
        seg1_name = Path(seg1_path).name
        seg2_name = Path(seg2_path).name

        # 3. 选 Pre (干扰前的参考，可选)
        # 简单起见，从同一个视频里再随机选一个片段，如果只有一个片段则复用
        pre1_candidates = [s for s in vid1_meta['segments'] if Path(s).name != seg1_name]
        pre1_path = random.choice(pre1_candidates) if pre1_candidates else seg1_path
        
        pre2_candidates = [s for s in vid2_meta['segments'] if Path(s).name != seg2_name]
        pre2_path = random.choice(pre2_candidates) if pre2_candidates else seg2_path
        
        pre1_name = Path(pre1_path).name
        pre2_name = Path(pre2_path).name

        # 4. 获取详细元数据 (优先读取json，否则使用文件夹名信息)
        txt1, gen1, emo1, lang1 = extract_info_safe(
            vid1_meta['meta_path'], seg1_name, 
            default_gender=vid1_meta['gender'], default_lang=vid1_meta['language']
        )
        txt2, gen2, emo2, lang2 = extract_info_safe(
            vid2_meta['meta_path'], seg2_name, 
            default_gender=vid2_meta['gender'], default_lang=vid2_meta['language']
        )


        # 5. 读取音频 & 截断
        try:
            a1, _ = librosa.load(seg1_path, sr=16000)
            a2, _ = librosa.load(seg2_path, sr=16000)
        except Exception as e:
            # print(f"Error loading: {e}")
            continue

        min_len = min(len(a1), len(a2))
        if min_len < 1600: continue # 太短跳过
        
        a1 = a1[:min_len]
        a2 = a2[:min_len]
        mixture_clean = a1 + a2 # Clean mix for reference

        # 6. dB 缩放
        db = 0.0
        if args.random_db:
            db = random.uniform(-args.max_db, args.max_db)
            scale = 10 ** (db / 20.0)
            a2 = a2 * scale # 只缩放其中一个源

        # 7. RIR & 多通道模拟
        # flag=0 强制不同扇区，保证空间分离度
        b1, b2, sectors = rir_model(a1, a2, flag=0) 

        # 8. 混合 & 归一化
        mix_4ch = (b1 + b2) * 10.0
        b1_4ch = b1 * 10.0
        b2_4ch = b2 * 10.0
        
        peak = np.max(np.abs(mix_4ch))
        if peak > 0.9:
            scale_norm = 0.9 / peak
            mix_4ch *= scale_norm
            b1_4ch *= scale_norm
            b2_4ch *= scale_norm
            a1 *= scale_norm
            a2 *= scale_norm
            mixture_clean *= scale_norm
            
        # 9. 保存文件 (结构与原脚本一致)
        mix_id = f"mix_{count:06d}"
        mix_dir = dst_dir / mix_id
        mix_dir.mkdir(exist_ok=True)
        
        write_wav(mix_dir / "spk1_clean.wav", a1)
        write_wav(mix_dir / "spk2_clean.wav", a2)
        write_wav(mix_dir / "mixture_4ch.wav", mix_4ch)
        write_wav(mix_dir / "spk1_reverb_4ch.wav", b1_4ch)
        write_wav(mix_dir / "spk2_reverb_4ch.wav", b2_4ch)
        write_wav(mix_dir / "mixture.wav", mixture_clean)

        # 10. 写入 info.csv (每个 mix 文件夹一个)
        with open(mix_dir / "info.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(INFO_HEADERS)
            
            # Speaker 字段存 VideoID
            writer.writerow([
                "Speaker 1", vid1_meta['video_id'], seg1_name, gen1, txt1, emo1, lang1, 
                pre1_name, sectors[0]
            ])
            writer.writerow([
                "Speaker 2", vid2_meta['video_id'], seg2_name, gen2, txt2, emo2, lang2, 
                pre2_name, sectors[1]
            ])
            writer.writerow(["Length (seconds)", f"{min_len / 16000:.3f}"])
            writer.writerow(["Relative dB (Speaker 2)", f"{db:+.2f}"])

        count += 1
        pbar.update(1)
    
    pbar.close()
    print(f"[{split_name.upper()}] Done! Saved to {dst_dir}")

def main(args):
    # 1. 扫描数据结构
    print("--- Parsing Dataset Structure ---")
    df = parse_dataset_structure(args.data_dir)
    print("\n--- Class Distribution (Top 10) ---")
    print(df['stratify_label'].value_counts().head(10))
    
    # 2. 分层划分 Train / Test
    print("\n--- Performing Stratified Split ---")
    train_df, test_df = perform_stratified_split(df, test_size=0.1, seed=args.seed)
    
    print(f"Total Videos: {len(df)}")
    print(f"Train Videos: {len(train_df)}")
    print(f"Test Videos:  {len(test_df)}")
    
    # 3. 初始化 RIR
    rir_model = RIRmodel()
    # rir_model = rir_model.cuda() # 如果环境配置好可开启

    # 4. 生成数据
    create_mixtures_from_pool(train_df, "train", args, rir_model)
    create_mixtures_from_pool(test_df, "test", args, rir_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改这里的默认路径为你现在的实际路径
    parser.add_argument("--data_dir", type=str, 
                        default='/path/to/your/yutto/filtered_output_all/audio_clean', 
                        help="Root path containing BVxxx_Gender_Language folders")
    parser.add_argument("--output_dir", type=str, 
                        default='/path/to/your/yutto/filtered_output_all/multi_mixtures',
                        help="Path to save multi_mixtures")
    
    parser.add_argument("--train_samples", type=int, default=20000)
    parser.add_argument("--test_samples", type=int, default=5000)
    parser.add_argument("--random_db", action="store_true", default=True)
    parser.add_argument("--max_db", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
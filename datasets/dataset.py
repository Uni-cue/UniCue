import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv
import pandas as pd
from prompt_template import ShortTemplate, build_short_style_descriptions_for_two_speakers
import soundfile as sf
import numpy as np
from tqdm import tqdm
import math
import random
import torch.nn.functional as F

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MixtureDataset(Dataset):
    """
    混合语音分离数据集（只保留 ≤6 秒的样本，且必须存在对应的 av_embedding）
    """
    
    def __init__(self, root_dir, split='test', sample_rate=16000, max_length_sec=6.0, start_index=0):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train', 'test', 'val'
            sample_rate: 目标采样率
            max_length_sec: 最大允许音频长度（秒），超过则丢弃
        """
        set_global_seed(42)
        self.root_dir = Path(root_dir)  
        self.split = split
        self.sample_rate = sample_rate
        self.max_length_sec = max_length_sec  # 新增：最大秒数
        self.start_index = max(0, int(start_index))
        self.max_length_samples = int(max_length_sec * sample_rate)
        # self.audio_dir = self.root_dir / 'audio_clean' / split
        self.audio_dir = self.root_dir / 'audio_clean' 

        # self.av_embedding_dir = self.root_dir / 'av_embedding_tmp' / split
        self.av_embedding_dir = self.root_dir / 'av_embedding' 
        
        self.data_dir = self.root_dir / 'multi_mixtures' / split

        self.cue_combinations = [
            {
                # "use_av":   (mask >> 3) & 1,
                # "use_text": (mask >> 2) & 1,
                # "use_main": (mask >> 1) & 1,
                # "use_pre":  (mask >> 0) & 1,
                "use_av":   0,
                "use_text": 1,
                "use_main": 0,
                "use_pre":  0,
            }
            for mask in range(16)
        ]

        print('cue_combinations:', self.cue_combinations)

        # 获取所有候选文件夹
        candidate_dirs = [
            d for d in self.data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('mix_')
        ]

        # 过滤：只保留 info.csv 中 length_sec <= max_length_sec 的样本
        self.sample_dirs = []
        discarded_len = 0       # 因长度丢弃
        discarded_video = 0     # 因缺失视频特征丢弃
        discarded_dim = 0       # 因维度错误(256维)丢弃

        print(f"Scanning {split} dataset and verifying feature dimensions, please wait...")

        for d in tqdm(candidate_dirs, desc="Init Dataset"):
            info_path = d / 'info.csv'
            if not info_path.exists():
                continue

            try:
                # 1. 预先读取 info 获取所有信息
                info = self.load_info(info_path)
                
                # 2. 检查长度
                if info.get("length_sec", 999) > self.max_length_sec:
                    discarded_len += 1
                    continue

                # 3.1 检查 .npz 文件是否存在！
                # 构建两个说话人的 npz 路径
                spk1_npz_rel = f"{info['speaker_1']['id']}_{info['speaker_1']['gender']}_{info['speaker_1']['language']}/{info['speaker_1']['utterance'].replace('.wav', '.npz')}"
                spk2_npz_rel = f"{info['speaker_2']['id']}_{info['speaker_2']['gender']}_{info['speaker_2']['language']}/{info['speaker_2']['utterance'].replace('.wav', '.npz')}"
                
                spk1_full_path = self.av_embedding_dir / spk1_npz_rel
                spk2_full_path = self.av_embedding_dir / spk2_npz_rel

                # print("spk1_full_path:", spk1_full_path)
                # print("spk2_full_path:", spk2_full_path)

                if not spk1_full_path.exists() or not spk2_full_path.exists():
                    # print(f"缺失视频特征: {d.name}") # 调试时可打开
                    discarded_video += 1
                    continue

                # 3.2 校验维度是否为 512
                is_dim_error = False
                
                # 只检查 'lip' 的维度
                try:
                    with np.load(spk1_full_path) as data1:
                        if data1['lip'].shape[1] != 512:
                            is_dim_error = True
                    
                    # 如果第一个已经是错的，就不查第二个了，节省时间
                    if not is_dim_error:
                        with np.load(spk2_full_path) as data2:
                            if data2['lip'].shape[1] != 512:
                                is_dim_error = True
                except Exception as e:
                    # 如果文件坏了打不开，也当做维度错误丢弃
                    is_dim_error = True
                
                if is_dim_error:
                    discarded_dim += 1
                    # print(f"发现坏数据 (维度不匹配): {d.name}") # 调试用
                    continue

                # 4. 全部通过，加入列表
                self.sample_dirs.append(d)

            except Exception as e:
                print(f"[Warning] Error processing {d.name}: {e}")
                discarded_len += 1

        print(f"[{split.upper()} Dataset Ready]")
        print(f"  - Total candidates: {len(candidate_dirs)}")
        print(f"  - Valid samples: {len(self.sample_dirs)}")
        print(f"  - Dropped (too long): {discarded_len}")
        print(f"  - Dropped (missing video): {discarded_video}")
        print(f"  - Dropped (invalid dim): {discarded_dim}  <-- Excluded")

        if len(self.sample_dirs) == 0:
            raise ValueError(f"No valid samples found! Please check data paths or length constraints.")

        self.sample_dirs = sorted(self.sample_dirs)
        if self.start_index > 0:
            if self.start_index >= len(self.sample_dirs):
                raise ValueError(
                    f"start_index {self.start_index} is out of range for {len(self.sample_dirs)} samples"
                )
            self.sample_dirs = self.sample_dirs[self.start_index:]

    def __len__(self):
        return len(self.sample_dirs)
    
    def down_up_resample(self, wav: torch.Tensor, sr_in: int = 16000, sr_down: int = 8000) -> torch.Tensor:
        """
        wav: [B, L] or [1, L]
        return: 下采样到 sr_down 再上采样回 sr_in 的波形，形状与输入一致
        """
        assert sr_in % sr_down == 0, "sr_in must be divisible by sr_down"
        x = wav.unsqueeze(1)  # [B, 1, L]
        r = sr_down / sr_in
        x_down = F.interpolate(x, scale_factor=r, mode="linear", align_corners=False)
        x_up = F.interpolate(x_down, scale_factor=1.0 / r, mode="linear", align_corners=False)
        return x_up.squeeze(1)[..., :wav.shape[-1]]


    def load_audio(self, spk1, spk2, mix):
        """使用 soundfile 加载音频 → 重采样 → 截取最小长度 → 返回 torch.Tensor"""
        # ---------- 1. 用 soundfile 读取 ----------
        spk1_np, sr1 = sf.read(spk1, dtype='float32')
        spk2_np, sr2 = sf.read(spk2, dtype='float32')
        mix_np , sr_mix = sf.read(mix, dtype='float32')

        
                # 简单归一化，避免裁剪
        max_abs = np.max(np.abs(mix_np))
        if max_abs > 0:
            mix_np = mix_np / max_abs * 0.99
        
        # print(f"Loaded audio shapes: spk1 {spk1_np.shape}, spk2 {spk2_np.shape}, mix {mix_np.shape}")

        # ---------- 2. 转成 torch.Tensor (C, T) ----------
        spk1_waveform = torch.from_numpy(spk1_np.T).unsqueeze(0) if spk1_np.ndim == 1 else torch.from_numpy(spk1_np.T)
        spk2_waveform = torch.from_numpy(spk2_np.T).unsqueeze(0) if spk2_np.ndim == 1 else torch.from_numpy(spk2_np.T)
        mix_waveform  = torch.from_numpy(mix_np.T).unsqueeze(0)  if mix_np.ndim  == 1 else torch.from_numpy(mix_np.T)

        # print(f"Converted to tensors: spk1 {spk1_waveform.shape}, spk2 {spk2_waveform.shape}, mix {mix_waveform.shape}")

        # ---------- 3. 重采样 ----------
        resampler = torchaudio.transforms.Resample

        if sr1 != self.sample_rate:
            spk1_waveform = resampler(sr1, self.sample_rate)(spk1_waveform)
        if sr2 != self.sample_rate:
            spk2_waveform = resampler(sr2, self.sample_rate)(spk2_waveform)
        if sr_mix != self.sample_rate:
            mix_waveform = resampler(sr_mix, self.sample_rate)(mix_waveform)

        # ---------- 4. 截取最小长度 ----------
        min_length = min(spk1_waveform.shape[1], spk2_waveform.shape[1], mix_waveform.shape[1])
        spk1_waveform = spk1_waveform[:, :min_length]
        spk2_waveform = spk2_waveform[:, :min_length]
        mix_waveform  = mix_waveform[:, :min_length]

        return spk1_waveform, spk2_waveform, mix_waveform, sr1

    def load_info(self, info_path):
        """加载 info.csv"""
        info = {
            "speaker_1": {},
            "speaker_2": {},
        }

        with open(info_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                speaker = row["Speaker"].strip()
                if speaker == "Speaker 1":
                    info["speaker_1"] = {
                        "id": row["ID"],
                        "utterance": row["Utterance"],
                        "gender": row["Gender"],
                        "text": row["Text"],
                        "emotion": row["Emotion"],
                        "language": row["Language"],
                        "pre": row["Pre"],
                        "sector": row["Sector"],
                    }
                elif speaker == "Speaker 2":
                    info["speaker_2"] = {
                        "id": row["ID"],
                        "utterance": row["Utterance"],
                        "gender": row["Gender"],
                        "text": row["Text"],
                        "emotion": row["Emotion"],
                        "language": row["Language"],
                        "pre": row["Pre"],
                        "sector": row["Sector"],
                    }
                elif speaker == "Length (seconds)":
                    info["length_sec"] = float(row["ID"])
                elif speaker == "Relative dB (Speaker 2)":
                    info["relative_db_spk2"] = float(row["ID"])
        return info
    
    # 视频掩码
    def apply_video_mask(self, feature, mask_ratio=0.4):
        """
        feature: [T, D]
        """
        if feature is None: 
            return None 
        
        # 复制一份，不修改原数据
        masked_feat = feature.clone()
        T, D = masked_feat.shape
            
        # 模拟部分遮挡/丢帧 (Augmentation)
        mask_len = int(T * mask_ratio)
        
        if mask_len > 0:
            start = random.randint(0, T - mask_len)
            end = start + mask_len
            masked_feat[start:end, :] = 0.0 
            # print(f"Applied video mask from frame {start} to {end}")
            # print("masked feature segment:", masked_feat[start:end, :])
            # print("feature segment before masking:", feature[start:end, :])
            # print("no mask part:", masked_feat[:start, :], masked_feat[end:, :])
            # print("feature segment before masking:", feature[:start, :], feature[end:, :])
        
        # print("feature shape:", feature.shape)
        # print("masked feature shape:", masked_feat.shape)
        
        return masked_feat
    
    def safe_cut_padding(self, feature, target_len):
        """
        强制对齐视频特征长度：
        - 如果 feature 长于 target_len -> 截取
        - 如果 feature 短于 target_len -> 补零
        """
        if feature is None:
            return None
            
        current_len = feature.shape[0]
        
        # 情况1: 视频特征足够长 -> 直接截取
        if current_len >= target_len:
            return feature[:target_len, :]
        
        # 情况2: 视频特征不够长 -> 补零填充
        else:
            diff = target_len - current_len
            # 创建全0的补充块，形状为 [差值, 维度]
            padding = np.zeros((diff, feature.shape[1]), dtype=feature.dtype)
            # 拼接
            padded_feature = np.concatenate([feature, padding], axis=0)
            return padded_feature

    def add_noise_to_pre_by_snr(self, clean_wav: torch.Tensor, snr_db: float = 0.0, clamp_audio: bool = True) -> torch.Tensor:
        """
        clean_wav: [L] or [C, L]
        snr_db: 目标 SNR(dB)
        """
        squeeze_back = False
        if clean_wav.dim() == 1:
            clean_wav = clean_wav.unsqueeze(0)
            squeeze_back = True
        elif clean_wav.dim() != 2:
            raise ValueError(f"Expected pre waveform dim 1 or 2, got {clean_wav.dim()}")

        clean_wav = clean_wav.float()
        eps = 1e-8

        signal_power = clean_wav.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, eps)

        noise = torch.randn_like(clean_wav)
        noise = noise / noise.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()
        noise = noise * noise_power.sqrt()

        noisy = clean_wav + noise
        if clamp_audio:
            noisy = torch.clamp(noisy, min=-1.0, max=1.0)

        if squeeze_back:
            noisy = noisy.squeeze(0)
        return noisy

    def __getitem__(self, idx):

        #---------random choice----------
        combo = self.cue_combinations[np.random.randint(1, 16)]

        sample_dir = self.sample_dirs[idx]
        sample_id = sample_dir.name

        # print("spmle_dir:", sample_dir)

        # 路径
        mixture_path_4 = sample_dir / 'mixture_4ch.wav'
        mixture_path_1 = sample_dir / 'mixture.wav'
        spk1_path_4 = sample_dir / 'spk1_reverb_4ch.wav'
        spk2_path_4 = sample_dir / 'spk2_reverb_4ch.wav'
        spk1_path_1 = sample_dir / 'spk1_clean.wav'
        spk2_path_1 = sample_dir / 'spk2_clean.wav'

        info_path = sample_dir / 'info.csv'

        # 加载音频
        spk1_4, spk2_4, mixture_4, sr = self.load_audio(spk1_path_4, spk2_path_4, mixture_path_4)
        spk1_1, spk2_1, mixture_1, sr = self.load_audio(spk1_path_1, spk2_path_1, mixture_path_1)

        # print(spk1_1.shape, spk2_1.shape, mixture_1.shape)
        # print(spk1_4.shape, spk2_4.shape, mixture_4.shape)

        # 构建返回
        # print(mixture.shape, spk1.shape, spk2.shape)
        if not combo["use_main"]:
            mixture = mixture_1
            mixtures = torch.cat((mixture, mixture), dim=0)  # [2, L]
            spk1 = spk1_1
            spk2 = spk2_1
            src = torch.cat((spk1, spk2), dim=0)         # [2, L]
        else:
            mixture = mixture_4.permute(1, 0).unsqueeze(0)
            mixtures = torch.cat((mixture, mixture), dim=0).squeeze(0)  # [2, L]
            spk1 = spk1_4.permute(1, 0).unsqueeze(0)
            spk2 = spk2_4.permute(1, 0).unsqueeze(0)
            src = torch.cat((spk1, spk2), dim=0).squeeze(0)              # [2, L]
        # print(f"Sample {sample_id}: mixtures shape {mixtures.shape}, src shape {src.shape}")
        # mixture = mixture_4.permute(1, 0).unsqueeze(0)
        # mixtures = torch.cat((mixture, mixture), dim=0).squeeze(0)  # [2, L]
        # spk1 = spk1_4.permute(1, 0).unsqueeze(0)
        # spk2 = spk2_4.permute(1, 0).unsqueeze(0)
        # src = torch.cat((spk1, spk2), dim=0).squeeze(0)              # [2, L]        


        # 加载 info
        info = self.load_info(info_path)

        spk1_path = f"{info['speaker_1']['id']}_{info['speaker_1']['gender']}_{info['speaker_1']['language']}"
        spk2_path = f"{info['speaker_2']['id']}_{info['speaker_2']['gender']}_{info['speaker_2']['language']}"

        spk1_rel = f"{spk1_path}/{info['speaker_1']['utterance'].replace('.wav', '.npz')}"
        spk2_rel = f"{spk2_path}/{info['speaker_2']['utterance'].replace('.wav', '.npz')}"

        # print(f"spk1_rel: {spk1_rel}")
        # print(f"spk2_rel: {spk2_rel}")

        # print("mixtures shape:", mixtures.shape)
        min_frame = math.floor((mixtures.shape[1] / sr) * 25)
        # print("sr:", sr)

        spk1_av_path = self.av_embedding_dir / spk1_rel
        spk2_av_path = self.av_embedding_dir / spk2_rel

        # print(f"spk1_av_path: {spk1_av_path}")
        # print(f"spk2_av_path: {spk2_av_path}")

        audio_samples = mixtures.shape[1]  # 样本点数

        target_video_frames = int(audio_samples / self.sample_rate * 25)

        # 性能优化：每个文件只读一次
        # 使用 allow_pickle=True 避免兼容性问题
        try:
            spk1_data = np.load(spk1_av_path, allow_pickle=True)
            spk2_data = np.load(spk2_av_path, allow_pickle=True)
            # print('spk1_av_path:', spk1_av_path)
            # print('spk2_av_path:', spk2_av_path)

            # 读取原始特征 (此时已经是 25 FPS 了，但长度可能很长)
            raw_lip1 = spk1_data["lip"]
            raw_body1 = spk1_data["body"]
            
            raw_lip2 = spk2_data["lip"]
            raw_body2 = spk2_data["body"]

            # 使用 helper 函数强制对齐长度
            # 注意：先处理 numpy 再转 tensor，效率更高
            spk1_face_np = self.safe_cut_padding(raw_lip1, target_video_frames)
            spk1_body_np = self.safe_cut_padding(raw_body1, target_video_frames)
            
            spk2_face_np = self.safe_cut_padding(raw_lip2, target_video_frames)
            spk2_body_np = self.safe_cut_padding(raw_body2, target_video_frames)

            # 转为 Tensor
            spk1_face = torch.from_numpy(spk1_face_np).float()
            spk1_body = torch.from_numpy(spk1_body_np).float()
            spk2_face = torch.from_numpy(spk2_face_np).float()
            spk2_body = torch.from_numpy(spk2_body_np).float()
            
        except Exception as e:
            # # 万一文件损坏，这里可以做一个兜底，或者直接报错
            # print(f"Error loading npz for {sample_dir}: {e}")
            # raise e

            # 损坏文件则随机返回另一个样本
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)
        
        spk1_face = spk1_face[:min_frame, :]
        spk1_body = spk1_body[:min_frame, :]
        spk2_face = spk2_face[:min_frame, :]
        spk2_body = spk2_body[:min_frame, :]

        #---------pre----------
        pre1_path = self.audio_dir / spk1_path / info["speaker_1"]["pre"]
        pre2_path = self.audio_dir / spk2_path / info["speaker_2"]["pre"]

        pre1, pre2, mixture, sr = self.load_audio(pre1_path, pre2_path, mixture_path_1)
        # Bandwidth loss:
        # sr_down = 8000 #8k, 4k
        # pre1 = self.down_up_resample(pre1, sr_in=sr, sr_down=sr_down)
        # pre2 = self.down_up_resample(pre2, sr_in=sr, sr_down=sr_down)

        # short enrollment:
        # second = 2.0 # 2.0, 1.0, 0.5
        # pre1 = pre1[:, :int(sr * second)]
        # pre2 = pre2[:, :int(sr * second)]

        pre = torch.cat((pre1, pre2), dim=0).squeeze(0)
        # low SNR pre-enrollment：
        # snr_db = 5 # 5 dB, 0 dB, -5 dB
        # pre = self.add_noise_to_pre_by_snr(pre, snr_db=snr_db)     

        
        #--------text-----------
        min_word_len = min(len(info['speaker_1'].get('text', '').split()),
                           len(info['speaker_2'].get('text', '').split()))

        def truncate_text(text):
            words = text.split()
            return ' '.join(words[:min_word_len]) if len(words) > min_word_len else text

        target_entry = {
            "aligned_transcription": truncate_text(info['speaker_1'].get('text', '')),
            "emotion": info['speaker_1'].get('emotion', ''),
            "gender": info['speaker_1'].get('gender', ''),
            "language": info['speaker_1'].get('language', ''),
        }
        interference_entry = {
            "aligned_transcription": truncate_text(info['speaker_2'].get('text', '')),
            "emotion": info['speaker_2'].get('emotion', ''),
            "gender": info['speaker_2'].get('gender', ''),
            "language": info['speaker_2'].get('language', ''),
        }

        # print(info['speaker_1']['gender'], info['speaker_2']['gender'])

        desc_tar, desc_int = build_short_style_descriptions_for_two_speakers(target_entry, interference_entry)
        template = ShortTemplate(acts=['1']) 
        text1 = template.templates[1][0].format(desc_tar)
        text2 = template.templates[1][0].format(desc_int)

        #---------sector----------
        sec1 = torch.tensor(int(info['speaker_1']['sector']), dtype=torch.long)
        sec2 = torch.tensor(int(info['speaker_2']['sector']), dtype=torch.long)
        main = torch.tensor([sec1, sec2], dtype=torch.long)

        # print("Combination:", combo)

        use_mask = False
        if not combo["use_av"]:
            spk1_face = None
            spk2_face = None
            spk1_body = None
            spk2_body = None
        elif use_mask:
            mask_ratio = 0.4
            T, D = spk1_face.shape
            mask_len = int(mask_ratio * T)
            start = random.randint(0, T - mask_len)
            # print("start:", start, "mask_len:", mask_len)
            spk1_face = self.apply_video_mask(spk1_face, start, mask_len)
            spk2_face = self.apply_video_mask(spk2_face, start, mask_len)
            spk1_body = self.apply_video_mask(spk1_body, start, mask_len)
            spk2_body = self.apply_video_mask(spk2_body, start, mask_len)

        if not combo["use_text"]:
            text1 = None
            text2 = None
            text = None
        else :
            text = [text1, text2]

        if not combo["use_main"]:
            main = None

        if not combo["use_pre"]:
            pre = None
        
        return mixtures, src, spk1_face, spk2_face, spk1_body, spk2_body, pre, text, main


def dummy_collate_fn(x):
    return x[0] if len(x) == 1 else x

def isNone(obj):
    return obj is None

if __name__ == "__main__":
    dataset = MixtureDataset(root_dir='/path/to/your/yutto/filtered_output_all', split='train')  # TODO: set your own data root
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dummy_collate_fn)

    for i, data in enumerate(dataloader):
        # if i == 6400:
        mixtures, src, spk1_face, spk2_face, spk1_body, spk2_body, pre, text, main = data
        print(src.shape)
        print(text)
        print(f"Sample {i}:")
        # print("Mixtures shape:", mixtures.shape)
        # print("Src shape:", src.shape)
        # print("face shapes:", spk1_face.shape if spk1_face is not None else None, spk2_face.shape if spk2_face is not None else None)
        # print("body shapes:", spk1_body.shape if spk1_body is not None else None, spk2_body.shape if spk2_body is not None else None)
        # # print("Pre shape:", pre.shape if pre is not None else None)
        print("Text:", text)
        print("Main:", main)
        # print("video is None:", isNone(spk1_face), isNone(spk2_face), isNone(spk1_body), isNone(spk2_body))
        # print("pre is None:", isNone(pre))
        # if not isNone(text):
        #     print(text)
        # if not isNone(pre):
        #     print(pre.shape)
        # if not isNone(spk1_body):
        #     print("spk1_body shape:", spk1_body.shape)
        # # print("text is None:", isNone(text[0]), isNone(text[1]))
        # print("main is None:", isNone(main))
        if i >= 17:  # 只查看前10个样本
            break

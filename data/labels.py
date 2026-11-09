#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像
import re
import json
import torch
from tqdm import tqdm
from pathlib import Path
from speechbrain.inference.interfaces import foreign_class # 修正引用以适配新版SpeechBrain，如旧版可用原引用
from faster_whisper import WhisperModel

# ------------------- 删除旧缓存（可选） -------------------
import shutil
if os.path.exists("pretrained_emotion_model"):
    shutil.rmtree("pretrained_emotion_model")

# ------------------- 配置 -------------------
# [修改] 直接指向 audio_clean 目录
BASE_DIR = Path("/path/to/your/yutto/filtered_output_all/audio_clean")
JSON_NAME = "metadata.json"

# 全局 JSON 文件路径
TOTAL_JSON_PATH = BASE_DIR / "total_metadata.json"

# ------------------- 情感识别模型 -------------------
# 注意：如果 speechbrain 版本更新，foreign_class 引用可能变化，这里保持你原本的逻辑
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

# ------------------- ASR 模型 -------------------
whisper_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

# ------------------- 工具函数 -------------------
def extract_info_from_path(wav_path: Path):
    """
    从父目录名提取 info，例如目录名为: BV1xx_male_chinese
    """
    parent_dir = wav_path.parent.name
    match = re.match(r"(BV[0-9A-Za-z]+)_(\w+)_(\w+)", parent_dir)
    if not match:
        raise ValueError(f"目录名格式错误 (期望 BVxxx_gender_lang): {parent_dir}")
    video_id, gender, language = match.groups()
    return video_id, gender, language


def get_emotion(audio_path: Path) -> str:
    try:
        out_prob, score, index, text_lab = classifier.classify_file(str(audio_path))
        label = text_lab[0]
        mapping = {'hap': 'happy', 'sad': 'sad', 'ang': 'angry', 'neu': 'neutral'}
        return mapping.get(label, label)
    except Exception as e:
        print(f"情感识别失败 {audio_path.name}: {e}")
        return "unknown"


def get_transcript(audio_path: Path) -> str:
    try:
        segments, _ = whisper_model.transcribe(str(audio_path), beam_size=5)
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"ASR 失败 {audio_path.name}: {e}")
        return ""


def update_local_json(json_path: Path, data: dict):
    """
    更新单目录的 metadata.json
    读取现有列表并追加，防止覆盖同目录下的其他文件记录
    """
    all_data = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    all_data = content
        except Exception:
            all_data = []

    # 检查是否已存在同名文件记录，避免重复追加
    # (简单去重：如果 filename 已存在，则先移除旧的)
    all_data = [d for d in all_data if d.get('filename') != data['filename']]
    
    all_data.append(data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


def save_total_json(total_data: list):
    """统一写盘全局 JSON"""
    TOTAL_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOTAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(total_data, f, ensure_ascii=False, indent=2)


# ------------------- 主流程 -------------------
def process_all_files(root_dir: Path, total_data: list):
    # rglob 会递归查找 root_dir 下所有子目录里的 .wav
    wav_files = list(root_dir.rglob("*.wav"))
    print(f"在 {root_dir} 下共发现 {len(wav_files)} 个 wav 文件")

    for wav_path in tqdm(wav_files, desc="处理音频文件"):
        try:
            # 1. 提取元数据
            video_id, gender, language = extract_info_from_path(wav_path)
            
            # 2. 推理
            emotion = get_emotion(wav_path)
            text = get_transcript(wav_path)

            item = {
                "filename": wav_path.name,
                "video_id": video_id,
                "gender": gender,
                "language": language,
                "emotion": emotion,
                "text": text,
                # relative_path 现在直接是 "子目录名/文件名.wav"
                "relative_path": str(wav_path.relative_to(BASE_DIR))
            }

            # 3. 写入该 wav 所在子目录的 metadata.json
            local_json_path = wav_path.parent / JSON_NAME
            update_local_json(local_json_path, item)

            # 4. 添加到全局列表
            total_data.append(item)

        except ValueError as ve:
            # 忽略不符合命名规范的文件夹（比如 audio_clean 下的杂文件）
            pass 
        except Exception as e:
            print(f"\n处理 {wav_path.name} 时出错: {e}")
            continue


def main():
    print(f"开始处理目录: {BASE_DIR}")
    
    # [修改] 不再加载旧的 total_metadata.json，而是重新生成
    # 因为路径结构变了，旧的相对路径可能无效
    total_data = []

    if not BASE_DIR.exists():
        print(f"错误：目录不存在 -> {BASE_DIR}")
        return

    # 直接处理根目录，rglob 会搞定所有子文件夹
    process_all_files(BASE_DIR, total_data)

    # 统一保存全局 JSON
    save_total_json(total_data)
    print(f"全部完成！全局信息已保存至：{TOTAL_JSON_PATH}")


if __name__ == "__main__":
    main()
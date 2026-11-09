#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量视频下载器（Python 版）
基于 yutto，支持高清 + AI 翻译字幕 + ASS 弹幕
下载完成后把文件重命名为 <video_id>_gender_lang.mp4
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List
import time

# ============ 配置区 ============
# TODO: replace /path/to/your below with your own directories

URLS_FILE = "/path/to/your/yutto/video_all.txt"                    # URL 列表文件
QUALITY = "112"                            # 1080P+，大会员用 "120"=4K
DANMAKU_FORMAT = "ass"                     # 字幕格式：ass 或 srt
AI_LANG = "zh-cn"                          # AI 翻译语言：zh-cn, en, ja 等
NUM_WORKERS = 8                            # 并发下载数（建议 4~16）
DOWNLOAD_DIR = "/path/to/your/yutto/videos"         # 下载目录（自动创建）

# 常见视频扩展名（用于重命名时匹配）
VIDEO_EXTS = ['.mp4', '.flv', '.mkv', '.webm', '.ts', '.avi']

# ==============================

def read_urls(file_path: str) -> List[str]:
    """读取 URL 文件，跳过空行和 # 注释"""
    if not os.path.isfile(file_path):
        print(f"错误：找不到 {file_path} 文件！")
        print(f"请在当前目录创建 {file_path}，每行写入一个视频 URL 和可选的 gender/lang（用空格分隔）")
        sys.exit(1)

    urls = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.replace('\r', '')
            urls.append(line)
    
    if not urls:
        print(f"错误：{file_path} 中没有有效的 URL！")
        sys.exit(1)
    
    return urls


def extract_bv(url: str) -> str:
    """从 URL 中提取 视频 ID"""
    import re
    match = re.search(r'BV[0-9a-zA-Z]+', url)
    return match.group(0) if match else "Unknown"


def safe_unique_name(target_path: Path) -> Path:
    """
    若 target_path 已存在，则在 视频 ID后添加 P1, P2, P3... 生成唯一路径。
    例如: BV1eM411S7Mi_male_Spanish.mp4 → BV1eM411S7MiP3_male_Spanish.mp4
    """
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix

    # 提取 视频 ID（假设以 BV 开头）
    import re
    bv_match = re.match(r'(BV[0-9a-zA-Z]+)', stem)
    if not bv_match:
        # 回退到旧逻辑
        i = 1
        while True:
            candidate = target_path.with_name(f"{stem}_{i}{suffix}")
            if not candidate.exists():
                return candidate
            i += 1

    bv = bv_match.group(0)
    rest = stem[len(bv):]  # 去掉 视频 ID后的部分（如 _male_Spanish）

    i = 1
    while True:
        new_stem = f"{bv}P{i}{rest}"
        candidate = target_path.with_name(f"{new_stem}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def rename_file(target: str, new_name: str, path: str = ".") -> bool:
    """
    在指定目录下查找第一个名称中包含 target 的文件（优先常见视频扩展名），
    并重命名为 new_name（new_name 可包含扩展名）。
    返回 True 表示已重命名，False 表示未找到或出错。
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"重命名失败：目录不存在 {path}")
        return False

    # 遍历目录，优先匹配包含 target 且扩展符合 VIDEO_EXTS 的文件
    candidates = []
    for fname in os.listdir(path):
        full = os.path.join(path, fname)
        if not os.path.isfile(full):
            continue
        if target in fname:
            _, ext = os.path.splitext(fname)
            candidates.append((fname, ext.lower()))

    # 先找视频扩展名文件，如果没有则任意匹配第一个包含 target 的文件
    chosen = None
    for fname, ext in candidates:
        if ext in VIDEO_EXTS:
            chosen = fname
            break
    if not chosen and candidates:
        chosen = candidates[0][0]

    if not chosen:
        # 未找到包含 target 的文件
        # print(f"❌ 未在 {path} 中找到包含 '{target}' 的文件。")
        return False

    old_path = Path(path) / chosen

    # 确保 new_name 有扩展名；若没有，使用原文件扩展名
    new_path = Path(path) / new_name
    if new_path.suffix == "":
        new_path = new_path.with_suffix(old_path.suffix)

    # 防止覆盖：生成唯一文件名
    new_path = safe_unique_name(new_path)

    try:
        old_path.rename(new_path)
        print(f"✅ 已重命名: {old_path.name} → {new_path.name}")
        return True
    except Exception as e:
        print(f"重命名失败: {e}")
        return False


def list_video_files(root_path: Path):
    """
    返回 root_path 下所有文件（包括子目录）中，带视频扩展名的文件集合（Path 对象）。
    """
    files = set()
    for p in root_path.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTS:
                files.add(p.resolve())
    return files


def run_yutto(line: str, download_path: Path):
    """
    执行 yutto 下载命令。line 是 URL 文件的一行，例如：
    <video_url> male English

    改进点：在下载前后对比目录，找出新文件并重命名。
    """
    parts = line.split()
    if not parts:
        return
    url = parts[0]
    bv = extract_bv(url)
    gender = parts[1] if len(parts) > 1 else "unknown"
    lang = parts[2] if len(parts) > 2 else "unknown"
    target_name = f"{bv}_{gender}_{lang}.mp4"  # 目标名

    print(f"正在下载: {bv}")
    print(f"URL: {url}")

    # 记录下载前目录中已有的视频文件
    before_files = list_video_files(download_path)

    cmd = [
        "yutto", url,
        "-q", QUALITY,
        "-df", DANMAKU_FORMAT,
        "--ai-translation-language", AI_LANG,
        "--no-danmaku",
        "-n", str(NUM_WORKERS),
        "-d", str(download_path),
        "--tmp-dir", str(download_path / "tmp"),
        "-w",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"完成下载: {bv}")
    except subprocess.CalledProcessError as e:
        print(f"失败: {bv}（可能需登录或网络问题）")
        if e.stderr:
            print(f"错误输出: {e.stderr.strip()}")
        else:
            print(e)
        return
    except FileNotFoundError:
        print("错误：未找到 'yutto' 命令！请确保已安装：pip install yutto")
        sys.exit(1)

    # 小睡一会儿，确保文件系统刷新（有时候子进程快速退出，文件写入稍有延迟）
    time.sleep(0.2)

    # 下载后重新扫描目录
    after_files = list_video_files(download_path)

    # 找到新增的文件集合（可能为空）
    new_files = after_files - before_files

    if new_files:
        # 从新增文件里选最近修改的那个（更有可能是我们刚下的）
        newest = max(new_files, key=lambda p: p.stat().st_mtime)
        # 构建目标路径（保留扩展名）
        new_target = Path(download_path) / target_name
        if new_target.suffix == "":
            new_target = new_target.with_suffix(newest.suffix)
        new_target = safe_unique_name(new_target)
        try:
            newest.rename(new_target)
            print(f"✅ 已重命名（基于新增文件）: {newest.name} → {new_target.name}")
            return
        except Exception as e:
            print(f"重命名新增文件失败: {e}")
            # 继续尝试按视频 ID 匹配（兜底）
    else:
        # 没检测到新增文件 —— 也许 yutto 把文件放在子目录以外的扩展格式，或下载速度慢
        print("⚠️ 未检测到新增的视频文件（基于前后对比）。尝试按文件名包含视频 ID 来匹配（回退策略）...")

    # 回退策略：按文件名包含视频 ID 来匹配（保持兼容旧逻辑）
    renamed = rename_file(bv, target_name, str(download_path))
    if not renamed:
        print(f"⚠️ 仍未能找到包含 {bv} 的文件以重命名。")
        print("可选方案：1) 检查下载目录确认 yutto 的实际输出文件名/位置；2) 使用 --filename-template 让 yutto 直接生成你想要的名字。")

def main():
    # 读取 URL
    urls = read_urls(URLS_FILE)
    print(f"开始批量下载 {len(urls)} 个视频...")

    # 创建下载目录
    download_path = Path(DOWNLOAD_DIR).resolve()
    download_path.mkdir(parents=True, exist_ok=True)
    print(f"下载目录: {download_path}")

    print(f"参数: {QUALITY}p + AI 翻译({AI_LANG}) + {DANMAKU_FORMAT} 字幕")
    print("-" * 50)

    # 遍历下载
    for line in urls:
        run_yutto(line, download_path)
        print("-" * 50)

    print("所有视频处理完成！")
    print(f"文件保存在: {download_path}")


if __name__ == "__main__":
    main()
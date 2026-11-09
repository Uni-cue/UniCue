#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from pathlib import Path

# ================== 配置区 ==================
# TODO: replace /path/to/your below with your own directories
INPUT_DIR      = Path("/path/to/your/yutto/videos")
OUTPUT_BASE    = Path("/path/to/your/yutto/split_output_all")
SEGMENT_TIME   = 6               # 每段时长（秒）
FFMPEG_LOGLEVEL = "error"        # ffmpeg 输出级别，可改成 "info" 看详细日志
# ============================================

def ensure_ffmpeg():
    """检查 ffmpeg 是否可用"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误：未找到 ffmpeg，请先安装 ffmpeg。")
        sys.exit(1)

def split_video(src: Path, dst_dir: Path):
    """使用 ffmpeg 把 src 切成若干段，输出到 dst_dir"""
    # -c copy   : 直接复制流，不重新编码，速度极快且无损
    # -f segment: 按时间切分
    # -segment_time: 每段时长（秒）
    # -reset_timestamps 1: 每段从 0 开始计时
    cmd = [
        "ffmpeg",
        "-i", str(src),
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(SEGMENT_TIME),
        "-reset_timestamps", "1",
        "-loglevel", FFMPEG_LOGLEVEL,
        str(dst_dir / "segment_%04d.mp4")
    ]

    print(f"正在处理: {src.name}  →  {dst_dir}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"ffmpeg 出错（{src.name}）:\n{result.stderr.decode()}")
    else:
        print(f"完成: {src.name}")

def main():
    ensure_ffmpeg()

    # 确保输出根目录存在
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # 遍历所有 .mp4 文件
    for mp4_path in INPUT_DIR.glob("*.mp4"):
        if not mp4_path.is_file():
            continue

        # 子目录名 = 文件名去掉 .mp4
        sub_dir_name = mp4_path.stem
        out_dir = OUTPUT_BASE / sub_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        split_video(mp4_path, out_dir)

    print("\n全部完成！输出目录结构：")
    subprocess.run(["tree", "-L", "2", str(OUTPUT_BASE)])

if __name__ == "__main__":
    main()
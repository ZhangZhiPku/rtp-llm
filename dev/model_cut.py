# simplified_test_rtp_client.py
import glob
import os

from safetensors.torch import load_file, save_file
from tqdm import tqdm

# PATH = "/root/hf/Qwen3-30B-A3B"
PATH = "/root/hf/Qwen3-30B-A3B"
STORE = "/root/hf/Qwen3-30B-A3B-Debug"

import glob
import os
import shutil

from safetensors.torch import load_file, save_file
from tqdm import tqdm


def load_and_cut(load_path: str, store_path: str):
    # 从 load path 中加载数据，从中只抽取第一层的权重，之后存入 store path
    # 如果 store path 不存在，创建它

    files = sorted(glob.glob(os.path.join(load_path, "model*.safetensors")))
    if not files:
        files = sorted(glob.glob(os.path.join(load_path, "*.safetensors")))

    processing_weights = {}
    for fn in tqdm(files, desc="loading weights"):
        part = load_file(fn, device="cpu")
        processing_weights.update(part)

    # 过滤掉除了全局权重和第一层权重以外的内容
    remaining_part = {}
    for name, value in processing_weights.items():
        if "layers.0" in name or "layers" not in name:
            remaining_part[name] = value

    # 创建目标目录（如果不存在）
    os.makedirs(store_path, exist_ok=True)

    # 以 safetensors 格式保存过滤后的权重
    output_path = os.path.join(store_path, "model.safetensors")
    save_file(remaining_part, output_path)
    print(f"✅ 已保存过滤后的权重到 {output_path}")
    print(f"包含 {len(remaining_part)} 个权重，原始权重共 {len(processing_weights)} 个")
    for remaining in remaining_part:
        print(remaining)

    # 拷贝非 safetensors 文件到 store_path
    """
    print(f"📂 正在拷贝非 safetensors 文件...")
    copied_count = 0
    for item in os.listdir(load_path):
        item_path = os.path.join(load_path, item)
        # 检查是否是文件且不是 safetensors 文件
        if os.path.isfile(item_path) and not item.endswith('.safetensors'):
            dest_path = os.path.join(store_path, item)
            shutil.copy2(item_path, dest_path)  # copy2 保留元数据
            copied_count += 1
            print(f"   已拷贝: {item}")

    print(f"✅ 共拷贝了 {copied_count} 个非 safetensors 文件")
    """


load_and_cut(load_path=PATH, store_path=STORE)

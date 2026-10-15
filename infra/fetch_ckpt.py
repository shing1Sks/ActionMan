"""On the box: download trained checkpoints from the private HF repo.
python infra/fetch_ckpt.py runs/s2-v1b/v1:/workspace/ckpt/v1 runs/s3-qwen/qwen:/workspace/ckpt/qwen"""
import os
import sys

from huggingface_hub import snapshot_download

for spec in sys.argv[1:]:
    src, dst = spec.split(":", 1)
    path = snapshot_download(os.environ["HF_REPO"], token=os.environ["HF_TOKEN"], allow_patterns=[f"{src}/*"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.join(path, src), dst)
    print("fetched", src, "->", dst, os.listdir(dst), flush=True)

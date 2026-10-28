"""云端解压后逐文件检查 SHA256，确认代码/数据传输完整，不联网。"""
from pathlib import Path
import hashlib
import json


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'upload_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'文件缺失/越界：{relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'上传校验失败：{relative}')
    print(f'上传验证通过：{len(manifest["files"])} 个文件，其中 CIFAR-10 PNG {manifest["png_files"]} 张。')

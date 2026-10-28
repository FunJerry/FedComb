"""只打包源码与 CIFAR-10，排除密钥、Git、IDE、虚拟环境和旧结果。

用自己的 allowlist，不用 git ls-files，避免遗漏仍未提交的数据加载模块。
"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'fedcomb_cifar10'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='deploy/fedcomb_cifar10_bundle.tar.gz')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'deploy'):
        raise ValueError('打包输出必须位于本工程 deploy 内')
    if output.exists():
        raise FileExistsError('包已存在，请使用新的 --output')
    files = []
    for directory in ['src', 'configs', 'experiments', 'scripts', 'tests', 'docs']:
        for path in sorted((ROOT / directory).rglob('*')):
            if (path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts
                and path.suffix != '.pyc' and not any(p.endswith('.egg-info') for p in path.parts)):
                files.append(path)
    files += [ROOT / name for name in ['pyproject.toml', 'README.md', '.gitignore']]
    data = sorted((ROOT / 'data/cifar10').rglob('*.png'))
    if len(data) != 60000:
        raise ValueError('没有完整的 CIFAR-10 60000 张 PNG，不生成部署包')
    files += data
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'source': 'local Fedcomb project; CIFAR-10 PNG imported from RingSFL',
                'png_files': len(data), 'files': {}}
    for path in files:
        if path.is_symlink():
            raise ValueError('部署载荷不能是符号链接')
        relative = str(path.relative_to(ROOT))
        manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = output.parent / (output.stem + '_manifest.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    with tarfile.open(output, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=f'{PREFIX}/{path.relative_to(ROOT)}', recursive=False)
        archive.add(manifest_path, arcname=f'{PREFIX}/upload_manifest.json', recursive=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f'部署包：{output}\n大小：{output.stat().st_size/1e6:.1f} MB\nSHA256：{digest}')


if __name__ == '__main__':
    main()

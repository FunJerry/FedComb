"""本地 PNG 数据兼容测试，不联网、不依赖实际 60000 张图。"""
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from fedcomb.config import Config
from fedcomb.data.datasets import load_datasets


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


class LocalDataTests(unittest.TestCase):
    def test_png_loading_and_label_mapping_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'test']:
                for name in CLASSES:
                    folder = root / 'data/cifar10' / split / name
                    folder.mkdir(parents=True)
                    Image.new('RGB', (32, 32)).save(folder / 'sample.png')
            cfg = Config()
            train, test, labels = load_datasets(cfg.data, cfg.model, 100, 42, root)
            self.assertEqual(len(train), 10)
            self.assertEqual(len(test), 10)
            self.assertEqual(labels.tolist(), list(range(10)))
            self.assertEqual(train.class_to_idx, test.class_to_idx)
            self.assertEqual(train[0][0].shape, (3, 32, 32))

    def test_wrong_classes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'test']:
                folder = root / 'data/cifar10' / split / 'wrong_class'
                folder.mkdir(parents=True)
                Image.new('RGB', (32, 32)).save(folder / 'sample.png')
            cfg = Config()
            with self.assertRaises(ValueError):
                load_datasets(cfg.data, cfg.model, 100, 42, root)


if __name__ == '__main__':
    unittest.main(verbosity=2)

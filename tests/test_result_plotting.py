"""绘图数据口径测试；只使用临时的两轮指标，不运行模型训练。"""
from pathlib import Path
import csv
import importlib.util
import json
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/show_results.py'
spec = importlib.util.spec_from_file_location('result_plotting', SCRIPT)
plotting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotting)


class ResultPlottingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / 'fedcomb/seed-42'
        self.folder.mkdir(parents=True)
        self.config = {'experiment': {'name': 'fedcomb', 'seed': 42},
                       'federation': {'participants': 2, 'rounds': 2}}
        (self.folder / 'config.json').write_text(json.dumps(self.config))
        self.rows = [dict(zip(('round', *plotting.FIELDS), values)) for values in
                     [(0, .2, 2., 2, 0, 36., 2**30, 18.),
                      (1, .6, .8, 1, 1, 72., 2 * 2**30, 36.)]]
        self.write_rows()

    def write_rows(self):
        with (self.folder / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self):
        summary = {'rounds': 2, 'last_accuracy': .6, 'best_accuracy': .6,
                   'wall_seconds': 54., 'simulated_seconds': 108.,
                   'communication_bytes': 3 * 2**30}
        (self.folder / 'summary.json').write_text(json.dumps(summary))

    def test_forward_only_smoothing_and_units(self):
        self.assertEqual(plotting.trailing_mean([1, 3, 8], 2), [1, 2, 5.5])
        self.assertEqual(plotting.trailing_mean([1, 3], 1), [1, 3])
        with self.assertRaises(ValueError):
            plotting.trailing_mean([1], 0)
        self.finish()
        record = plotting.collect(self.root)[0]
        series = plotting.chart_series(record)
        self.assertEqual(series[0], ([1, 2], [20., 60.]))
        self.assertEqual(series[2][0], [.005, .015])
        self.assertEqual(series[4][0], [1., 3.])
        summary = plotting.summary_rows([record])[0]
        self.assertAlmostEqual(summary['last50_accuracy_pct'], 40.)
        self.assertEqual(summary['best_round'], 2)
        self.assertEqual(summary['acceptance_pct'], 75.)

    def test_incomplete_is_not_completed(self):
        self.assertFalse(plotting.collect(self.root)[0]['completed'])

    def test_bad_rounds_values_and_counts_are_rejected(self):
        for key, value in [('round', 0), ('accuracy', float('nan')),
                           ('accuracy', 12), ('wall_seconds', -1), ('accepted', 2)]:
            with self.subTest(key=key, value=value):
                old = self.rows[1][key]
                self.rows[1][key] = value
                self.write_rows()
                with self.assertRaises(ValueError):
                    plotting.collect(self.root)
                self.rows[1][key] = old

    def test_summary_disagreement_is_rejected(self):
        self.finish()
        self.rows[-1]['accuracy'] = .5
        self.write_rows()
        with self.assertRaises(ValueError):
            plotting.collect(self.root)

    def test_png_pdf_and_csv_export(self):
        self.finish()
        output = self.root / 'figures'
        plotting.plot(plotting.collect(self.root), output, window=2)
        self.assertEqual(len(list(output.glob('*.png'))), 7)
        self.assertEqual(len(list(output.glob('*.pdf'))), 7)
        self.assertTrue((output / 'comparison.png').read_bytes().startswith(b'\x89PNG'))
        with (output / 'summary_comparison.csv').open(encoding='utf-8-sig') as handle:
            self.assertEqual(float(next(csv.DictReader(handle))['last_accuracy_pct']), 60.)


if __name__ == '__main__':
    unittest.main()

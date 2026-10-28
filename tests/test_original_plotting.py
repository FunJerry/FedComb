"""旧日志解析只接受完整且连续的轮次，不把缺失通信量补成零。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_original_results import parse_lines


class OriginalPlottingTests(unittest.TestCase):
    def test_original_units_and_pending_round(self):
        rows = parse_lines(['Client 1 Model Parts: 0-2, Aux Type: tail, P Index: 0'] * 8
                           + ['Client 2 No model parts selected'] * 2
                           + ['Epoch 0, Training Time: 3600.0, Test Loss: 1.2, Test Accuracy: 63.38',
                              'Client 3 Model Parts: 1-3, Aux Type: head, P Index: 1'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['accuracy_pct'], 63.38)
        self.assertEqual(rows[0]['round'], 1)
        self.assertEqual(rows[0]['new_client_trainings'], 8)
        self.assertIsNone(rows[0]['logged_communication_bytes'])

    def test_gaps_invalid_accuracy_and_bad_counts(self):
        plans = ['Client 1 Model Parts: 0-2'] * 10
        for index, accuracy, count in [(1, 60, 10), (0, 101, 10), (0, 60, 9)]:
            with self.assertRaises(ValueError):
                parse_lines(plans[:count] + [f'Epoch {index}, Training Time: 1, Training Comm: 20, Test Loss: 1, Test Accuracy: {accuracy}'])


if __name__ == '__main__':
    unittest.main()

"""原作者风格日志必须对应真实训练记录，不改变指标或增加训练。"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from fedcomb.config import load_config, apply_overrides
from fedcomb.experiment import Experiment


ROOT = Path(__file__).resolve().parents[1]


class TrainingLoggingTests(unittest.TestCase):
    def test_console_matches_saved_clients_and_cumulative_metrics(self):
        cfg = apply_overrides(load_config(ROOT / 'configs/smoke.json'), {
            'federation.rounds': 2, 'federation.checkpoint_every': 1})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'run'
            output = StringIO()
            with redirect_stdout(output):
                Experiment(cfg, target, ROOT).run()
            lines = output.getvalue().splitlines()
            records = [json.loads(line) for line in (target / 'rounds.jsonl').read_text().splitlines()]
            self.assertTrue(any(record['clients'] for record in records))
            position = -1
            for record in records:
                for client in record['clients']:
                    plan = client['plan']
                    parts = (f'Client {client["client"]} Model Parts: {plan["start"]}-{plan["end"]}, '
                             f'Aux Type: {plan["student_kind"]}, P Index: {plan["student_index"]}')
                    loss = f'Client {client["client"]} Epoch {record["round"]} Loss: {client["task_loss"]}'
                    position = lines.index(parts, position + 1)
                    self.assertEqual(lines[position + 1], loss)
                    position += 1
                expected = (f'Epoch {record["round"]}, Training Time: {record["cumulative_simulated_seconds"]}, '
                            f'Training Comm: {int(record["cumulative_communication_bytes"])}, '
                            f'Test Loss: {record["evaluation"]["loss"]}, '
                            f'Test Accuracy: {100 * record["evaluation"]["accuracy"]}')
                position = lines.index(expected, position + 1)


if __name__ == '__main__':
    unittest.main()

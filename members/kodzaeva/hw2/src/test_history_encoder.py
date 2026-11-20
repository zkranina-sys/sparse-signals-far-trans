from __future__ import annotations

import importlib.util
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("history_encoder", Path(__file__).with_name("history_encoder.py"))
encoder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(encoder)


def fictional_history(size=4):
    return [{"id": f"fictional-{number}", "name": "never in prompt", "buy_count": number,
             "metadata_timestamp": "2099-01-01", "attributes": ["assetCategory=stock", "sector=utilities"]}
            for number in range(1, size + 1)]


class HistoryEncoderTests(unittest.TestCase):
    def test_failed_pilot_is_reused_without_hidden_second_model_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / 'baseline'
            baseline.mkdir()
            cohort = ['fiction-u1', 'fiction-u2', 'fiction-u3']
            contexts = [{'user_id': uid, 'history': fictional_history()} for uid in cohort]
            for context in contexts:
                for item in context['history']:
                    item['id'] = context['user_id'] + '-' + item['id']
            (baseline / 'contexts.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in contexts))
            (baseline / 'protocol.json').write_text(json.dumps({'users': cohort, 'protocol_hash': 'fixture'}))
            snapshot = root / encoder.MODEL_REVISION
            snapshot.mkdir()
            model = SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda: {}))
            backend = (SimpleNamespace(__version__='fixture'), SimpleNamespace(__version__='fixture'),
                       None, model, snapshot, encoder.MODEL_REVISION)
            valid = '{"H01":1,"H02":2,"H03":3,"H04":4}'
            invalid = '{"H01":0,"H02":0,"H03":0,"H04":0}'
            base_args = ['encoder', '--baseline-dir', str(baseline), '--model-path', str(snapshot)]
            with patch.object(encoder, 'load_local_model', return_value=backend), \
                 patch.object(encoder, 'generate', side_effect=[(invalid, 10, 10, .1),
                         (valid, 10, 10, .1), (valid, 10, 10, .1)]) as generated:
                with patch('sys.argv', base_args + ['--output-dir', str(root/'pilot'), '--pilot-only']):
                    encoder.main()
                with patch('sys.argv', base_args + ['--output-dir', str(root/'full'),
                                                     '--resume-pilot-dir', str(root/'pilot')]):
                    encoder.main()
                self.assertEqual(generated.call_count, 3)
            manifest = json.loads((root/'full/run_manifest.json').read_text())
            self.assertEqual((manifest['calls'], manifest['reused'], manifest['failures']), (0, 3, 1))
            responses = encoder.read_jsonl(root/'full/responses.jsonl')
            self.assertEqual(responses[0]['history_weights'], encoder.original_weights(contexts[0]))

    def test_full_history_is_sanitized_and_missing_attributes_are_unknown(self):
        context = {"user_id": "private-person", "profile": {"risk": "private"},
                   "candidate_assets": [{"id": "future-asset"}], "eligible_asset_ids": ["future-asset"],
                   "history": fictional_history(23)}
        prompt, labels = encoder.build_prompt(context)
        self.assertEqual(len(labels), 23)
        self.assertIn("H23", prompt)
        self.assertIn("assetSubCategory=unknown", prompt)
        self.assertIn("industry=unknown", prompt)
        for forbidden in ("fictional-1", "never in prompt", "private-person", "future-asset", "metadata_timestamp"):
            self.assertNotIn(forbidden, prompt)

    def test_validation_is_strict_and_mapping_preserves_ids_outside_prompt(self):
        labels = {"H01": "fictional-a", "H02": "fictional-b"}
        self.assertEqual(encoder.validate_model_weights('{"H01": 1, "H02": 2}', labels), {"H01": 1, "H02": 2})
        for invalid in ('{"H01": 1}', '{"H01": 1, "H02": 2, "H03": 3}',
                        '{"H01": 1, "H01": 2, "H02": 3}', '{"H01": true, "H02": 1}',
                        '{"H01": -1, "H02": 1}', '{"H01": NaN, "H02": 1}', '{"H01": 0, "H02": 0}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                encoder.validate_model_weights(invalid, labels)
        prompt, labels = encoder.build_prompt({"history": fictional_history()})
        weights = encoder.validate_model_weights('{"H01": 1, "H02": 0, "H03": 2, "H04": 3}', labels)
        self.assertEqual(encoder.response_weights(labels, weights)["fictional-3"], 2)
        self.assertNotIn("fictional-3", prompt)

    def test_pilot_uses_first_second_and_last_subgroup_in_frozen_order(self):
        contexts = [{"user_id": user_id, "history": fictional_history(size)}
                    for user_id, size in (("u1", 4), ("u2", 1), ("u3", 5), ("u4", 4), ("u5", 6))]
        selected = encoder.select_pilot_contexts(contexts, ["u1", "u2", "u3", "u4", "u5"])
        self.assertEqual([item["user_id"] for item in selected], ["u1", "u3", "u5"])


if __name__ == "__main__":
    unittest.main()

"""CPU regressions: real Praat DSP; mocked model inference and GPU calls."""
from __future__ import annotations

import importlib
import json
import math
import sys
import tempfile
import time
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import parselmouth
import soundfile as sf
from scipy.signal import resample_poly

import tune


# Import orchestration without downloading Torch or any model weights.
# DSP assertions below still use the actual Praat and SoundFile packages.
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=Mock())
fake_torch.from_numpy = lambda x: x
with patch.dict(sys.modules, {"torch": fake_torch}):
    instrumental = importlib.import_module("instrumental")
    handler = importlib.import_module("handler")


def tone(cents=35, sr=24000, seconds=2):
    t = np.arange(int(sr * seconds)) / sr
    return (0.2 * np.sin(2 * np.pi * 440 * 2 ** (cents / 1200) * t)).astype(np.float32)


class PitchTests(unittest.TestCase):
    def setUp(self):
        self.sr = 24000
        self.key = tune.parse_keyscale("A minor")

    def test_natural_strength_reduces_measured_pitch_error(self):
        source = tone()
        result, report = tune.tune_to_key(source, self.sr, self.key, 0.65)
        self.assertTrue(report["applied"])
        # Measure the output independently of the tuner report.
        pitch = parselmouth.Sound(result, sampling_frequency=self.sr).to_pitch_ac()
        hz = pitch.selected_array["frequency"]
        cents = 1200 * np.log2(hz[hz > 0] / 440)
        self.assertAlmostEqual(float(np.median(cents)), 12.25, delta=2)
        self.assertEqual(result.shape, source.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_zero_strength_leaves_audio_exactly_unchanged(self):
        source = tone()
        result, report = tune.tune_to_key(source, self.sr, self.key, 0)
        np.testing.assert_array_equal(result, source)
        self.assertFalse(report["applied"])

    def test_silence_and_in_tune_audio_are_unchanged(self):
        for source in (np.zeros(self.sr * 2, np.float32), tone(cents=0)):
            with self.subTest(silent=not source.any()):
                result, report = tune.tune_to_key(source, self.sr, self.key)
                np.testing.assert_array_equal(result, source)
                self.assertFalse(report["applied"])

    def test_stereo_duration_and_channel_balance_are_preserved(self):
        mono = tone()
        source = np.column_stack([mono, mono * 0.5])
        result, report = tune.tune_to_key(source, self.sr, self.key)
        self.assertTrue(report["applied"])
        self.assertEqual(result.shape, source.shape)
        np.testing.assert_allclose(result[:, 1], result[:, 0] * 0.5, atol=1e-6)

    def test_unvoiced_audio_away_from_notes_is_unchanged(self):
        rng = np.random.default_rng(17)
        noise = rng.normal(0, 0.02, self.sr // 2).astype(np.float32)
        source = np.concatenate([noise, np.zeros(self.sr // 4), tone()]).astype(np.float32)
        result, _ = tune.tune_to_key(source, self.sr, self.key)
        np.testing.assert_array_equal(result[:self.sr // 4], source[:self.sr // 4])

    def test_unpitched_take_has_no_numeric_pitch_score(self):
        report = tune.pitch_score(np.zeros(self.sr * 2), self.sr, self.key)
        self.assertEqual(report["notes"], 0)
        self.assertIsNone(report["score"])
        json.dumps(report, allow_nan=False)

    def test_unpitched_take_cannot_outrank_measurable_singing(self):
        unpitched = tune.pitch_score(np.zeros(self.sr * 2), self.sr, self.key)
        off_key = tune.pitch_score(tone(cents=80), self.sr, self.key)
        self.assertGreater(off_key["score"], 50)
        self.assertLess(tune.take_rank(off_key), tune.take_rank(unpitched))
        self.assertEqual(tune.take_rank(unpitched), tune.take_rank({"notes": 0, "score": None}))


def cpu_resample(audio, src_sr, dst_sr):
    divisor = math.gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // divisor, src_sr // divisor, axis=0).astype(np.float32)


class SeparationTests(unittest.TestCase):
    def test_each_stem_is_resampled_and_aligned_before_subtraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "vocals.wav")
            sf.write(path, np.ones(8000, np.float32) * 0.125, 8000, subtype="FLOAT")
            with patch.object(instrumental, "_resample", side_effect=cpu_resample) as resample:
                result = instrumental._read_stem(path, 44100, 44110)
            resample.assert_called_once()
            self.assertEqual(result.shape, (44110, 2))
            np.testing.assert_allclose(result[1000:43000], 0.125, atol=0.001)
            np.testing.assert_array_equal(result[44100:], 0)

    def test_invalid_stem_raises_so_handler_can_fall_back(self):
        with patch.object(instrumental.sf, "read", return_value=(np.array([[np.nan, 0]], np.float32), 44100)):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                instrumental._read_stem("unused", 44100, 100)

    def test_short_single_stem_keeps_the_remaining_beat(self):
        class Separator:
            def __init__(self, **kwargs):
                self.directory = Path(kwargs["output_dir"])

            def load_model(self, **kwargs):
                pass

            def separate(self, path):
                name = "beat_(Vocals).wav"
                sf.write(self.directory / name, np.full((50, 2), 0.2, np.float32), 44100, subtype="FLOAT")
                return [name]

        module = types.ModuleType("audio_separator.separator")
        module.Separator = Separator
        with patch.dict(sys.modules, {"audio_separator": types.ModuleType("audio_separator"), "audio_separator.separator": module}):
            beat = np.full((100, 2), 0.5, np.float32)
            inst, vocal, _, _ = instrumental.separate(beat, 44100)
        np.testing.assert_allclose(inst[:50], 0.3)
        np.testing.assert_allclose(inst[50:], 0.5)
        np.testing.assert_allclose(inst + vocal, beat)

    def test_gpu_cleanup_runs_if_model_load_fails(self):
        separator = Mock()
        separator.load_model.side_effect = RuntimeError("load failed")
        module = types.ModuleType("audio_separator.separator")
        module.Separator = Mock(return_value=separator)
        with patch.dict(sys.modules, {"audio_separator": types.ModuleType("audio_separator"), "audio_separator.separator": module}), \
                patch.object(instrumental.torch.cuda, "is_available", return_value=True), \
                patch.object(instrumental.torch.cuda, "empty_cache") as empty_cache:
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                instrumental.separate(np.zeros((100, 2), np.float32), 44100)
            empty_cache.assert_called_once()


class TakeSelectionTests(unittest.TestCase):
    def test_handler_chooses_sung_take_after_unpitched_take(self):
        sr = 24000
        beat = np.column_stack([tone(cents=0), tone(cents=0)])
        candidates = [np.zeros_like(beat), np.column_stack([tone(cents=80), tone(cents=80)])]
        job = {
            "beat": {"vocals": "keep", "start": 0}, "user_key": tune.parse_keyscale("A minor"),
            "keyscale": "A minor", "timesignature": "4", "takes": 2, "seed": 11,
            "tags": "dry vocals", "lyrics": "test", "bpm": 92, "vocal_language": "en",
            "tune": False, "tune_strength": 0.65, "layers": [], "return_untuned": False, "return_raw": False,
        }
        generated = [(audio, sr, 0.1, "test", {"ok": True}) for audio in candidates]
        with ExitStack() as stack:
            for name, value in {
                "load_beat": beat, "detect_and_strip_vocal": (beat, {}, None, None),
                "cuda_free": None,
            }.items():
                stack.enter_context(patch.object(handler, name, return_value=value))
            stack.enter_context(patch.object(handler, "generate_mix", side_effect=generated))
            stack.enter_context(patch.object(handler.vlayers, "estimate_key", return_value=job["user_key"]))
            stack.enter_context(patch.object(handler, "separate_vocals", side_effect=lambda x, sr: (x, sr, 0)))
            stack.enter_context(patch.object(handler, "dereverb_vocals", side_effect=lambda x, sr: (x, sr, 0)))
            stack.enter_context(patch.object(handler, "gate_vocal", side_effect=lambda x, sr: (x, {})))
            stack.enter_context(patch.object(handler, "encode_mp3", side_effect=lambda x, sr, path, **kw: path.write_bytes(b"test-output")))
            result = handler.handle_beat(types.SimpleNamespace(sample_rate=sr), job, time.time(), "test-cpu")
        self.assertEqual(result["seed"], 12)
        self.assertEqual(result["seed_requested"], 11)
        self.assertIsNone(result["takes"][0]["score"])
        self.assertEqual(result["takes"][1]["notes"], 1)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

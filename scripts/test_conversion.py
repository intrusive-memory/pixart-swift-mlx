#!/usr/bin/env python3
"""
Unit tests for weight conversion scripts.

Tests the pure functions in convert_pixart_weights.py, convert_t5_weights.py,
convert_vae_weights.py, and validate_conversion.py without downloading models.

Run: python3 -m unittest scripts/test_conversion.py
  or: python3 -m pytest scripts/test_conversion.py -v
"""

import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Stub out heavy dependencies (torch, diffusers, transformers, safetensors)
# so we can import the pure-numpy functions from the conversion scripts
# without needing PyTorch installed.
# ---------------------------------------------------------------------------
_STUBS = {}
for mod_name in [
    "torch", "torch.cuda", "torch.backends", "torch.backends.mps",
    "diffusers", "transformers",
    "safetensors", "safetensors.numpy", "safetensors.torch",
]:
    stub = types.ModuleType(mod_name)
    sys.modules[mod_name] = stub
    _STUBS[mod_name] = stub

# safetensors.numpy needs a save_file attribute
sys.modules["safetensors.numpy"].save_file = MagicMock()

# transformers needs importable names
for attr in ["T5EncoderModel", "T5Tokenizer", "AutoTokenizer"]:
    setattr(sys.modules["transformers"], attr, MagicMock())

# Add scripts/ to path so we can import the conversion modules
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from convert_pixart_weights import (
    build_key_mapping,
    quantize_int4 as pixart_quantize_int4,
    should_skip_quantization as pixart_should_skip,
    DISCARDED_KEYS as PIXART_DISCARDED_KEYS,
    NUM_BLOCKS,
    GROUP_SIZE,
)
from convert_t5_weights import (
    quantize_int4 as t5_quantize_int4,
    should_skip_quantization as t5_should_skip,
)
from convert_vae_weights import transpose_conv2d
from validate_conversion import compute_psnr, dequantize_int4


# ---------------------------------------------------------------------------
# Task 1: compute_psnr()
# ---------------------------------------------------------------------------


class TestComputePSNR(unittest.TestCase):
    """Tests for validate_conversion.compute_psnr()."""

    def test_identical_arrays_return_inf(self):
        a = np.random.randn(64, 64).astype(np.float32)
        self.assertEqual(compute_psnr(a, a), float("inf"))

    def test_known_mse(self):
        """For a known constant difference, PSNR should match manual calculation."""
        ref = np.ones((100,), dtype=np.float32) * 10.0
        # Add constant error of 1.0 -> MSE = 1.0
        test = ref + 1.0
        psnr = compute_psnr(ref, test)
        # PSNR = 20*log10(max_val) - 10*log10(MSE) = 20*log10(10) - 10*log10(1) = 20 - 0 = 20
        self.assertAlmostEqual(psnr, 20.0, places=1)

    def test_shape_mismatch_raises(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.zeros((4, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_psnr(a, b)

    def test_zero_reference_uses_unit_peak(self):
        """When reference is all zeros, max_val falls back to 1.0."""
        ref = np.zeros((10,), dtype=np.float32)
        test = np.ones((10,), dtype=np.float32) * 0.1
        # MSE = 0.01, max_val=1.0, PSNR = 20*log10(1) - 10*log10(0.01) = 0 + 20 = 20
        psnr = compute_psnr(ref, test)
        self.assertAlmostEqual(psnr, 20.0, places=1)

    def test_psnr_decreases_with_more_noise(self):
        ref = np.random.randn(256).astype(np.float32)
        small_noise = ref + np.random.randn(256).astype(np.float32) * 0.01
        large_noise = ref + np.random.randn(256).astype(np.float32) * 1.0
        self.assertGreater(compute_psnr(ref, small_noise), compute_psnr(ref, large_noise))


# ---------------------------------------------------------------------------
# Task 2: quantize_int4() / dequantize_int4() roundtrip
# ---------------------------------------------------------------------------


class TestQuantizeInt4Roundtrip(unittest.TestCase):
    """Tests for int4 quantization and dequantization roundtrip."""

    def _roundtrip(self, weight: np.ndarray) -> float:
        """Quantize then dequantize and return PSNR vs original."""
        q = pixart_quantize_int4(weight)
        deq = dequantize_int4(
            q["weight"],
            q["scales"].astype(np.float32),
            q["biases"].astype(np.float32),
        )
        return compute_psnr(weight.astype(np.float32), deq)

    def test_random_weight_roundtrip_psnr(self):
        """Random weights should survive int4 roundtrip with PSNR > 25 dB."""
        np.random.seed(42)
        weight = np.random.randn(128, 256).astype(np.float32)
        psnr = self._roundtrip(weight)
        self.assertGreater(psnr, 25.0, f"Roundtrip PSNR {psnr:.1f} dB too low")

    def test_constant_weight_roundtrip(self):
        """Constant weights should survive roundtrip with very high PSNR.

        Not exactly inf because scales/biases are stored as float16,
        introducing small precision loss on the zero-point.
        """
        weight = np.full((64, 128), 3.14, dtype=np.float32)
        psnr = self._roundtrip(weight)
        self.assertGreater(psnr, 60.0, f"Constant roundtrip PSNR {psnr:.1f} dB too low")

    def test_zero_weight_roundtrip(self):
        """All-zero weights should roundtrip exactly."""
        weight = np.zeros((64, 128), dtype=np.float32)
        q = pixart_quantize_int4(weight)
        deq = dequantize_int4(
            q["weight"],
            q["scales"].astype(np.float32),
            q["biases"].astype(np.float32),
        )
        np.testing.assert_allclose(deq, weight, atol=1e-6)

    def test_output_shapes(self):
        """Verify packed weight, scales, biases have correct shapes."""
        M, N = 64, 256
        weight = np.random.randn(M, N).astype(np.float32)
        q = pixart_quantize_int4(weight)
        self.assertEqual(q["weight"].shape, (M, N // 8))
        self.assertEqual(q["weight"].dtype, np.uint32)
        self.assertEqual(q["scales"].shape, (M, N // GROUP_SIZE))
        self.assertEqual(q["scales"].dtype, np.float16)
        self.assertEqual(q["biases"].shape, (M, N // GROUP_SIZE))
        self.assertEqual(q["biases"].dtype, np.float16)

    def test_non_2d_raises(self):
        with self.assertRaises(AssertionError):
            pixart_quantize_int4(np.zeros((4, 4, 4), dtype=np.float32))

    def test_n_not_divisible_by_group_size_raises(self):
        """PixArt quantizer requires N divisible by GROUP_SIZE."""
        with self.assertRaises(AssertionError):
            pixart_quantize_int4(np.zeros((4, 100), dtype=np.float32))

    def test_t5_quantizer_pads_n(self):
        """T5 quantizer pads N to GROUP_SIZE boundary instead of asserting."""
        weight = np.random.randn(4, 100).astype(np.float32)
        q = t5_quantize_int4(weight)
        # Padded to 128 (next multiple of 64)
        self.assertEqual(q["weight"].shape, (4, 128 // 8))

    def test_packed_values_in_range(self):
        """Each 4-bit nibble in packed uint32 should be in [0, 15]."""
        weight = np.random.randn(32, 128).astype(np.float32)
        q = pixart_quantize_int4(weight)
        packed = q["weight"]
        for bit_idx in range(8):
            nibbles = (packed >> (4 * bit_idx)) & 0xF
            self.assertTrue(np.all(nibbles <= 15))
            self.assertTrue(np.all(nibbles >= 0))


# ---------------------------------------------------------------------------
# Task 3: build_key_mapping()
# ---------------------------------------------------------------------------


class TestBuildKeyMapping(unittest.TestCase):
    """Tests for convert_pixart_weights.build_key_mapping()."""

    def setUp(self):
        self.mapping = build_key_mapping()

    def test_expected_key_count(self):
        """23 global keys + 28 blocks * 23 per-block keys = 667 total."""
        # Global: patch embed 2 + caption proj 4 + timestep 4 + resolution 4
        #       + aspect ratio 4 + t_block 2 + final proj 2 + scale_shift_table 1 = 23
        # Per-block: scale_shift_table 1 + self-attn 8 + qk_norm 4
        #          + cross-attn 8 + ffn 4 = 25
        # But let's just count the actual mapping
        global_keys = 23
        per_block_keys = 25  # counted from the code
        expected = global_keys + NUM_BLOCKS * per_block_keys
        self.assertEqual(len(self.mapping), expected)

    def test_no_duplicate_values(self):
        """All MLX key values should be unique (no two HF keys map to same MLX key)."""
        values = list(self.mapping.values())
        self.assertEqual(len(values), len(set(values)))

    def test_all_28_blocks_present(self):
        """Every block index 0..27 should appear in both keys and values."""
        for i in range(NUM_BLOCKS):
            hf_keys = [k for k in self.mapping if f"transformer_blocks.{i}." in k]
            mlx_keys = [v for v in self.mapping.values() if f"blocks.{i}." in v]
            self.assertGreater(len(hf_keys), 0, f"Block {i} missing from HF keys")
            self.assertGreater(len(mlx_keys), 0, f"Block {i} missing from MLX values")

    def test_global_keys_present(self):
        """Spot-check that all global key groups exist."""
        self.assertIn("pos_embed.proj.weight", self.mapping)
        self.assertIn("caption_projection.linear_1.weight", self.mapping)
        self.assertIn("adaln_single.emb.timestep_embedder.linear_1.weight", self.mapping)
        self.assertIn("adaln_single.emb.resolution_embedder.linear_1.weight", self.mapping)
        self.assertIn("adaln_single.emb.aspect_ratio_embedder.linear_1.weight", self.mapping)
        self.assertIn("adaln_single.linear.weight", self.mapping)
        self.assertIn("proj_out.weight", self.mapping)
        self.assertIn("scale_shift_table", self.mapping)

    def test_discarded_keys_not_in_mapping(self):
        """Discarded keys should not appear in the mapping table."""
        for k in PIXART_DISCARDED_KEYS:
            self.assertNotIn(k, self.mapping)

    def test_per_block_key_groups_complete(self):
        """Block 0 should have all expected sub-keys."""
        block0_keys = [k for k in self.mapping if k.startswith("transformer_blocks.0.")]
        suffixes = {k.replace("transformer_blocks.0.", "") for k in block0_keys}
        expected_suffixes = {
            "scale_shift_table",
            "attn1.to_q.weight", "attn1.to_q.bias",
            "attn1.to_k.weight", "attn1.to_k.bias",
            "attn1.to_v.weight", "attn1.to_v.bias",
            "attn1.to_out.0.weight", "attn1.to_out.0.bias",
            "attn1.q_norm.weight", "attn1.q_norm.bias",
            "attn1.k_norm.weight", "attn1.k_norm.bias",
            "attn2.to_q.weight", "attn2.to_q.bias",
            "attn2.to_k.weight", "attn2.to_k.bias",
            "attn2.to_v.weight", "attn2.to_v.bias",
            "attn2.to_out.0.weight", "attn2.to_out.0.bias",
            "ff.net.0.proj.weight", "ff.net.0.proj.bias",
            "ff.net.2.weight", "ff.net.2.bias",
        }
        self.assertEqual(suffixes, expected_suffixes)


# ---------------------------------------------------------------------------
# Task 4: should_skip_quantization()
# ---------------------------------------------------------------------------


class TestShouldSkipQuantizationPixArt(unittest.TestCase):
    """Tests for convert_pixart_weights.should_skip_quantization()."""

    def test_scale_shift_table_skipped(self):
        self.assertTrue(pixart_should_skip("finalLayer.scaleShiftTable"))
        self.assertTrue(pixart_should_skip("blocks.0.scaleShiftTable"))

    def test_qk_norms_skipped(self):
        self.assertTrue(pixart_should_skip("blocks.0.attn.q_norm.weight"))
        self.assertTrue(pixart_should_skip("blocks.0.attn.k_norm.bias"))

    def test_patch_embed_skipped(self):
        self.assertTrue(pixart_should_skip("patchEmbed.weight"))
        self.assertTrue(pixart_should_skip("patchEmbed.bias"))

    def test_bias_tensors_skipped(self):
        self.assertTrue(pixart_should_skip("blocks.0.attn.to_q.bias"))
        self.assertTrue(pixart_should_skip("captionProjection.linear1.bias"))

    def test_linear_weights_not_skipped(self):
        self.assertFalse(pixart_should_skip("blocks.0.attn.to_q.weight"))
        self.assertFalse(pixart_should_skip("captionProjection.linear1.weight"))
        self.assertFalse(pixart_should_skip("blocks.0.mlp.fc1.weight"))


class TestShouldSkipQuantizationT5(unittest.TestCase):
    """Tests for convert_t5_weights.should_skip_quantization()."""

    def test_shared_weight_skipped(self):
        dummy = np.zeros((10, 10))
        self.assertTrue(t5_should_skip("shared.weight", dummy))

    def test_relative_attention_bias_skipped(self):
        dummy = np.zeros((10, 10))
        self.assertTrue(t5_should_skip("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", dummy))

    def test_layer_norm_skipped(self):
        dummy = np.zeros((10,))
        self.assertTrue(t5_should_skip("encoder.block.0.layer.0.layer_norm.weight", dummy))
        self.assertTrue(t5_should_skip("encoder.final_layer_norm.weight", dummy))

    def test_bias_skipped(self):
        dummy = np.zeros((10,))
        self.assertTrue(t5_should_skip("some_module.bias", dummy))

    def test_1d_tensor_skipped(self):
        dummy_1d = np.zeros((10,))
        self.assertTrue(t5_should_skip("encoder.block.0.layer.1.DenseReluDense.wi.weight", dummy_1d))

    def test_2d_linear_weight_not_skipped(self):
        dummy_2d = np.zeros((10, 10))
        self.assertFalse(t5_should_skip("encoder.block.0.layer.1.DenseReluDense.wi.weight", dummy_2d))


# ---------------------------------------------------------------------------
# Task 5: transpose_conv2d()
# ---------------------------------------------------------------------------


class TestTransposeConv2d(unittest.TestCase):
    """Tests for convert_vae_weights.transpose_conv2d()."""

    def test_4d_transposition(self):
        """[O,I,kH,kW] -> [O,kH,kW,I]"""
        O, I, kH, kW = 3, 4, 5, 6
        weight = np.arange(O * I * kH * kW).reshape(O, I, kH, kW).astype(np.float32)
        result = transpose_conv2d(weight)
        self.assertEqual(result.shape, (O, kH, kW, I))
        # Verify specific element: weight[o,i,h,w] should be at result[o,h,w,i]
        for o in range(O):
            for i in range(I):
                for h in range(kH):
                    for w in range(kW):
                        self.assertEqual(weight[o, i, h, w], result[o, h, w, i])

    def test_non_4d_passthrough(self):
        """Non-4D arrays should pass through unchanged."""
        weight_2d = np.ones((3, 4), dtype=np.float32)
        result = transpose_conv2d(weight_2d)
        np.testing.assert_array_equal(result, weight_2d)

        weight_1d = np.ones((10,), dtype=np.float32)
        result = transpose_conv2d(weight_1d)
        np.testing.assert_array_equal(result, weight_1d)


# ---------------------------------------------------------------------------
# Task 6: Cross-language key mapping consistency
# ---------------------------------------------------------------------------


class TestKeyMappingConsistency(unittest.TestCase):
    """
    Verify that Python's build_key_mapping() produces the same key pairs
    as WeightMapping.swift's pixArtKeyTable.
    """

    @classmethod
    def setUpClass(cls):
        """Parse WeightMapping.swift to extract key pairs."""
        swift_path = SCRIPTS_DIR.parent / "Sources" / "PixArtBackbone" / "WeightMapping.swift"
        if not swift_path.exists():
            raise unittest.SkipTest(f"WeightMapping.swift not found at {swift_path}")

        cls.swift_pairs = {}
        cls.swift_discarded = set()
        content = swift_path.read_text()

        # Pattern for table["literal_key"] = "literal_value" (no interpolation)
        literal_pattern = re.compile(
            r'table\["([^"\\]+)"\]\s*=\s*\n?\s*"([^"\\]+)"'
        )
        # Pattern for table["\(var)..."] = "\(var)..." (with string interpolation)
        interp_pattern = re.compile(
            r'table\["([^"]+)"\]\s*=\s*\n?\s*"([^"]+)"'
        )

        # First pass: collect all raw pairs (including interpolated ones)
        raw_pairs = []
        for match in interp_pattern.finditer(content):
            raw_pairs.append((match.group(1), match.group(2)))

        # Expand string interpolations for the for-loop block
        # The Swift code uses: let hf = "transformer_blocks.\(i)"
        #                       let mlx = "blocks.\(i)"
        for raw_hf, raw_mlx in raw_pairs:
            if "\\(hf)" in raw_hf or "\\(mlx)" in raw_mlx:
                # This is inside the for i in 0..<28 loop
                for i in range(28):
                    hf_key = raw_hf.replace("\\(hf)", f"transformer_blocks.{i}")
                    mlx_key = raw_mlx.replace("\\(mlx)", f"blocks.{i}")
                    cls.swift_pairs[hf_key] = mlx_key
            else:
                cls.swift_pairs[raw_hf] = raw_mlx

        # Extract discarded keys
        discard_pattern = re.compile(r'"([^"\\]+)"')
        in_discarded = False
        for line in content.splitlines():
            if "pixArtDiscardedKeys" in line:
                in_discarded = True
            if in_discarded:
                for m in discard_pattern.finditer(line):
                    cls.swift_discarded.add(m.group(1))
                if "]" in line and in_discarded and "pixArtDiscardedKeys" not in line:
                    in_discarded = False

    def test_python_keys_match_swift(self):
        """Every Python key mapping should exist in Swift."""
        py_mapping = build_key_mapping()
        missing_in_swift = {k: v for k, v in py_mapping.items() if k not in self.swift_pairs}
        self.assertEqual(
            len(missing_in_swift), 0,
            f"Python has {len(missing_in_swift)} keys not in Swift: {list(missing_in_swift.keys())[:5]}..."
        )

    def test_swift_keys_match_python(self):
        """Every Swift key mapping should exist in Python."""
        py_mapping = build_key_mapping()
        missing_in_python = {k: v for k, v in self.swift_pairs.items() if k not in py_mapping}
        self.assertEqual(
            len(missing_in_python), 0,
            f"Swift has {len(missing_in_python)} keys not in Python: {list(missing_in_python.keys())[:5]}..."
        )

    def test_values_match(self):
        """For every shared key, the MLX value should be identical."""
        py_mapping = build_key_mapping()
        mismatches = []
        for hf_key in py_mapping:
            if hf_key in self.swift_pairs:
                if py_mapping[hf_key] != self.swift_pairs[hf_key]:
                    mismatches.append(
                        f"  {hf_key}: py={py_mapping[hf_key]} swift={self.swift_pairs[hf_key]}"
                    )
        self.assertEqual(
            len(mismatches), 0,
            f"Value mismatches:\n" + "\n".join(mismatches[:10])
        )

    def test_discarded_keys_match(self):
        """Python and Swift should agree on discarded keys."""
        self.assertEqual(PIXART_DISCARDED_KEYS, self.swift_discarded)

    def test_key_counts_match(self):
        """Total number of key mappings should be the same."""
        py_mapping = build_key_mapping()
        self.assertEqual(len(py_mapping), len(self.swift_pairs))


if __name__ == "__main__":
    unittest.main()

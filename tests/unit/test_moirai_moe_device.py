"""Unit tests for Moirai-MoE device resolution (GPU vs CPU)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tempus_bench.models.moirai_moe.device_resolve import resolve_moirai_moe_torch_device


class TestResolveMoiraiMoeTorchDevice(unittest.TestCase):
    def test_cpu_literal_returns_cpu(self) -> None:
        self.assertEqual(resolve_moirai_moe_torch_device("cpu").type, "cpu")

    def test_auto_selects_cpu_when_cuda_unavailable(self) -> None:
        with patch(
            "tempus_bench.models.moirai_moe.device_resolve.torch.cuda.is_available",
            return_value=False,
        ):
            d = resolve_moirai_moe_torch_device("auto")
        self.assertEqual(d.type, "cpu")

    def test_auto_selects_cuda_when_cuda_available(self) -> None:
        with patch(
            "tempus_bench.models.moirai_moe.device_resolve.torch.cuda.is_available",
            return_value=True,
        ):
            d = resolve_moirai_moe_torch_device("auto")
        self.assertEqual(d.type, "cuda")
        self.assertIn(d.index, (None, 0))

    def test_cuda_raises_when_unavailable(self) -> None:
        with patch(
            "tempus_bench.models.moirai_moe.device_resolve.torch.cuda.is_available",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_moirai_moe_torch_device("cuda")
        self.assertIn("cuda", str(ctx.exception).lower())

    def test_cuda_index_preserved(self) -> None:
        with patch(
            "tempus_bench.models.moirai_moe.device_resolve.torch.cuda.is_available",
            return_value=True,
        ):
            d = resolve_moirai_moe_torch_device("cuda:1")
        self.assertEqual(d.type, "cuda")
        self.assertEqual(d.index, 1)


if __name__ == "__main__":
    unittest.main()

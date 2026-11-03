from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from stegtext.v2.model.device import resolve_physical_device


class DeviceIsolationTests(unittest.TestCase):
    def test_visible_gpu1_maps_to_logical_cuda0(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False), patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_physical_device(1), "cuda:0")
            with self.assertRaises(RuntimeError):
                resolve_physical_device(0)

    def test_unmasked_physical_gpu1_maps_to_cuda1(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=2):
            self.assertEqual(resolve_physical_device(1), "cuda:1")

    def test_unavailable_gpu1_fails_without_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=1):
            with self.assertRaises(RuntimeError):
                resolve_physical_device(1)


if __name__ == "__main__":
    unittest.main()

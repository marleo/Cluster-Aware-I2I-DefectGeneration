from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cluster_defects.config import Config
from cluster_defects.dataset import read_yolo_boxes
from cluster_defects.similarity import leave_one_out_scores, top_k_mean_similarity
from cluster_defects.workflow import build_img2img_api_workflow
from cluster_defects.report import _extract_hog_lbp


class DatasetTests(unittest.TestCase):
    def test_yolo_box_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.5 0.25 0.50\n", encoding="utf-8")
            box = read_yolo_boxes(label, 100, 200)[0]
        self.assertEqual(box.class_id, 0)
        self.assertAlmostEqual(box.x1, 37.5)
        self.assertAlmostEqual(box.y1, 50.0)
        self.assertAlmostEqual(box.x2, 62.5)
        self.assertAlmostEqual(box.y2, 150.0)


class SimilarityTests(unittest.TestCase):
    def test_top_k_and_leave_one_out(self) -> None:
        embeddings = torch.eye(3)
        score = top_k_mean_similarity(embeddings[:1], embeddings, k=1)
        self.assertAlmostEqual(float(score[0]), 1.0)
        top_k, maximum = leave_one_out_scores(embeddings, k=1)
        self.assertTrue(torch.allclose(top_k, torch.zeros(3)))
        self.assertTrue(torch.allclose(maximum, torch.zeros(3)))


class WorkflowTests(unittest.TestCase):
    def test_full_lora_and_img2img_connections(self) -> None:
        workflow = build_img2img_api_workflow(
            input_image="source.jpg",
            checkpoint="v1-5-pruned.safetensors",
            lora="crack.safetensors",
            lora_model_strength=0.85,
            lora_clip_strength=0.80,
            positive_prompt="crack",
            negative_prompt="text",
            width=512,
            height=512,
            seed=1,
            steps=30,
            cfg=6.5,
            sampler_name="dpmpp_2m",
            scheduler="karras",
            denoise=0.42,
            filename_prefix="test",
        )
        self.assertEqual(workflow["5"]["class_type"], "LoraLoader")
        self.assertEqual(workflow["6"]["inputs"]["clip"], ["5", 1])
        self.assertEqual(workflow["8"]["inputs"]["model"], ["5", 0])
        self.assertEqual(workflow["8"]["inputs"]["latent_image"], ["4", 0])
        self.assertEqual(workflow["4"]["inputs"]["pixels"], ["3", 0])


class ConfigTests(unittest.TestCase):
    def test_relative_output_path_is_project_relative(self) -> None:
        config_path = Path(__file__).parents[1] / "config.toml"
        config = Config.load(config_path)
        self.assertEqual(config.output_root.name, "outputs")
        self.assertEqual(config.output_root.parent, config_path.parent.resolve())


class ReportFeatureTests(unittest.TestCase):
    def test_hog_and_lbp_descriptors_are_finite(self) -> None:
        image = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
        hog_features, lbp_features = _extract_hog_lbp(image)

        self.assertEqual(hog_features.shape, (1764,))
        self.assertEqual(lbp_features.shape, (28,))
        self.assertTrue(np.isfinite(hog_features).all())
        self.assertTrue(np.isfinite(lbp_features).all())
        self.assertAlmostEqual(float(lbp_features[:10].sum()), 1.0)
        self.assertAlmostEqual(float(lbp_features[10:].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()

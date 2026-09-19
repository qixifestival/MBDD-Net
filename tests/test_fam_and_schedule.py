import sys
import unittest
from pathlib import Path

import torch


TRAIN_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(TRAIN_ROOT))

from calc_utils.calc_loss import MBDDLoss
from model.mbdd.blocks.flow_resnet import FlowN_ResNet


class FAMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = FlowN_ResNet(64, window_size=4).eval()

    def test_window_partition_round_trip(self):
        x = torch.randn(2, 64, 15, 17)
        windows, shape = self.model._window_partition(x)
        restored = self.model._window_reverse(windows, shape)
        torch.testing.assert_close(restored, x)

    def test_output_is_independent_of_batch_companions(self):
        pre = torch.randn(1, 64, 8, 8)
        post = torch.randn(1, 64, 8, 8)
        other_pre = torch.randn(1, 64, 8, 8)
        other_post = torch.randn(1, 64, 8, 8)

        with torch.no_grad():
            output_single, flow_single = self.model(pre, post)
            output_batch, flow_batch = self.model(
                torch.cat([pre, other_pre]),
                torch.cat([post, other_post]),
            )

        torch.testing.assert_close(output_single, output_batch[:1])
        torch.testing.assert_close(flow_single, flow_batch[:1])

    def test_batch_order_does_not_change_corresponding_output(self):
        pre_a = torch.randn(1, 64, 8, 8)
        post_a = torch.randn(1, 64, 8, 8)
        pre_b = torch.randn(1, 64, 8, 8)
        post_b = torch.randn(1, 64, 8, 8)

        with torch.no_grad():
            output_ab, flow_ab = self.model(
                torch.cat([pre_a, pre_b]),
                torch.cat([post_a, post_b]),
            )
            output_ba, flow_ba = self.model(
                torch.cat([pre_b, pre_a]),
                torch.cat([post_b, post_a]),
            )

        torch.testing.assert_close(output_ab[0], output_ba[1])
        torch.testing.assert_close(flow_ab[0], flow_ba[1])

    def test_zero_flow_is_identity(self):
        x = torch.randn(2, 64, 7, 9)
        flow = torch.zeros(2, 2, 7, 9)
        with torch.no_grad():
            warped = self.model.flow_warp(x, flow, (7, 9))
        torch.testing.assert_close(warped, x, rtol=1e-5, atol=1e-6)

    def test_forward_and_gradients_are_finite(self):
        model = FlowN_ResNet(64, window_size=4).train()
        pre = torch.randn(2, 64, 8, 8, requires_grad=True)
        post = torch.randn(2, 64, 8, 8, requires_grad=True)
        output, flow = model(pre, post)
        loss = output.square().mean() + flow.square().mean()
        loss.backward()

        self.assertEqual(output.shape, post.shape)
        self.assertEqual(flow.shape, (2, 2, 8, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(flow).all())
        self.assertTrue(torch.isfinite(pre.grad).all())
        self.assertTrue(torch.isfinite(post.grad).all())


class AuxiliaryWeightScheduleTests(unittest.TestCase):
    def test_schedule_values_and_boundaries(self):
        loss = MBDDLoss(max_iters=1000)
        expected = {
            0: (0.2, 0.5),
            849: (0.2, 0.5),
            850: (0.2, 0.5),
            900: (0.1, 0.35),
            950: (0.0, 0.2),
            1000: (0.0, 0.2),
        }
        for step, target in expected.items():
            with self.subTest(step=step):
                loss.current_step = step
                actual = loss._get_dynamic_aux_weight()
                self.assertAlmostEqual(actual[0], target[0], places=7)
                self.assertAlmostEqual(actual[1], target[1], places=7)


if __name__ == "__main__":
    unittest.main()

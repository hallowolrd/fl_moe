from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402
import omoe_uniform as omoe  # noqa: E402
import torch  # noqa: E402


class OMoEGramSchmidtTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_well_conditioned_outputs_are_orthonormal(self) -> None:
        expert_outputs = torch.randn(5, 4, 12)
        selected_experts = torch.tensor([0, 3, 1, 2, 0], dtype=torch.long)

        orthogonalized = omoe._gram_schmidt_expert_outputs(
            expert_outputs,
            selected_experts,
        )

        self.assertEqual(orthogonalized.shape, expert_outputs.shape)
        self.assertEqual(orthogonalized.dtype, expert_outputs.dtype)
        self.assertEqual(orthogonalized.device, expert_outputs.device)

        gram = torch.matmul(
            orthogonalized,
            orthogonalized.transpose(1, 2),
        )
        identity = torch.eye(4).expand(5, -1, -1)
        self.assertTrue(torch.allclose(gram, identity, atol=2e-4, rtol=2e-4))

    def test_selected_expert_is_orthogonalized_after_all_other_experts(self) -> None:
        expert_outputs = torch.randn(4, 4, 10)
        selected_experts = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        orthogonalized = omoe._gram_schmidt_expert_outputs(
            expert_outputs,
            selected_experts,
        )

        for token_index, selected_id in enumerate(selected_experts.tolist()):
            other_ids = [
                expert_id
                for expert_id in range(4)
                if expert_id != selected_id
            ]
            selected_output = orthogonalized[token_index, selected_id]
            original_other_outputs = expert_outputs[token_index, other_ids]
            inner_products = torch.mv(
                original_other_outputs,
                selected_output,
            )
            self.assertTrue(
                torch.allclose(
                    inner_products,
                    torch.zeros_like(inner_products),
                    atol=2e-4,
                    rtol=2e-4,
                )
            )

    def test_nearly_dependent_but_resolvable_outputs_remain_orthogonal(self) -> None:
        expert_outputs = torch.tensor(
            [[
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1e-3, 0.0, 0.0],
                [1.0, 1e-3, 1e-3, 0.0],
            ]]
        )
        selected_experts = torch.tensor([2], dtype=torch.long)

        orthogonalized = omoe._gram_schmidt_expert_outputs(
            expert_outputs,
            selected_experts,
        )
        gram = torch.matmul(
            orthogonalized,
            orthogonalized.transpose(1, 2),
        )
        off_diagonal = gram - torch.diag_embed(
            torch.diagonal(gram, dim1=1, dim2=2)
        )
        self.assertLess(
            float(off_diagonal.abs().max().item()),
            1e-3,
        )

    def test_gradient_reaches_all_experts_for_well_conditioned_inputs(self) -> None:
        expert_outputs = torch.randn(
            6,
            4,
            12,
            requires_grad=True,
        )
        # All tokens select expert 0, which is deliberately processed last.
        selected_experts = torch.zeros(6, dtype=torch.long)
        probe = torch.randn(6, 12)

        orthogonalized = omoe._gram_schmidt_expert_outputs(
            expert_outputs,
            selected_experts,
        )
        loss = (
            orthogonalized[
                torch.arange(6),
                selected_experts,
            ] * probe
        ).sum()
        loss.backward()

        self.assertIsNotNone(expert_outputs.grad)
        per_expert_gradient = expert_outputs.grad.abs().sum(dim=(0, 2))
        self.assertTrue(torch.isfinite(per_expert_gradient).all())
        self.assertTrue(torch.all(per_expert_gradient > 1e-7))

    def test_nearly_collinear_outputs_have_finite_bounded_gradients(self) -> None:
        expert_outputs = torch.tensor(
            [[
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1e-8, 0.0, 0.0],
                [1.0, 2e-8, 0.0, 0.0],
            ]],
            requires_grad=True,
        )
        selected_experts = torch.tensor([2], dtype=torch.long)

        orthogonalized, diagnostics = omoe._gram_schmidt_expert_outputs(
            expert_outputs,
            selected_experts,
            return_diagnostics=True,
        )

        self.assertTrue(torch.isfinite(orthogonalized).all())
        self.assertGreater(
            int(diagnostics["orthogonalization_regularized_vectors"].item()),
            0,
        )
        self.assertEqual(diagnostics["orthogonalization_vector_count"], 3)
        self.assertTrue(
            torch.isfinite(
                diagnostics["orthogonalization_residual_norm_sum"]
            )
        )
        self.assertTrue(
            torch.isfinite(
                diagnostics["orthogonalization_relative_residual_norm_sum"]
            )
        )

        probe = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        loss = (orthogonalized[:, 2, :] * probe).sum()
        loss.backward()

        self.assertIsNotNone(expert_outputs.grad)
        self.assertTrue(torch.isfinite(expert_outputs.grad).all())
        self.assertLessEqual(
            float(expert_outputs.grad.abs().max().item()),
            1000.0,
        )
        self.assertGreater(
            float(expert_outputs.grad[:, 2, :].abs().sum().item()),
            0.0,
        )

    def test_exactly_collinear_outputs_do_not_produce_nan_or_inf(self) -> None:
        expert_outputs = torch.ones(
            2,
            3,
            5,
            requires_grad=True,
        )
        selected_experts = torch.tensor([1, 2], dtype=torch.long)

        orthogonalized = omoe._gram_schmidt_expert_outputs(
            expert_outputs,
            selected_experts,
        )
        self.assertTrue(torch.isfinite(orthogonalized).all())

        probe = torch.randn(2, 5)
        loss = (
            orthogonalized[
                torch.arange(2),
                selected_experts,
            ] * probe
        ).sum()
        loss.backward()

        self.assertIsNotNone(expert_outputs.grad)
        self.assertTrue(torch.isfinite(expert_outputs.grad).all())
        self.assertGreater(
            float(expert_outputs.grad.abs().sum().item()),
            0.0,
        )

    def test_half_precision_inputs_are_computed_safely_in_fp32(self) -> None:
        expert_outputs = torch.randn(2, 3, 8, dtype=torch.float16)
        selected_experts = torch.tensor([0, 2], dtype=torch.long)

        orthogonalized = omoe._gram_schmidt_expert_outputs(
            expert_outputs,
            selected_experts,
        )

        self.assertEqual(orthogonalized.dtype, torch.float16)
        self.assertTrue(torch.isfinite(orthogonalized).all())

    def test_invalid_shapes_and_expert_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            omoe._gram_schmidt_expert_outputs(
                torch.randn(2, 3),
                torch.tensor([0, 1]),
            )

        with self.assertRaisesRegex(ValueError, "num_experts"):
            omoe._gram_schmidt_expert_outputs(
                torch.randn(2, 4, 3),
                torch.tensor([0, 1]),
            )

        with self.assertRaisesRegex(ValueError, "selected_experts"):
            omoe._gram_schmidt_expert_outputs(
                torch.randn(2, 3, 4),
                torch.tensor([0]),
            )

        with self.assertRaisesRegex(TypeError, "torch.long"):
            omoe._gram_schmidt_expert_outputs(
                torch.randn(2, 3, 4),
                torch.tensor([0, 1], dtype=torch.int32),
            )

        with self.assertRaisesRegex(ValueError, "取值"):
            omoe._gram_schmidt_expert_outputs(
                torch.randn(2, 3, 4),
                torch.tensor([0, 3], dtype=torch.long),
            )


class OMoESwitchMoETests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def _make_moe(
        self,
        *,
        capacity_factor: float = 0.0,
        token_dropping: bool = False,
        aux_loss_coef: float = 0.0,
    ) -> omoe.OMoESwitchMoE:
        moe = base.SwitchMoE(
            hidden_dim=4,
            expert_hidden_dim=8,
            num_experts=3,
            dropout=0.0,
            collect_router_statistics=True,
            aux_loss_coef=aux_loss_coef,
            capacity_factor=capacity_factor,
            token_dropping=token_dropping,
        )
        moe.__class__ = omoe.OMoESwitchMoE

        # Force all tokens to prefer expert 0. Capacity tests can then verify
        # the original fallback assignment behavior deterministically.
        with torch.no_grad():
            moe.router.projection.weight.zero_()
            moe.router.projection.bias.copy_(
                torch.tensor([3.0, 2.0, 1.0])
            )
        return moe

    def test_forward_keeps_top1_and_propagates_gradients_to_all_experts(self) -> None:
        moe = self._make_moe(aux_loss_coef=0.01)
        moe.train()
        inputs = torch.randn(2, 5, 4)
        outputs, stats, aux_loss = moe(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 5, 4))
        self.assertIsNotNone(aux_loss)
        self.assertTrue(torch.isfinite(aux_loss))
        self.assertEqual(stats["tokens_per_expert"].tolist(), [10, 0, 0])
        self.assertEqual(stats["orthogonalization_vector_count"], 30)
        self.assertIn("orthogonalization_regularized_vectors", stats)

        probe = torch.randn_like(outputs)
        (outputs * probe).sum().backward()

        for expert_id, expert in enumerate(moe.experts.values()):
            with self.subTest(expert_id=expert_id):
                gradients = [
                    parameter.grad
                    for parameter in expert.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )
                self.assertGreater(
                    sum(float(gradient.abs().sum().item()) for gradient in gradients),
                    0.0,
                )

    def test_capacity_rerouting_is_preserved(self) -> None:
        moe = self._make_moe(capacity_factor=1.0)
        inputs = torch.randn(1, 6, 4)

        outputs, stats, _ = moe(inputs)

        self.assertEqual(tuple(outputs.shape), (1, 6, 4))
        self.assertEqual(stats["tokens_per_expert"].tolist(), [6, 0, 0])
        self.assertEqual(
            stats["processed_tokens_per_expert"].tolist(),
            [2, 2, 2],
        )
        self.assertEqual(stats["rerouted_tokens"], 4)
        self.assertEqual(stats["dropped_tokens"], 0)

    def test_token_dropping_is_preserved(self) -> None:
        moe = self._make_moe(
            capacity_factor=0.5,
            token_dropping=True,
        )
        inputs = torch.randn(1, 8, 4)

        outputs, stats, _ = moe(inputs)

        self.assertEqual(tuple(outputs.shape), (1, 8, 4))
        self.assertEqual(
            stats["processed_tokens_per_expert"].tolist(),
            [2, 2, 2],
        )
        self.assertEqual(stats["dropped_tokens"], 2)
        self.assertTrue(torch.equal(outputs[:, 6:, :], torch.zeros_like(outputs[:, 6:, :])))


class OMoEIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.cfg = {
            "dataset": "cifar10",
            "num_experts": 3,
            "model_cfg": {
                "transformer_layers": 2,
                "hidden_dim": 16,
                "attention_heads": 4,
                "patch_size": 16,
                "expert_hidden_dim": 32,
                "moe_block_indices": [1, 2],
                "routing": "top1",
                "aux_loss_coef": 0.01,
                "capacity_factor": 0.0,
                "token_dropping": False,
                "router_statistics": True,
                "dropout": 0.0,
            },
        }

    def test_injection_preserves_state_dict_and_federated_parameter_groups(self) -> None:
        model = base.build_switch_vit_model(self.cfg)
        state_keys_before = list(model.state_dict().keys())
        parameter_ids_before = [id(parameter) for parameter in model.parameters()]
        groups_before = base.build_param_groups(
            model=model,
            expected_num_experts=3,
            strict=True,
        )

        converted_model = omoe._inject_omoe_forward(model)

        self.assertIs(converted_model, model)
        self.assertEqual(state_keys_before, list(model.state_dict().keys()))
        self.assertEqual(
            parameter_ids_before,
            [id(parameter) for parameter in model.parameters()],
        )

        groups_after = base.build_param_groups(
            model=model,
            expected_num_experts=3,
            strict=True,
        )
        self.assertEqual(groups_before.all, groups_after.all)
        self.assertEqual(groups_before.non_expert, groups_after.non_expert)
        self.assertEqual(groups_before.expert, groups_after.expert)
        self.assertEqual(
            groups_before.expert_by_layer_and_id,
            groups_after.expert_by_layer_and_id,
        )

        for block in model.blocks:
            if block.use_moe:
                self.assertIsInstance(block.mlp, omoe.OMoESwitchMoE)

    def test_omoe_model_runs_in_train_and_eval_modes(self) -> None:
        model = omoe.build_omoe_switch_vit_model(self.cfg)
        images = torch.randn(2, 3, 32, 32)
        targets = torch.tensor([1, 4])

        model.train()
        train_output = model(images)
        loss = base.compute_model_loss(
            train_output,
            targets,
            torch.nn.CrossEntropyLoss(),
        )
        self.assertTrue(torch.isfinite(loss.total_loss))
        loss.total_loss.backward()

        model.eval()
        with torch.no_grad():
            eval_output = model(images)

        self.assertEqual(tuple(eval_output.logits.shape), (2, 10))
        self.assertEqual(
            set(eval_output.moe_stats),
            {"blocks.0.mlp", "blocks.1.mlp"},
        )

    def test_expert_aggregator_uses_client_equal_weights(self) -> None:
        aggregator = omoe.build_expert_aggregator({})
        updates = [
            base.ClientUpdate(
                client_id=0,
                round_id=1,
                num_samples=100,
                model_delta={"dummy": torch.zeros(1)},
            ),
            base.ClientUpdate(
                client_id=1,
                round_id=1,
                num_samples=900,
                model_delta={"dummy": torch.zeros(1)},
            ),
        ]

        self.assertEqual(aggregator.method_name, "omoe_uniform")
        self.assertEqual(
            aggregator.compute_weights(updates),
            {0: 0.5, 1: 0.5},
        )


if __name__ == "__main__":
    unittest.main()

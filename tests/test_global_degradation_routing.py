"""Regression tests for paper-aligned global degradation-conditioned routing."""

import inspect

import torch

from net.fmoe import RoutingFunction


def _router(*, training: bool = False) -> RoutingFunction:
    router = RoutingFunction(
        condition_dim=3,
        num_experts=4,
        complexity=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        enable_routing_logging=True,
    )
    router.train(training)
    return router


def test_router_accepts_only_global_z_deg_and_matches_linear_projection():
    router = _router()
    router.eval()
    with torch.no_grad():
        router.condition_projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, -1.0, -1.0],
                ]
            )
        )

    z_deg = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    stage_bias = torch.tensor([0.0, 0.1, -0.2, 0.0])

    gates, _ = router(z_deg, stage_bias=stage_bias)
    logits = router.condition_projection(z_deg) + stage_bias
    probabilities = torch.softmax(logits, dim=-1)
    expected = torch.zeros_like(probabilities).scatter_(
        1, probabilities.argmax(dim=-1, keepdim=True), probabilities.max(dim=-1, keepdim=True).values
    )

    torch.testing.assert_close(gates, expected)
    assert tuple(inspect.signature(router.forward).parameters) == (
        "z_deg",
        "stage_bias",
    )
    assert not hasattr(router, "feature_gate")


def test_global_condition_projection_receives_restoration_gradients():
    router = _router()
    router.eval()
    with torch.no_grad():
        router.condition_projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, -1.0, -1.0],
                ]
            )
        )

    z_deg = torch.tensor([[1.0, 0.7, 0.4]], requires_grad=True)
    gates, _ = router(z_deg, stage_bias=torch.zeros(4))
    gates.max(dim=-1).values.sum().backward()

    assert z_deg.grad is not None and torch.isfinite(z_deg.grad).all()
    assert z_deg.grad.abs().sum() > 0
    weight_grad = router.condition_projection.weight.grad
    assert weight_grad is not None and torch.isfinite(weight_grad).all()
    assert weight_grad.abs().sum() > 0


def test_router_is_deterministic_in_eval_and_noisy_during_training():
    router = _router()
    with torch.no_grad():
        router.condition_projection.weight.zero_()
    z_deg = torch.zeros(128, 3)
    stage_bias = torch.zeros(4)

    router.eval()
    eval_first, _ = router(z_deg, stage_bias=stage_bias)
    eval_second, _ = router(z_deg, stage_bias=stage_bias)
    torch.testing.assert_close(eval_first, eval_second)

    router.train()
    torch.manual_seed(11)
    train_first, _ = router(z_deg, stage_bias=stage_bias)
    torch.manual_seed(29)
    train_second, _ = router(z_deg, stage_bias=stage_bias)
    assert not torch.equal(train_first.argmax(dim=-1), train_second.argmax(dim=-1))


def test_each_routed_block_has_an_independent_condition_projection():
    first = _router()
    second = _router()

    assert first.condition_projection is not second.condition_projection
    assert (
        first.condition_projection.weight.data_ptr()
        != second.condition_projection.weight.data_ptr()
    )

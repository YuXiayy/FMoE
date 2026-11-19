from net.model_factory import fmoe_kwargs
from options import train_options


def test_training_and_inference_share_one_complete_model_contract():
    options = train_options(["--config", "configs/fmoe_train.yaml"])

    training = fmoe_kwargs(options)
    inference = fmoe_kwargs(options, inference=True)

    assert set(training) == set(inference)
    assert training["routing_balance_ema_decay"] == 0.99
    assert inference["routing_balance_ema_decay"] == 0.99
    assert training["enable_routing_logging"] is True
    assert inference["enable_routing_logging"] is False
    assert training["profile_expert_flops"] is False
    assert inference["profile_expert_flops"] is False
    assert training["include_training_head"] is True
    assert inference["include_training_head"] is False

    for key in set(training) - {
        "enable_routing_logging",
        "profile_expert_flops",
        "include_training_head",
    }:
        assert training[key] == inference[key]

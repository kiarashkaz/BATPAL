from BATPAL.runner.on_policy_ma_runner_advt_with_belief import OnPolicyMARunnerAdvtBelief
from BATPAL.runner.on_policy_multi_type_runner import OnPolicyMultiTypeRunner
from BATPAL.runner.fixed_types_adv_runner import FixedTypesAdvRunner

RUNNER_REGISTRY = {
    "mappo_fixed_benign": OnPolicyMARunnerAdvtBelief,
    "mappo_no_adv": OnPolicyMARunnerAdvtBelief,
    "mappo_advt_ec_belief": OnPolicyMARunnerAdvtBelief,
    "mappo_multi_type_belief": OnPolicyMultiTypeRunner,
    "fixed_types_adv": FixedTypesAdvRunner,
}

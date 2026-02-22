from BATPAL.runner.on_policy_ma_runner_advt_with_belief import OnPolicyMARunnerAdvtBelief
from BATPAL.runner.on_policy_multi_type_runner import OnPolicyMultiTypeRunner
#from BATPAL.runner.gen_maxmin_runner import GenMaxminRunner
#from BATPAL.runner.rap_runner import RAPRunner
#from BATPAL.runner.fixed_adv_runner import FixedAdvRunner
from BATPAL.runner.fixed_types_adv_runner import FixedTypesAdvRunner

RUNNER_REGISTRY = {
    "mappo_advt_belief": OnPolicyMARunnerAdvtBelief,
    "mappo_traitor_belief": OnPolicyMARunnerAdvtBelief,
    "mappo_fixed_benign": OnPolicyMARunnerAdvtBelief,
    "mappo_no_adv": OnPolicyMARunnerAdvtBelief,
    "mappo_advt_ec_belief": OnPolicyMARunnerAdvtBelief,
    "mappo_multi_type_belief": OnPolicyMultiTypeRunner,
#    "fixed_adv": FixedAdvRunner,
    "fixed_types_adv": FixedTypesAdvRunner,
}

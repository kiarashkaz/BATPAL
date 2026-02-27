from BATPAL.algo.mappo_traitor_belief import MAPPOTraitorBelief
from BATPAL.algo.mappo_advt_with_belief import MAPPOAdvtBelief
from BATPAL.algo.mappo_advt_EC_with_belief import MAPPOAdvtECBelief
from BATPAL.algo.mappo_multi_type_with_belief import MAPPOMultiTypeBelief

ALGO_REGISTRY = {
    "mappo_advt_belief": MAPPOAdvtBelief,
    "mappo_traitor_belief": MAPPOTraitorBelief,
    "mappo_fixed_benign": MAPPOTraitorBelief,
    "mappo_no_adv": MAPPOAdvtBelief,
    "mappo_advt_ec_belief": MAPPOAdvtECBelief,
    "mappo_multi_type_belief": MAPPOMultiTypeBelief,
    "fixed_types_adv": MAPPOMultiTypeBelief,
}

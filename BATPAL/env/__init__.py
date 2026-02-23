import os
import socket
import sys
from absl import flags
from BATPAL.env.smac.logger import SMACLogger, BaseLogger
from BATPAL.env.smacv2.logger import SMACv2Logger
#from BATPAL.env.mamujoco.logger import MAMuJoCoLogger
from BATPAL.env.pettingzoo_mpe.logger import PettingZooMPELogger
from BATPAL.env.gym.logger import GYMLogger
#from BATPAL.env.football.logger import FootballLogger
#from BATPAL.env.dexhands.logger import DexHandsLogger
from BATPAL.env.toy_example.logger import ToyLogger
from BATPAL.env.ma_envs.rendezvous_logger import RendezvousLogger
from BATPAL.env.ma_envs.pursuit_logger import PursuitLogger
from BATPAL.env.ma_envs.navigation_logger import NavigationLogger
from BATPAL.env.ma_envs.cover_logger import CoverLogger

FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "smac_traitor": SMACLogger,
#    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
#    "football": FootballLogger,
#    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "toy": ToyLogger,
    "lbforaging": ToyLogger,
    "rware": ToyLogger,
    "rendezvous": ToyLogger,
    "pursuit": ToyLogger,
    "navigation": ToyLogger,
    "cover": ToyLogger
}

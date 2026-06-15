# traffic_task_cfg.py
from dataclasses import dataclass, field
from cfg.observations_cfg import TrafficObservationsCfg
from cfg.rewards_cfg import TrafficRewardsCfg
from cfg.terminations_cfg import TrafficTerminationsCfg

@dataclass
class TrafficTaskCfg:
    observations: TrafficObservationsCfg = field(default_factory=TrafficObservationsCfg)
    rewards: TrafficRewardsCfg = field(default_factory=TrafficRewardsCfg)
    terminations: TrafficTerminationsCfg = field(default_factory=TrafficTerminationsCfg)
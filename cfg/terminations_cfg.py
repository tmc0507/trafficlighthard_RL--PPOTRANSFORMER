# cfg/traffic_terminations_cfg.py
from dataclasses import dataclass, field

@dataclass
class TrafficTerminationsCfg:
    terminate: list = field(default_factory=lambda: [])
    truncate: list = field(default_factory=lambda: [
        lambda env: env.frame_count > 2000
    ])
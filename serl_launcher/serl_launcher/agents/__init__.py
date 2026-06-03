from .continuous.bc import BCAgent
from .continuous.sac import SACAgent
# Optional hybrid SAC agents are not present in this fork.
try:
    from .continuous.sac_hybrid_single import SACAgentHybridSingleArm
except ImportError:
    SACAgentHybridSingleArm = None

try:
    from .continuous.sac_hybrid_dual import SACAgentHybridDualArm
except ImportError:
    SACAgentHybridDualArm = None

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
}

if SACAgentHybridSingleArm is not None:
    agents["sac_hybrid_single"] = SACAgentHybridSingleArm
if SACAgentHybridDualArm is not None:
    agents["sac_hybrid_dual"] = SACAgentHybridDualArm

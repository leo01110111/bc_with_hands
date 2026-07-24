"""wuji_bc — behavior cloning with flow-matching policies.

Subpackages:
- actors: policy networks (flow-matching vector fields, MLP and transformer)
- agents: training agents and loss helpers
- networks: reusable building blocks (MLP, attention, time embedding)
- utils: datasets, logging, evaluation, config loading

Import submodules directly to avoid pulling in heavy optional dependencies:
    from wuji_bc.actors.actors import ActorVectorField, ActorVectorFieldTF
    from wuji_bc.agents.modules.bc_helper import compute_flow_bc_loss
"""

__all__ = [
    'actors',
    'agents',
    'networks',
    'utils',
]

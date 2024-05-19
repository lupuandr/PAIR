from pathlib import Path

from .agent import Agent
from .celery_agent import CeleryAgent
from .guardian_agent import LlamaGuardAgent

# default globals
agent_dict = {}
gen_params = {}
base_path = None
batch_size = 32
rm_batch_size = 32


def init_agent_dict(args):
    # update the following global variables
    global agent_dict, base_path, batch_size, rm_batch_size, gen_params

    agent_dict = {}
    if args.model_base_path is not None:
        base_path = Path(args.model_base_path)
    batch_size = args.batch_size
    if "eval_type" in args and args.eval_type == "reward_model":
        rm_batch_size = args.rm_batch_size
    gen_params = dict(args.gen_params)


def get_agent(name):
    # update the following global variables
    global agent_dict, base_path

    if name not in agent_dict:
        if name.endswith("-celery"):
            agent_dict[name] = CeleryAgent(name, gen_params)
        elif name.endswith("-guard"):
            agent_dict[name] = LlamaGuardAgent(name, gen_params)
        else:
            raise ValueError(f"Unknown agent name: {name}")
    return agent_dict[name]


__all__ = [
    "Agent",
    "CeleryAgent",
]

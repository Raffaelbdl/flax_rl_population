import numpy as np

from rl.base import Deployed
from rl.types import ParallelEnv


def multi_agents_evaluate(
    agents: list[Deployed], env: ParallelEnv, n_episodes: int
) -> np.ndarray:
    """Evaluates a list of agents in two players environments

    Environment is a simple PettingZoo ParallelEnv, no batch
    """
    returns_matrix = np.zeros((len(agents), len(agents)))

    def roll_episode(agent_pair: list[Deployed], env: ParallelEnv):
        obs, info = env.reset()
        terminated = False
        episode_return = 0.0
        while not terminated:
            action = {
                agent_id: agent_pair[i].select_action(o)[0]
                for i, (agent_id, o) in enumerate(obs.items())
            }
            obs, reward, done, trunc, info = env.step(action)
            episode_return += sum(reward.values())

            terminated = any(done.values()) or any(trunc.values())

        return episode_return

    for i in range(len(agents)):
        for j in range(len(agents)):
            for e in range(n_episodes):
                returns_matrix[i][j] += roll_episode([agents[i], agents[j]], env)

    returns_matrix /= n_episodes
    return returns_matrix

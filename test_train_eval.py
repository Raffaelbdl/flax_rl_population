from evals.eval_envs import CartPoleParallel, make_cartpole_parallel_vector
import ml_collections
from rl_population.algos import population_ppo
from evals.eval_callbacks import TimeCallback

from rl_population.evaluate import multi_agents_evaluate

import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"


def main():
    TASK_ID = "CartPole-Parallel"
    _, env_config = make_cartpole_parallel_vector(2)
    envs = [make_cartpole_parallel_vector(2)[0] for _ in range(5)]
    config = ml_collections.ConfigDict(
        {
            "env_config": env_config,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "_lambda": 0.95,
            "normalize": True,
            "max_buffer_size": 64,
            "batch_size": 32,
            "num_epochs": 4,
            "learning_rate_annealing": True,
            "max_grad_norm": 0.5,
            "n_env_steps": 10**4 // 4,
            "shared_encoder": False,
            "save_frequency": -1,
            "population_size": len(envs),
            "jsd_coef": 0.01,
        }
    )

    population = population_ppo.PopulationPPO(
        0, config, tabulate=True, run_name="12-13-2023_13-52-25"
    )
    last_step = population.restore()

    deployed = population.to_list_of_deployed(False)
    returns_matrix = multi_agents_evaluate(deployed, CartPoleParallel(), 2)
    print(returns_matrix)

    population.resume(envs, config["n_env_steps"] * 2, [])

    deployed = population.to_list_of_deployed(False)
    returns_matrix = multi_agents_evaluate(deployed, CartPoleParallel(), 2)
    print(returns_matrix)


if __name__ == "__main__":
    main()

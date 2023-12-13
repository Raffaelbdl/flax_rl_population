from typing import Any, Callable

from flax.training.train_state import TrainState
import ml_collections
import jax

from rl.algos.general_fns import explore_general_factory
from rl.base import Base
from rl.base import Deployed


class Population(Base):
    def to_list_of_deployed(self, batched: bool) -> list[Deployed]:
        return [
            Deployed(
                i,
                self.state.params[i],
                explore_general_factory(
                    self.explore_factory(self.state, self.config),
                    batched=batched,
                    parallel=False,
                ),
            )
            for i in range(len(self.state.params))
        ]

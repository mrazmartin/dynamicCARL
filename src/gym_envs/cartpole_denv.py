from __future__ import annotations

from typing import Optional

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv


class CARLCartPole(CARLGymnasiumEnv):
    env_name: str = "CartPole-v1"
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, contexts=None, render_mode="rgb_array", **kwargs):
        self.render_mode = render_mode  # ✅ set before gym.make()
        super().__init__(contexts=contexts, render_mode=render_mode, **kwargs)

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=0.1, upper=1000.0, default_value=9.8
            ),
            "masscart": UniformFloatContextFeature(
                "masscart", lower=0.1, upper=10, default_value=1.0
            ),
            "masspole": UniformFloatContextFeature(
                "masspole", lower=0.01, upper=1, default_value=0.1
            ),
            "length": UniformFloatContextFeature(
                "length", lower=0.05, upper=5, default_value=0.5
            ),
            "force_mag": UniformFloatContextFeature(
                "force_mag", lower=1, upper=100, default_value=10.0
            ),
            "tau": UniformFloatContextFeature(
                "tau", lower=0.002, upper=0.2, default_value=0.02
            ),
            # the following we replaced np.ing with np.power to avoid normalization issues
            "initial_state_lower": UniformFloatContextFeature(
                "initial_state_lower", lower=-np.power(2, 31), upper=np.power(2, 31), default_value=-0.1 
            ),
            "initial_state_upper": UniformFloatContextFeature(
                "initial_state_upper", lower=-np.power(2, 31), upper=np.power(2, 31), default_value=0.1
            ),
            # OUR CUSTOM PAYLOAD CONTEXT
            "PAYLOAD_X": UniformFloatContextFeature(
                "PAYLOAD_X", lower=-5.0, upper=5.0, default_value=0.0
            ),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        self.env.unwrapped.state = self.env.np_random.uniform(
            low=self.context.get(
                "initial_state_lower",
                self.get_context_features()["initial_state_lower"].default_value,
            ),
            high=self.context.get(
                "initial_state_upper",
                self.get_context_features()["initial_state_upper"].default_value,
            ),
            size=(4,),
        )
        state = np.array(self.env.unwrapped.state, dtype=np.float32)
        info = {}
        state = self._add_context_to_state(state)
        info["context_id"] = self.context_id
        return state, info

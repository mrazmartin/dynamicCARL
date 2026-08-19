import numpy as np
# wrapper for limiting the contextual features --> dont include observations of static context
import gymnasium as gym
from gymnasium import spaces

class PartialContextWrapper(gym.ObservationWrapper):
    """
    Wrapper for partially observing the context in a Gym environment.
    """
    def __init__(self, env, selected_context_keys, flatten_obs=True):
        super().__init__(env)
        self.selected_keys = list(selected_context_keys or [])
        self.flatten_obs = flatten_obs

        assert isinstance(env.observation_space, spaces.Dict) and "obs" in env.observation_space.spaces, \
            "PartialContextWrapper expects a Dict observation space with key 'obs'"
        if self.selected_keys:
            assert "context" in env.observation_space.spaces, \
                "selected_context_keys provided but base env has no 'context' space"

        if self.flatten_obs:
            obs_space: spaces.Box = env.observation_space["obs"]
            obs_low  = np.asarray(obs_space.low, dtype=np.float32).reshape(-1)
            obs_high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)

            if self.selected_keys:
                ctx_space: spaces.Dict = env.observation_space["context"]
                # concatenate ALL dims of each key, not just [0], to support vector contexts
                ctx_lows  = np.concatenate([np.asarray(ctx_space.spaces[k].low,  dtype=np.float32).reshape(-1) for k in self.selected_keys])
                ctx_highs = np.concatenate([np.asarray(ctx_space.spaces[k].high, dtype=np.float32).reshape(-1) for k in self.selected_keys])
                low  = np.concatenate([obs_low,  ctx_lows])
                high = np.concatenate([obs_high, ctx_highs])
            else:
                low, high = obs_low, obs_high

            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            obs_space = env.observation_space["obs"]
            if not self.selected_keys:
                self.observation_space = obs_space
            else:
                # Flatten all selected context keys into a single Box
                ctx_space: spaces.Dict = env.observation_space["context"]
                ctx_lows  = np.concatenate([np.asarray(ctx_space.spaces[k].low,  dtype=np.float32).reshape(-1) for k in self.selected_keys])
                ctx_highs = np.concatenate([np.asarray(ctx_space.spaces[k].high, dtype=np.float32).reshape(-1) for k in self.selected_keys])
                self.observation_space = spaces.Dict({
                    "obs": obs_space,
                    "context": spaces.Box(low=ctx_lows, high=ctx_highs, dtype=np.float32)
                })

    def observation(self, obs_dict):
        if self.flatten_obs:
            # --- ORIGINAL LOGIC ---
            if isinstance(obs_dict, np.ndarray):
                return obs_dict.astype(np.float32).reshape(-1)
            
            if isinstance(obs_dict, dict) and "obs" in obs_dict:
                obs = np.asarray(obs_dict["obs"], dtype=np.float32).reshape(-1)
            else:
                return np.asarray(obs_dict, dtype=np.float32).reshape(-1)

            if not self.selected_keys:
                return obs

            if "context" not in obs_dict:
                raise RuntimeError("PartialContextWrapper received an observation without 'context'.")

            if not obs_dict["context"]:
                context = np.empty((0,), dtype=np.float32)
            else:
                context = np.asarray([obs_dict["context"][k] for k in self.selected_keys], dtype=np.float32).reshape(-1)
                
            return np.concatenate([obs, context], dtype=np.float32)
        else:
            # --- NEW DICT-PRESERVING LOGIC ---
            obs = obs_dict["obs"]
            if not self.selected_keys:
                return obs
            
            context = np.asarray([obs_dict["context"][k] for k in self.selected_keys], dtype=np.float32).reshape(-1)
            return {"obs": obs, "context": context}

    def get_sc_ep(self):
        return self.env.get_sc_ep()

    def set_sc_ep(self, value: int):
        return self.env.set_sc_ep(value)
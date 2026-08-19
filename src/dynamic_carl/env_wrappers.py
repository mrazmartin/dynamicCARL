from gymnasium.spaces import Dict as SpaceDict
from gymnasium import spaces
from carl.context.context_space import NumericalContextFeature
import numpy as np
import gymnasium as gym

class GymDynamicContextCarlWrapper(gym.Wrapper):
    """
    Wraps a CARLGymnasiumEnv to support context features that change dynamically
    during an episode (e.g. a pole length that drifts, a payload that shifts).

    CARL handles per-episode context sampling and static context injection.
    This wrapper adds the step-level scheduling layer on top.

    -----------------------------------------------------------------------
    Key concepts
    -----------------------------------------------------------------------

    observe_context_mode  -- controls what the AGENT sees in obs["context"]:

        "live"      The agent observes the TRUE current context value every step.
                    Use this when the agent should adapt to the changing context
                    in real time (context-aware policy).
                    REQUIREMENT: ctx_getters must be provided for every key in
                    feature_update_fns.

        "initial"   The agent observes the context value from the START of the
                    episode (i.e. the CARL-sampled value), not the live one.
                    Use this to study whether an agent that knows its starting
                    context but not its drift can still generalise.
                    REQUIREMENT: ctx_getters are needed at reset() to take the
                    initial snapshot; provide them even for "initial" mode.

        "none"      The agent receives NO context in its observation. The
                    obs["context"] dict is emptied at runtime and the context
                    space is built with zero keys.
                    REQUIREMENT: ctx_to_observe MUST be [] (empty list/set).
                    Passing non-empty ctx_to_observe with mode "none" will raise
                    at construction time to prevent a silent runtime crash.

    ctx_to_observe  -- which context keys appear in obs["context"].

        Pass the exact same list to BOTH:
          - the env factory  (obs_context_features=ctx_to_observe)
          - this wrapper     (ctx_to_observe=ctx_to_observe)
        so that CARL's observation space and this wrapper's space stay in sync.

        Rules by mode:
          "live" / "initial"  ->  list the keys the agent should see, e.g. ["length"]
          "none"              ->  must be [] (empty)
          default (None)      ->  all static CARL context keys + dynamic keys

    feature_update_fns  -- dict mapping context key -> scheduler object.

        Each scheduler must:
          - implement reset(env, episode_id, ctx, worker_id)
          - implement __call__(env, step_id, ctx_dict, verbose) -> new_value
          - have its key registered in the CARL env's get_context_features()

        Schedulers are called BEFORE env.step() on every step after the first.
        Step 0 of each episode always uses the reset context unchanged.

    ctx_getters / ctx_setters  -- fn(env) -> value  /  fn(env, value).

        These teach the wrapper how to read and write each context attribute on
        the underlying env. If omitted, the wrapper falls back to getattr/setattr
        on env.unwrapped, which works for simple attribute-backed contexts.

    -----------------------------------------------------------------------
    Minimal usage example
    -----------------------------------------------------------------------

        from dynamic_crl.src.dynamic_carl.env_wrappers import GymDynamicContextCarlWrapper
        from dynamic_crl.src.dynamic_carl.gym_context_updates import make_sinusoidal

        carl_env = CARLCartPole(contexts={0: {"length": 0.5}}, obs_context_features=["length"])

        updater = make_sinusoidal("length", amplitude=0.2, period=100, min_val=0.05, max_val=5.0)

        env = GymDynamicContextCarlWrapper(
            env=carl_env,
            feature_update_fns={"length": updater},
            ctx_getters={"length": lambda e: float(e.unwrapped.length)},
            ctx_setters={"length": lambda e, v: setattr(e.unwrapped, "length", float(v))},
            ctx_to_observe=["length"],       # must match obs_context_features above
            observe_context_mode="live",     # agent sees the live drifting value
        )
    -----------------------------------------------------------------------
    """
    def __init__(
        self,
        env,
        feature_update_fns,
        ctx_getters=None,
        ctx_setters=None,
        ctx_to_observe: set[str] = None,
        worker_id=0,
        observe_context_mode: str = "live",
        mutate_obs_space: bool = False,
        dctx_features_definitions=None,
        verbose=False,
    ):
        super().__init__(env)
        self.worker_id = worker_id

        # dynamic context handling
        self.feature_update_fns = feature_update_fns or {}
        self.dctx_features_definitions = dctx_features_definitions or {}
        # dynamic ctx manipulation
        self.ctx_getters = ctx_getters or {}
        self.ctx_setters = ctx_setters or {}

        # self.ctx_to_change = list(ctx_to_change) if ctx_to_change else []

        self.ctx_to_observe = set(ctx_to_observe) if ctx_to_observe is not None else None
        self.verbose = verbose

        self._pending_seed = None # this is to be set upon first reset of the env to enforce seeding

        if observe_context_mode not in {"live", "initial", "none"}:
            raise ValueError(
                f"observe_context_mode must be one of {{'live','initial','none'}}, got {observe_context_mode}"
            )
        self.observe_context_mode = observe_context_mode
        
        # check whether each dynamic ctx has a getter and setter
        self._check_args() # validate getters/setters for dynamic contexts


        # Validate feature keys (OUTDATED - some dynamic contexts may not be in env.contexts)
        # context_keys = (
        #    env.contexts[0].keys()
        #    if hasattr(env, "contexts")
        #    else set(self.ctx_getters.keys()) | set(self.ctx_setters.keys())
        # )
        # for feature in self.feature_update_fns:
        #    if feature not in context_keys:
        #        raise ValueError(f"Feature '{feature}' not found in context keys: {context_keys}")

        # Optionally mutate observation_space if context will never be exposed
        # NOTE: this is important as empty 'context' dicts can cause issues can break flatten wrappers
        if mutate_obs_space and self.observe_context_mode == "none":
            # if we don't want to observe context at all, remove it from the space
            if isinstance(self.env.observation_space, SpaceDict) and \
                    "context" in self.env.observation_space.spaces:
                new_spaces = dict(self.env.observation_space.spaces)
                new_spaces.pop("context", None)
                self.observation_space = SpaceDict(new_spaces)
            else:
                self.observation_space = env.observation_space
                if self.verbose:
                    print("[Wrapper] Cannot remove 'context' from observation_space; leaving as is.")
        else:
            # NOTE: if we don't remove the context key from observations, make sure dynamic ctxs are included
            # also make sure that the space won't include non-observable or default contexts
            self._update_obs_space()

        self.episode_count = -1
        self.step_count = 0
        self._ctx_reset_snapshot = {}

    def _dctx_feature_to_gym_space(self, cf_name):
        """
        Convert a dynamic context feature name to a Gym space.
        """
        context_feature = self.dctx_features_definitions[cf_name]
        if isinstance(context_feature, NumericalContextFeature):
            return spaces.Box(
                low=context_feature.lower, high=context_feature.upper
            )
        else:
            return spaces.Discrete(
                len(context_feature.choices)
            )

    def _update_obs_space(self):
        """
        Update the observation space to include the dynamic contexts.
        Also set self.ctx_to_observe if it was None.
        """
        
        # old/default observation of the CarlGymEnv space -> env.observation_space
        
        # 0. get the default observation space from the env
        base_obs_space = self.env.observation_space['obs']
        default_ctx_space = self.env.observation_space['context']

        # 1. get the keys of the context that should be observable
        # i) if no keys are specified, we have to take the whole context space + dynamic_ctxs
        if self.ctx_to_observe is None:
            static_ctx_keys = self.env.observation_space['context'].keys()
            dynamic_ctx_keys = set(self.feature_update_fns.keys())

            # union of both sets
            self.ctx_to_observe = static_ctx_keys | dynamic_ctx_keys

        # ii) if keys are specified, we don't have to do anything
        else:
            pass

        # 2. build the new context space for each self.ctx_to_observe key
        new_ctx_obs_spaces = {}
        for key in self.ctx_to_observe:
            # if the key is already in the default context space, take it from there
            if key in default_ctx_space.spaces:
                new_ctx_obs_spaces[key] = default_ctx_space.spaces[key]
            # else, we have to do it manually (for dynamic contexts)
            else:
                new_ctx_obs_spaces[key] = self._dctx_feature_to_gym_space(key)

        self.observation_space = SpaceDict({
            'obs': base_obs_space,
            'context': SpaceDict(new_ctx_obs_spaces)
        })

    def _check_args(self):
        # Validate that all dynamic context keys are registered with the CARL env.
        # We need this because CARL manages the context space, observation injection,
        # and range metadata -- an unregistered key would silently corrupt observations.
        # Walk the wrapper chain via .env (not .unwrapped which skips past CARL to the
        # base physics env) to find the first layer that has get_context_features.
        carl_env = None
        cur = self.env
        for _ in range(8):
            if hasattr(cur, "get_context_features"):
                carl_env = cur
                break
            cur = getattr(cur, "env", None)
            if cur is None:
                break

        if carl_env is not None:
            registered = carl_env.get_context_features()
            unregistered = [k for k in self.feature_update_fns if k not in registered]
            if unregistered:
                raise ValueError(
                    f"Dynamic context key(s) {unregistered} are not registered in the CARL env. "
                    f"Add them to get_context_features() before adding dynamic updates. "
                    f"Registered keys: {list(registered.keys())}"
                )

        # mode="none" requires ctx_to_observe=[] -- a non-empty list would build a
        # context space with entries, but _patch_obs_context empties the dict at runtime,
        # causing a KeyError in any downstream FlattenObservation wrapper.
        if self.observe_context_mode == "none" and (self.ctx_to_observe is None or self.ctx_to_observe):
            raise ValueError(
                f"observe_context_mode='none' requires ctx_to_observe=[] (empty), "
                f"but got: {self.ctx_to_observe}. "
                f"Pass ctx_to_observe=[] and obs_context_features=[] to the env factory."
            )

        # make sure that getters and setters are provided for all dynamic contexts
        for key in self.feature_update_fns.keys():
            if self.observe_context_mode in ('live', 'initial') and key not in self.ctx_getters:
                raise ValueError(f"No getter provided for dynamic context '{key}' (required for mode='{self.observe_context_mode}')")
            if key not in self.ctx_setters:
                raise ValueError(f"No setter provided for dynamic context '{key}'")

    def _snapshot_ctx(self):
        # just check if some of them are corrupted into None somehow
        # if not (self.ctx_to_change or self.ctx_to_observe):
        if not self.ctx_to_observe:
            keys = self.ctx_getters.keys()
        else:
            # take union of both sets
            keys = set(self.ctx_getters.keys())
            keys |= set(self.ctx_to_observe)

        out = {}
        for key in keys:
            getter = self.ctx_getters.get(key, None)
            if getter is None:
                out[key] = getattr(self.env.unwrapped, key, None)
            else:
                out[key] = getter(self.env)
        return out

    def _apply_ctx_updates(self, updates: dict):
        for k, v in updates.items():
            setter = self.ctx_setters.get(k, None)
            if setter is not None:
                setter(self.env, v)
            else:
                if hasattr(self.env.unwrapped, k):
                    setattr(self.env.unwrapped, k, v)
                elif self.verbose:
                    print(f"[Wrapper] No setter for '{k}', and no env.unwrapped.{k}")

    def _filter_ctx_keys(self, ctx_dict):
        """
        When correctly initialized, this should not filter anything out.
        -> fallback safety only
        """
        if self.ctx_to_observe is None:
            return dict(ctx_dict)
        return {k: v for k, v in ctx_dict.items() if k in self.ctx_to_observe}

    def _make_observed_context(self, true_ctx):
        """
        Return either
         - the real state of the context (live),
         - the initial state of the context (initial),
         - or None (no context observed).

        This is mainly support for the initial mode for unobserved dynamics of known contexts.
        """
        mode = self.observe_context_mode
        if mode == "live":
            return self._filter_ctx_keys(true_ctx)
        if mode == "initial":
            return self._filter_ctx_keys(self._ctx_reset_snapshot)
        if mode == "none":
            return None
        raise ValueError(f"Unknown observe_context_mode '{mode}'")

    def _patch_obs_context(self, obs, true_ctx):
        # Only handle dict observations that contain "context"
        if not (isinstance(obs, dict) and "context" in obs):
            return obs

        observed_ctx = self._make_observed_context(true_ctx)

        if observed_ctx is None:
            # Keep key, but make it empty so downstream flatteners can handle it
            obs = dict(obs)
            obs["context"] = {}
            return obs

        # IMPORTANT: the dynamic changes are not caught by CARL 'context' observation
        # -> we have to patch them in manually here
        if isinstance(obs["context"], dict):
            for k in list(obs["context"].keys()):
                if k in observed_ctx and observed_ctx[k] is not None:
                    obs["context"][k] = float(observed_ctx[k])
                else:
                    # If not observed, remove it to avoid implying information
                    # another part of the safety net to filter out unobserved contexts
                    # NOTE: if the carl envs was properly set, none of these should be present...
                    obs["context"].pop(k, None)
        else:
            if self.verbose:
                print("[Wrapper] obs['context'] not dict; leaving as is.")
        return obs

    def set_next_seed(self, seed: int | None):
        self._pending_seed = None if seed is None else int(seed)

    def reset(self, *, seed=None, options=None, **kwargs):
        # If caller (e.g., SB3) passes seed=None, use our pending one once.
        if seed is None and self._pending_seed is not None:
            seed = self._pending_seed
            self._pending_seed = None

        self.step_count = 0
        self.episode_count += 1

        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        ctx_now = self._snapshot_ctx()
        self._ctx_reset_snapshot = dict(ctx_now)

        for fn in self.feature_update_fns.values():
            if hasattr(fn, "reset"):
                fn.reset(self.env, self.episode_count, ctx=ctx_now, worker_id=self.worker_id)
            else:
                raise ValueError("Feature update function missing 'reset' method.")

        obs = self._patch_obs_context(obs, ctx_now)
        return obs, info

    def step(self, action):
        if self.step_count == 0:
            obs, reward, terminated, truncated, info = self.env.step(action)
            true_ctx = self._snapshot_ctx()
            obs = self._patch_obs_context(obs, true_ctx)
            self.step_count += 1
            return obs, reward, terminated, truncated, info

        to_update_ctx = {
            k: self.ctx_getters.get(k, lambda e: getattr(e.unwrapped, k, None))(self.env)
            for k in self.feature_update_fns
        }
        for key, update_fn in self.feature_update_fns.items():
            res = update_fn(self.env, self.step_count, to_update_ctx, self.verbose)
            if isinstance(res, dict):
                self._apply_ctx_updates(res)
            else:
                self._apply_ctx_updates({key: res})

        obs, reward, terminated, truncated, info = self.env.step(action)
        true_ctx = self._snapshot_ctx()
        obs = self._patch_obs_context(obs, true_ctx)

        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def get_true_context(self):
        """Return a snapshot of the real env context right now."""
        return self._snapshot_ctx()

    def get_observed_context_from_obs(self, obs, info=None):
        """
        Return the context as seen by the agent for this obs.
        Never infer from env when it is not present in obs.
        """
        if isinstance(obs, dict):
            if "context" in obs:
                return obs["context"]
            for k in ("ctx", "context_features"):
                if k in obs:
                    return obs[k]
        if isinstance(info, dict) and "context" in info:
            # only if your env explicitly uses info to expose observed context
            return info["context"]
        return None

class ContextOnlyNormalize(gym.ObservationWrapper):
    """
    Normalize only the 'context' part of the observation Dict using fixed ranges.
    Supports either:
        - observation_space["context"] is a Box
        - observation_space["context"] is a Dict[str, Box]
    Mapping per dim/key: v -> (v - mid) / half_range.
    """
    def __init__(self, env, ranges: dict[str, tuple[float, float]]):
        super().__init__(env)
        self.ranges = {k: (float(lo), float(hi)) for k, (lo, hi) in ranges.items()}

        obs_space = self.env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict) or "context" not in obs_space.spaces:
            raise ValueError("ContextOnlyNormalize expects a Dict obs space with a 'context' key.")

        ctx_space = obs_space.spaces["context"]
        self._is_dict_ctx = isinstance(ctx_space, gym.spaces.Dict)

        # Figure out keys and shapes
        if self._is_dict_ctx:
            # Order: use env.ctx_to_observe if present and matches, else keep Dict insertion order
            all_ctx_keys = list(ctx_space.spaces.keys())
            order_from_env = getattr(self.env, "ctx_to_observe", None)
            if order_from_env and all(k in all_ctx_keys for k in order_from_env):
                self._ctx_keys = list(order_from_env)
            else:
                self._ctx_keys = all_ctx_keys

            # Build per-key mid/half and the new per-key Box(-1,1)
            self._mid_per_key: dict[str, float] = {}
            self._half_per_key: dict[str, float] = {}
            new_ctx_spaces: dict[str, gym.spaces.Box] = {}

            for k in self._ctx_keys:
                sub: gym.spaces.Box = ctx_space.spaces[k]
                if not isinstance(sub, gym.spaces.Box):
                    raise ValueError(f"context subspace '{k}' must be Box, got {type(sub)}")

                # only support scalar (shape (1,) or ()) for context dims
                if sub.shape not in [(), (1,)]:
                    # If you really have multi-dim context entries, you can extend this logic,
                    # but for CartPole contexts these should be scalars.
                    raise ValueError(f"context subspace '{k}' must be scalar Box, got shape {sub.shape}")

                lo, hi = self.ranges.get(k, (float(sub.low.flatten()[0]), float(sub.high.flatten()[0])))
                mid = 0.5 * (lo + hi)
                half = max(1e-8, 0.5 * (hi - lo))
                self._mid_per_key[k] = float(mid)
                self._half_per_key[k] = float(half)

                new_ctx_spaces[k] = gym.spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                                   high=np.array([+1.0], dtype=np.float32),
                                                   dtype=np.float32)

            self.observation_space = gym.spaces.Dict({
                **{k: v for k, v in obs_space.spaces.items() if k != "context"},
                "context": gym.spaces.Dict(new_ctx_spaces),
            })

        else:
            # Single Box
            sub: gym.spaces.Box = ctx_space
            low  = np.array(sub.low,  dtype=np.float32).flatten()
            high = np.array(sub.high, dtype=np.float32).flatten()

            # Order: use env.ctx_to_observe if present; otherwise assume per-dim order
            order_from_env = getattr(self.env, "ctx_to_observe", None)
            if order_from_env is None:
                # fabricate keys k0..k{n-1} matching dims
                self._ctx_keys = [f"k{i}" for i in range(len(low))]
            else:
                self._ctx_keys = list(order_from_env)
                if len(self._ctx_keys) != len(low):
                    raise ValueError("ctx_to_observe length does not match context Box dims")

            mids, halves = [], []
            for i, k in enumerate(self._ctx_keys):
                lo_i, hi_i = self.ranges.get(k, (float(low[i]), float(high[i])))
                mid = 0.5 * (lo_i + hi_i)
                half = max(1e-8, 0.5 * (hi_i - lo_i))
                mids.append(mid)
                halves.append(half)
            self._mid_vec  = np.array(mids,  dtype=np.float32)
            self._half_vec = np.array(halves, dtype=np.float32)

            new_low  = np.full_like(low,  -1.0, dtype=np.float32)
            new_high = np.full_like(high, +1.0, dtype=np.float32)
            self.observation_space = gym.spaces.Dict({
                **{k: v for k, v in obs_space.spaces.items() if k != "context"},
                "context": gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32),
            })

    def observation(self, obs):
        obs = dict(obs)
        ctx = obs["context"]

        if self._is_dict_ctx:
            ctx_out = {}
            for k in self._ctx_keys:
                x = np.asarray(ctx[k], dtype=np.float32).reshape(-1)  # scalar -> (1,)
                mid = self._mid_per_key[k]
                half = self._half_per_key[k]
                y = (x - mid) / half
                y = np.clip(y, -5.0, 5.0).astype(np.float32)
                # keep same scalar/shape convention as input subspace (use (1,) for Box scalars)
                ctx_out[k] = y
            obs["context"] = ctx_out
        else:
            x = np.asarray(ctx, dtype=np.float32).reshape(-1)
            y = (x - self._mid_vec) / self._half_vec
            y = np.clip(y, -5.0, 5.0).astype(np.float32)
            obs["context"] = y

        return obs

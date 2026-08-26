from pathlib import Path
from typing import Any, Dict, Sequence, Callable, List, Optional, Tuple
import dynamic_carl.gym_context_updates as upd

def build_update_fn_from_spec(
    spec: Dict[str, Any],
    min_value: float,
    max_value: float,
    *,
    attribute: str,
) -> Callable[..., Any]:
    dyn_kind = spec.get("dyn_kind", "sinusoidal")
    fixed = spec.get("fixed", {}) or {}
    dyn_seed = spec.get("dyn_seed", None)

    if dyn_kind == "sinusoidal":
        frac = float(fixed.get("amplitude_frac", 0.0))
        period = int(fixed.get("period", 0))
        offset_from_context = bool(fixed.get("offset_from_context", True))
        amp = frac * (max_value - min_value) / 2.0

        return upd.make_sinusoidal(
            attribute=attribute,
            amplitude=amp,
            period=period,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
            dir_sign=+1,
            offset_from_context=offset_from_context,
        )

    elif dyn_kind == "continuous_incrementer":
        delta_frac = float(fixed["delta_frac"])
        delta = delta_frac * (max_value - min_value)
        direction = fixed.get("direction", "both")
        edge_mode = fixed.get("edge_mode", "reflect")
        episode_direction = fixed.get("episode_direction", "random")
        follow_predefined_prob = float(fixed.get("follow_predefined_prob", 0.8))

        return upd.make_continuous_incrementer(
            attribute=attribute,
            delta=delta,                      # <--- IMPORTANT: scheduler expects `delta`
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
            direction=direction,
            edge_mode=edge_mode,
            episode_direction=episode_direction,
            follow_predefined_prob=follow_predefined_prob,
        )

    elif dyn_kind == "cosine_annealing":
        # mode="once" is intentionally not exposed here: it requires T_max to match
        # the episode length, which is only reliable for fixed-length envs (e.g. CartPole).
        # Variable-length envs (Walker, CarRacing) should always use mode="cycle".
        T_0 = int(fixed["T_0"])
        T_mult = int(fixed.get("T_mult", 1))
        mode = fixed.get("mode", "cycle")

        # YAML uses a *fraction* of the full [min_value, max_value] span
        neighborhood_radius_frac = float(fixed.get("neighborhood_radius_frac", 0.0))
        span = (max_value - min_value)
        neighborhood_radius = (
            0.5 * neighborhood_radius_frac * span
            if neighborhood_radius_frac > 0.0
            else None
        )

        direction = fixed.get("direction", "auto")
        offset_from_context = bool(fixed.get("offset_from_context", True))
        retarget = fixed.get("retarget", "restart")

        return upd.make_cosine_annealing(
            attribute=attribute,
            T_0=T_0,
            T_mult=T_mult,
            mode=mode,
            min_val=min_value,
            max_val=max_value,
            offset_from_context=offset_from_context,
            neighborhood_radius=neighborhood_radius,   # <--- IMPORTANT: correct kwarg
            direction=direction,
            seed=dyn_seed,
            retarget=retarget,
        )

    elif dyn_kind == "random_walk":
        std_frac = float(fixed.get("std_frac", 0.01))
        std = std_frac * (max_value - min_value)
        return upd.make_random_walk(
            attribute=attribute,
            std=std,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
        )

    elif dyn_kind == "piecewise_constant":
        values = list(fixed["values"])
        interval_range = tuple(fixed.get("interval_range", (30, 50)))
        return upd.make_piecewise_constant(
            attribute=attribute,
            values=values,
            interval_range=interval_range,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
        )

    elif dyn_kind == "sudden_jump":
        # step_size expressed as fractions of the full range, consistent with other schedulers
        lo_frac = float(fixed.get("step_size_frac_range", [0.1, 0.3])[0])
        hi_frac = float(fixed.get("step_size_frac_range", [0.1, 0.3])[1])
        span = max_value - min_value
        step_size_range = (lo_frac * span, hi_frac * span)
        interval_range = tuple(fixed.get("interval_range", (10, 30)))
        direction = fixed.get("direction", "both")
        direction_prob = float(fixed.get("direction_prob", 0.5))
        edge_mode = fixed.get("edge_mode", "clip")
        return upd.make_sudden_jump(
            attribute=attribute,
            step_size_range=step_size_range,
            interval_range=interval_range,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
            direction=direction,
            direction_prob=direction_prob,
            edge_mode=edge_mode,
        )

    elif dyn_kind == "ornstein_uhlenbeck":
        span = max_value - min_value
        theta = float(fixed.get("theta", 0.05))
        mu_frac = fixed.get("mu_frac", None)
        mu = (min_value + mu_frac * span) if mu_frac is not None else None
        sigma_frac = float(fixed.get("sigma_frac", 0.08))
        sigma = sigma_frac * span
        return upd.make_ornstein_uhlenbeck(
            attribute=attribute,
            theta=theta,
            mu=mu,
            sigma=sigma,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
        )

    elif dyn_kind == "sinusoidal_jump":
        span = max_value - min_value
        amplitude_frac = float(fixed.get("amplitude_frac", 0.3))
        period = int(fixed.get("period", 100))
        lo_frac = float(fixed.get("step_size_frac_range", [0.2, 0.4])[0])
        hi_frac = float(fixed.get("step_size_frac_range", [0.2, 0.4])[1])
        step_size_range = (lo_frac * span, hi_frac * span)
        interval_range = tuple(fixed.get("interval_range", (80, 150)))
        return upd.make_sinusoidal_jump(
            attribute=attribute,
            amplitude=amplitude_frac * span * 0.5,
            period=period,
            step_size_range=step_size_range,
            interval_range=interval_range,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
        )

    elif dyn_kind == "phased_ou":
        span = max_value - min_value
        theta = float(fixed.get("theta", 0.05))
        sigma_frac = float(fixed.get("sigma_frac", 0.04))
        sigma = sigma_frac * span
        values = list(fixed.get("values", [min_value, (min_value + max_value) / 2, max_value]))
        interval_range = tuple(fixed.get("interval_range", (100, 150)))
        return upd.make_phased_ou(
            attribute=attribute,
            theta=theta,
            sigma=sigma,
            values=values,
            interval_range=interval_range,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
        )

    elif dyn_kind == "levy_walk":
        span = max_value - min_value
        std_frac = float(fixed.get("std_frac", 0.008))
        std = std_frac * span
        lo_frac = float(fixed.get("step_size_frac_range", [0.2, 0.4])[0])
        hi_frac = float(fixed.get("step_size_frac_range", [0.2, 0.4])[1])
        step_size_range = (lo_frac * span, hi_frac * span)
        interval_range = tuple(fixed.get("interval_range", (40, 80)))
        return upd.make_levy_walk(
            attribute=attribute,
            std=std,
            step_size_range=step_size_range,
            interval_range=interval_range,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
        )

    else:
        raise ValueError(f"Unknown dyn_kind: {dyn_kind}")

import os, sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

from dynamic_crl.src.gym_envs.cartpole_denv import CARLCartPole as cartpole
from dynamic_crl.src.gym_envs.walker_denv import CARLBipedalWalker # has the payload context
from dynamic_crl.src.gym_envs.vehicle_racing_denv import CARLVehicleRacing # has the payload context
from dynamic_crl.src.dynamic_carl.env_wrappers import GymDynamicContextCarlWrapper

from dynamic_crl.src.utils.log_msgs import warn_msg, info_msg
import gymnasium as gym
from Box2D.b2 import fixtureDef, polygonShape, weldJointDef
import math
import pygame
import numpy as np

def get_gym_base_env(e):
    """
    Unwrapes nested Gym wrappers (CARL included).
    Use to directly modify the environment physics engines.
    """
    cur = e
    for _ in range(16):
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return getattr(cur, "unwrapped", cur)

def get_CARL_env(env):
    """
    Unwraps nested Gym wrappers (FlattenObservation, DummyVecEnv, etc.)
    but stops at CARL envs which define `.context`.
    """
    while hasattr(env, "env") and not hasattr(env, "context"):
        env = env.env
    return env

def get_ctx_env_from_dummy_vec_env(env):
    """
    Unwraps DummyVecEnv to get the base environment.
    """
    if hasattr(env, "envs"):
        return env.envs[0].env # we have DummyVecEnv - OurContextWrapper -> the env used for training
    else:
        raise ValueError("Provided environment is not a DummyVecEnv")

# cartpole
def make_cp_ctx_accessor(attr: str):
    def getter(e):
        base = get_gym_base_env(e)
        return getattr(base, attr, None)
    def setter(e, v):
        base = get_gym_base_env(e)
        setattr(base, attr, float(v))
    return getter, setter

def _update_cp_payload(env, x_offset, mass):
    base = get_gym_base_env(env)
    base.masspayload = float(mass)
    base.payload_x_offset = float(x_offset)
    base.total_mass = base.masscart + base.masspole + base.masspayload
    
    # NEW: Since payload is at the center, its vertical distance is just half_l
    half_l = base.length 
    base.polemass_length = (base.masspole * half_l) + (base.masspayload * half_l)
    


class AttachCartPolePayload(gym.Wrapper):
    def __init__(self, env, payload_mass=0.2):
        super().__init__(env)
        self.payload_mass = payload_mass
        self.x_offset = 0.0

    def _apply(self):
        ctx = getattr(self.env, "context", {})
        self.x_offset = float(ctx.get("PAYLOAD_X", 0.0))
        # Update analytical constants (mass, polemass_length, etc.)
        _update_cp_payload(self.env, self.x_offset, self.payload_mass)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply()
        return obs, info

    def step(self, action):
        # 1. Let the original engine do its thing
        obs, reward, terminated, truncated, info = self.env.step(action)

        base = get_gym_base_env(self.env)
        x, x_dot, theta, theta_dot = base.state

        torque = self.payload_mass * base.gravity * (-self.x_offset) * math.cos(theta)
        inertia = base.polemass_length * base.length * (4.0 / 3.0)
        theta_acc_extra = torque / inertia

        # 2. Update state
        new_theta_dot = theta_dot + theta_acc_extra * base.tau
        new_theta = theta + new_theta_dot * base.tau
        base.state = (x, x_dot, new_theta, new_theta_dot)
        new_obs_arr = np.array(base.state, dtype=np.float32)

        # 3. Re-calculate termination based on the NEW modified angle
        terminated = bool(
            x < -base.x_threshold
            or x > base.x_threshold
            or new_theta < -base.theta_threshold_radians
            or new_theta > base.theta_threshold_radians
        )

        # 4. Preserve Dict obs structure if CARL wrapping is active
        if isinstance(obs, dict) and "obs" in obs:
            obs = dict(obs)
            obs["obs"] = new_obs_arr
            return obs, reward, terminated, truncated, info
        return new_obs_arr, reward, terminated, truncated, info

    def render(self):
        # 1. Let the original env draw the cart and pole
        self.env.render_mode = "rgb_array"
        img = self.env.render()
        self.env.render_mode = "human"
        
        # 2. Draw the payload overlay using your fixed coordinates
        base = get_gym_base_env(self.env)
        if self.env.render_mode == "human" and hasattr(base, "screen"):
            self._draw_payload(base)
            
        return img

    def _draw_payload(self, base):
        world_width = base.x_threshold * 2
        scale = base.screen_width / world_width
        
        x, x_dot, theta, theta_dot = base.state
        # scale * length * 2 is the full visual pole length
        full_pole_len = base.length * scale * 2 
        # The center point is just base.length * scale
        center_dist = base.length * scale 
        
        cartx = x * scale + base.screen_width / 2.0
        carty = 100 
        
        payload_local_x = self.x_offset * scale
        
        # NEW: use center_dist instead of full pole_len for the pivot math
        offset_x = payload_local_x * math.cos(theta) - center_dist * math.sin(theta)
        offset_y = payload_local_x * math.sin(theta) + center_dist * math.cos(theta)

        payload_screen_x = int(cartx - offset_x)
        payload_screen_y = int(base.screen_height - carty - offset_y)

        
        # Draw line from pole CENTER to payload
        mid_x = int(cartx + center_dist * math.sin(theta))
        mid_y = int(base.screen_height - carty - center_dist * math.cos(theta))
        pygame.draw.line(base.screen, (0, 0, 0), (mid_x, mid_y), (payload_screen_x, payload_screen_y), 2)
        
        # Draw Cred circle
        pygame.draw.circle(base.screen, (200, 0, 0), (payload_screen_x, payload_screen_y), 10)

        pygame.display.flip()

def cartpole_env_factory(contexts=None, render_mode=None, ctx_to_observe=None, payloaded=False, mass=0.1):

    if contexts is None:
        env = cartpole(render_mode=render_mode)
    else:
        env = cartpole(
            contexts=contexts, 
            render_mode=render_mode, 
            obs_context_features=ctx_to_observe
        )

    if payloaded:
        env = AttachCartPolePayload(env, payload_mass=mass)

    return env

# #################### #
# Bipedal Walker Utils #
# #################### #
def _get_attr(env, name: str):
    """
    Robust attribute fetch that:
      1) Reads from the base env (env.unwrapped) if present.
      2) Otherwise asks wrappers via get_wrapper_attr(name) without touching wrapper.<name>.
    Never falls back to getattr(wrapper, name) to avoid Gymnasium deprecation warnings.
    """
    # 1) base env first (no warnings)
    base = getattr(env, "unwrapped", env)
    if hasattr(base, name):
        return getattr(base, name)

    # 2) ask wrappers in a warning-free way
    try:
        val = env.get_wrapper_attr(name)
        if val is not None:
            return val
    except Exception:
        pass

    # 3) walk the chain, but still use get_wrapper_attr only
    cur = env
    for _ in range(16):
        try:
            val = cur.get_wrapper_attr(name)
            if val is not None:
                return val
        except Exception:
            pass
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break

    raise AttributeError(f"Attribute '{name}' not found on env or wrappers")

def _get_hull(e):
    # Wrapper-safe and future-proof: use our robust accessor
    return _get_attr(e, "hull")

def _get_payload_com_x(e) -> float:
    hull = _get_hull(e)
    # Read the exact local position of the payload fixture we attached!
    if hasattr(hull, "_com_payload") and hull._com_payload is not None:
        return float(hull._com_payload.shape.pos[0])
    return 0.0

def _set_payload_com_x(e, x: float, mass: float = 2.0, radius: float = 0.10, y_fixed: float = 0.0):
    """
    Shift COM by attaching a circular fixture at (x, y_fixed) in HULL LOCAL coords.
    Stores the created fixture as hull._com_payload so we can replace it next call.
    """
    hull = _get_hull(e)
    x = float(x); mass = float(mass); radius = float(radius); y_fixed = float(y_fixed)

    # remove previous payload if any
    if hasattr(hull, "_com_payload") and hull._com_payload is not None:
        try:
            hull.DestroyFixture(hull._com_payload)
        except Exception:
            pass
        hull._com_payload = None

    # add new payload only if mass > 0
    if mass > 0.0 and radius > 0.0:
        area = math.pi * (radius ** 2)
        density = mass / area
        hull._com_payload = hull.CreateCircleFixture(
            pos=(x, y_fixed), radius=radius, density=density,
            friction=0.0, restitution=0.0
        )

    # recompute mass/inertia/center
    hull.ResetMassData()
    return float(hull.massData.center[0])  # local COM x

class AttachWalkerPayload(gym.Wrapper):
    """
    Ensures a COM payload fixture exists on the hull.

    On reset:
      - Read current episode 'COM_X' from env.context (sampled by CARL).
      - Create/replace the circular fixture at that x with given mass/radius/y.
      - Remember this episode's pinned x.

    On step (optional):
      - If keep_center=True and there is NO COM_X dynamic updater, re-assert the
        *episode's* pinned x every step (prevents drift/flicker for static setups).
    """
    def __init__(
        self, env,
        *, mass: float = 5.0, radius: float = 0.40, y_fixed: float = 0.0,
        keep_center: bool = False, default_x0: float = 0.0
    ):
        super().__init__(env)
        self.mass = float(mass)
        self.radius = float(radius)
        self.y_fixed = float(y_fixed)
        self.keep_center = bool(keep_center)
        self.default_x0 = float(default_x0)

        self._has_dyn_comx: bool | None = None
        self._episode_pin_x: float = self.default_x0  # set at reset()

    def _refresh_dyn_flag(self) -> None:
        if self._has_dyn_comx is not None:
            return
        base = getattr(self.env, "unwrapped", self.env)
        fns = getattr(base, "feature_update_fns", None)
        if fns is None:
            try:
                fns = self.env.get_wrapper_attr("feature_update_fns")
            except Exception:
                fns = None
        self._has_dyn_comx = isinstance(fns, dict) and ("COM_X" in fns)

    def _read_com_x_from_context(self) -> float:
        # Try the standard CARL location
        ctx = None
        try:
            ctx = self.env.get_wrapper_attr("context")
        except Exception:
            pass
        if ctx is None:
            # Fallback to base env attribute if exposed
            base = getattr(self.env, "unwrapped", self.env)
            ctx = getattr(base, "context", None)

        if isinstance(ctx, dict) and ("COM_X" in ctx):
            try:
                return float(ctx["COM_X"])
            except Exception:
                pass
        return self.default_x0

    def reset(self, *args, **kwargs):
        # Let CARL sample the context first
        out = self.env.reset(*args, **kwargs)

        # Detect whether a dynamic COM_X updater exists
        self._has_dyn_comx = None
        self._refresh_dyn_flag()

        # Read the *sampled* COM_X for this episode and attach payload there
        self._episode_pin_x = self._read_com_x_from_context()
        _set_payload_com_x(
            self.env,
            x=self._episode_pin_x,
            mass=self.mass,
            radius=self.radius,
            y_fixed=self.y_fixed,
        )
        return out

    def step(self, action):
        result = self.env.step(action)

        # If static setup (no COM_X updater) and keep_center=True, re-assert pin
        self._refresh_dyn_flag()
        if self.keep_center and not self._has_dyn_comx:
            _set_payload_com_x(
                self.env,
                x=self._episode_pin_x,
                mass=self.mass,
                radius=self.radius,
                y_fixed=self.y_fixed,
            )
        return result

def walker_env_factory(
    contexts=None, render_mode=None, ctx_to_observe=None,
    payloaded=False, keep_center=False, payload_kwargs=None
):
    """
    Factory for CARL Walker with optional COM payload.
    - 'contexts' lets CARL sample COM_X per episode (we use it on reset).
    - 'keep_center': if True and NO COM_X updater, re-assert per-episode COM_X every step.
    """
    if contexts is None:
        from carl.envs import CARLBipedalWalker as walker
        env = walker(render_mode=render_mode)
    else:
        env = CARLBipedalWalker(
            contexts=contexts,
            render_mode=render_mode,
            obs_context_features=ctx_to_observe
        )

    env.render_mode = render_mode

    if payloaded:
        pk = payload_kwargs or {}
        env = AttachWalkerPayload(
            env,
            mass=pk.get("mass", 1.5),
            radius=pk.get("radius", 0.25),
            y_fixed=pk.get("y_fixed", 0.0),
            keep_center=keep_center,
            default_x0=0.0,   # only used if COM_X is missing in context
        )

    return env

# ################ #
# Car Racing Utils #
# ################ #

# Note/Guide for selecting the right offset mass:
# the RacerCar (ID 0) has approx weight 7.34 --> (7.06 hull + 4x0.06 wheels)
# the StreetCar (ID 9) has approx weight 12.29 --> (12.12 hull + 4x0.043 wheels)
# the Bus (ID 18) has approx weight 15.07 --> (14.72 hull + 4x0.09 wheels)
# the TukTuk (ID 27) has approx weight 7.90 --> (7.76 hull + 3x0.048 wheels)

# Helper to get the internal Box2D car object
# --- Define Getter/Setter for the Car Payload ---
def get_racer_payload_x(env):
    """Reads the current X position of the payload from the car object."""
    car = _get_car(env)
    return getattr(car, "payload_current_x")

def get_racer_payload_y(env):
    """Reads the current Y position of the payload from the car object."""
    car = _get_car(env)
    return getattr(car, "payload_current_y")

def set_racer_payload(env, x_value, y_value, mass=1.0, radius=0.5):
    """
    Updates the payload position using the helper we created in gym_utils.
    We match the mass/radius used in the factory to keep it consistent.
    """
    # if one or the other is None, its comming from the dynamic update
    # and we need to fetch the value first
    if x_value is None:
        x_value = get_racer_payload_x(env)
    elif y_value is None:
        y_value = get_racer_payload_y(env)

    _update_payload(
        env, 
        x_value=x_value, 
        y_value=y_value, 
        mass=mass,
        radius=radius
    )

def _get_car(env):
    # Unwrap until we find the car or hit the bottom
    base = env
    while True:
        if hasattr(base, "car") and base.car is not None:
            return base.car
        if hasattr(base, "env"):
            base = base.env
        else:
            break
    # Fallback: try wrapper attribute getter
    try:
        return env.get_wrapper_attr("car")
    except Exception:
        pass
    return None

def get_init_y_per_car(vehicle_id):
    if vehicle_id >=0 and vehicle_id < 9:
        return 0
    elif vehicle_id < 18:
        return 1.2
    elif vehicle_id < 27:
        return 5.5
    elif vehicle_id < 29:
        return 0
    else:
        raise ValueError("Incorrect vehicle id provided")

def _update_payload(env, x_value, y_value, mass, radius):
    """
    Manages the payload body with state handover to preserve momentum.
    """
    car = _get_car(env)
    if car is None or car.hull is None:
        return

    # Check for identical position to skip redundant calculations
    if (
        hasattr(car, "payload_current_x")
        and car.payload_current_x == x_value
        and hasattr(car, "payload_current_y")
        and car.payload_current_y == y_value
        ):
        return

    world = car.hull.world
    x_value, y_value = float(x_value), float(y_value)
    mass, radius = float(mass), float(radius)

    # 1. REMOVE OLD
    if hasattr(car, "payload_body") and car.payload_body is not None:
        if car.payload_body in car.drawlist:
            car.drawlist.remove(car.payload_body)
        world.DestroyBody(car.payload_body)
        car.payload_body = None

    # 2. CREATE NEW
    if mass > 0.0 and radius > 0.0:
        # Calculate world position based on CAR's current transform + offset
        spawn_pos = car.hull.GetWorldPoint(localPoint=(x_value, y_value))
        
        payload_body = world.CreateDynamicBody(
            position=spawn_pos,
            angle=car.hull.angle,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[
                    (radius * math.cos(2 * math.pi * i / 16), 
                     radius * math.sin(2 * math.pi * i / 16))
                    for i in range(16)
                ]),
                density=mass / (math.pi * radius**2),
                friction=0.0,
                restitution=0.0
            )
        )
        
        # --- STATE HANDOVER ---
        # We set the payload velocity to match the car BEFORE welding.
        # This ensures the 0.5kg mass is already 'moving' with the car,
        # preventing the car from losing speed due to inelastic collision.
        payload_body.linearVelocity = car.hull.linearVelocity
        payload_body.angularVelocity = car.hull.angularVelocity
        # ----------------------

        payload_body.color = (0.0, 1.0, 1.0) # Cyan

        # 3. WELD IT
        joint_def = weldJointDef(
            bodyA=car.hull,
            bodyB=payload_body,
            localAnchorA=(x_value, y_value),
            localAnchorB=(0, 0),
            referenceAngle=0.0
        )
        world.CreateJoint(joint_def)

        # 4. REGISTER
        car.drawlist.append(payload_body)
        car.payload_body = payload_body
        car.payload_current_x = x_value 
        car.payload_current_y = y_value

class AttachCarPayload(gym.Wrapper):
    def __init__(self, env, mass=5.0, radius=0.5):
        super().__init__(env)
        self.mass = mass
        self.radius = radius

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        
        # Initial create
        ctx = getattr(self.env, "context", {})

        y_value = ctx.get("COM_Y", None)
        if y_value is None:
            y_value = get_init_y_per_car(ctx['VEHICLE_ID']) if 'VEHICLE_ID' in ctx else 0.0
        y_value = float(y_value)

        x_value = float(ctx.get("COM_X", 0.0)) # if given static, use it, else 0

        _update_payload(self.env, x_value, y_value, self.mass, self.radius)
        return obs, info

    def step(self, action):   
        # Convert actions to float (Box2D safety)
        action = [float(x) for x in action]
        return self.env.step(action)

def racer_env_factory(
    contexts=None,
    ctx_to_observe=None,
    render_mode=None,
    payloaded=False,
    payload_kwargs=None,
    verbose=False
):
    env = CARLVehicleRacing(
        contexts=contexts,
        render_mode=render_mode,
        obs_context_features=ctx_to_observe,
    )
    # Bypass all wrappers and tell the raw Box2D CarRacing environment to shut up!
    if hasattr(env.unwrapped, "verbose"):
        env.unwrapped.verbose = 1 if verbose else 0

    if payloaded:
        pk = payload_kwargs or {}
        env = AttachCarPayload(
            env,
            mass=pk.get("mass", 2.0),
            radius=pk.get("radius", 0.5),
        )

    return env

# #################### #
#  Development Script  #
# #################### #
if __name__ == "__main__":

    test = "cartpole"

    if test == "car":
        # Test specific vehicle ID + Payload
        # ID 6 is usually a Bus or Heavy vehicle in your list
        test_context = {
            0: {
                "VEHICLE_ID": 0,   # Try changing this (0=RaceCar, 1=FWD, etc.)
                "COM_X": 1,       # Offset payload to the right
                "COM_Y": 0
            }
        }

        env = racer_env_factory(
            contexts=test_context,
            ctx_to_observe=None, # blind to context features
            render_mode="human",
            payloaded=True,
            payload_kwargs={"mass": 2.0, "radius": 0.5, "axis": "x"} # Heavy payload
        )

        print("Resetting...")
        env.reset()

        try:
            i=0
            while i < 100:
                # Drive forward and slightly right to feel the weight
                action = [0.0, 1.0, 0.0]
                env.step(action)
                env.render() # Crucial for Pygame window updates!
                i += 1
        except KeyboardInterrupt:
            env.close()

    elif test == "cartpole":
        test_context = {
            0: {
                "PAYLOAD_X": 0.2  # Offset payload to the right
            }
        }

        env = cartpole_env_factory(
            contexts=test_context,
            ctx_to_observe=None,
            render_mode="human",
            payloaded=True,
            mass=0.3
        )

        print("Resetting...")
        env.reset(seed=0)

        try:
            i=0
            while i < 50:
                # action = env.action_space.sample()
                action = np.int64(i%2) 
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    obs, info = env.reset()

                env.render()
                i += 1
                #time.sleep(0.4)

        except KeyboardInterrupt:
            env.close()
    else:
        print("Unknown test specified.")
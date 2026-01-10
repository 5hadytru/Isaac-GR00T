"""
SO100 Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the SO100 / SO101 robots
using the GR00T Policy API.

Major responsibilities:
    • Initialize robot hardware from a RobotConfig (LeRobot)
    • Convert robot observations into GR00T VLA inputs
    • Query the GR00T policy server (PolicyClient)
    • Decode multi-step (temporal) model actions back into robot motor commands
    • Stream actions to the real robot in real time

This file is meant to be a simple, readable reference
for real-world policy debugging and demos.
"""

# =============================================================================
# Imports
# =============================================================================
from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import time
from typing import Any, Dict, List

import draccus
from gr00t.policy.server_client import PolicyClient

# Importing various robot configs ensures CLI autocompletion works
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import init_logging, log_say
import numpy as np


# image compression stuff
import base64

def _is_uint8_hwc3(x: Any) -> bool:
    return (
        isinstance(x, np.ndarray)
        and x.dtype == np.uint8
        and x.ndim == 3
        and x.shape[2] == 3
    )

def compress_obs_images(
    obs,
    *,
    image_keys = ("front", "overhead"),
    codec: str = "jpg",               # "jpg" or "png" (jpg recommended for bandwidth)
    quality: int = 70,                # for jpg/webp; 1-100
    png_compression: int = 3,         # 0-9 (higher = smaller/slower)
    assume_rgb: bool = True,          # if True, preserves RGB when using OpenCV
    base64_encode: bool = False,      # True if you need JSON-safe strings
):
    """
    Returns a shallow-copied obs dict where selected images are replaced by a compressed payload:
      {"__compressed__": True, "codec": ..., "data": bytes or b64 str, "shape": ..., "dtype": "uint8", "color": "rgb|bgr"}
    """
    out: Dict[str, Any] = dict(obs)

    # Prefer OpenCV if available (fast). Fall back to PIL if not.
    try:
        import cv2  # type: ignore
        use_cv2 = True
    except Exception:
        use_cv2 = False

    for k in image_keys:
        if k not in obs:
            continue
        img = obs[k]
        if not _is_uint8_hwc3(img):
            continue

        h, w, c = img.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels for {k}, got {img.shape}")

        codec_l = codec.lower().lstrip(".")
        if use_cv2:
            enc_img = img
            color = "rgb" if assume_rgb else "bgr"
            # OpenCV treats arrays as BGR for typical image conventions; convert if your arrays are RGB.
            if assume_rgb:
                enc_img = img[..., ::-1]  # RGB -> BGR

            if codec_l in ("jpg", "jpeg"):
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                ext = ".jpg"
            elif codec_l == "png":
                params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
                ext = ".png"
            else:
                raise ValueError(f"Unsupported codec for cv2: {codec}. Use 'jpg' or 'png'.")

            ok, buf = cv2.imencode(ext, enc_img, params)
            if not ok:
                raise RuntimeError(f"cv2.imencode failed for key={k}, codec={codec_l}")
            data_bytes = buf.tobytes()

        else:
            from io import BytesIO
            from PIL import Image  # type: ignore

            if not assume_rgb:
                # If your arrays are BGR, convert to RGB before PIL.
                pil_arr = img[..., ::-1]
                color = "bgr"
            else:
                pil_arr = img
                color = "rgb"

            im = Image.fromarray(pil_arr, mode="RGB")
            bio = BytesIO()
            if codec_l in ("jpg", "jpeg"):
                im.save(bio, format="JPEG", quality=int(quality), optimize=True)
            elif codec_l == "png":
                im.save(bio, format="PNG", compress_level=int(png_compression))
            else:
                raise ValueError(f"Unsupported codec for PIL: {codec}. Use 'jpg' or 'png'.")
            data_bytes = bio.getvalue()

        data: Any = data_bytes
        if base64_encode:
            data = base64.b64encode(data_bytes).decode("ascii")

        out[k] = {
            "__compressed__": True,
            "codec": codec_l,
            "data": data,
            "shape": (h, w, c),
            "dtype": "uint8",
            "color": color,          # what the ORIGINAL obs used (so we can restore correctly)
            "assume_rgb": assume_rgb # stored for clarity/debugging
        }

    return out
# image compression stuff

def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
    return obs


class So100Adapter:
    """
    Adapter between:
        • Raw robot observation dictionary
        • GR00T VLA input format
        • GR00T action chunk → robot joint commands

    Responsible for:
        • Packaging camera frames as obs["video"]
        • Building obs["state"] for arm + gripper
        • Adding language instruction
        • Adding batch/time dimensions
        • Decoding model action chunks into real robot actions
    """

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client

        # SO100 joint ordering used for BOTH training + robot execution
        self.robot_state_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.camera_keys = ["overhead", "front"]

    # -------------------------------------------------------------------------
    # Observation → Model Input
    # -------------------------------------------------------------------------
    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """
        Convert raw robot observation dict into the structured GR00T VLA input.
        """
        model_obs = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Arm + gripper state
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:5],  # (5,)
            "gripper": state[5:6],  # (1,)
        }

        # (3) Language
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    # -------------------------------------------------------------------------
    # Model Action Chunk → Robot Motor Commands
    # -------------------------------------------------------------------------
    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        """
        chunk["single_arm"]: (B, T, 5)
        chunk["gripper"]:    (B, T, 1)

        Convert to:
            {
                "shoulder_pan.pos": val,
                ...
            }
        for timestep t.
        """
        single_arm = chunk["single_arm"][0][t]  # (5,)
        gripper = chunk["gripper"][0][t]  # (1,)

        full = np.concatenate([single_arm, gripper], axis=0)  # (6,)

        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        """
        Returns a list of robot motor commands (one per model timestep).
        """
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        # Determine horizon
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Evaluation Config
# =============================================================================


@dataclass
class EvalConfig:
    """
    Command-line configuration for real-robot policy evaluation.
    """

    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "Grab markers and place into pen holder."
    play_sounds: bool = False
    timeout: int = 60


# =============================================================================
# Main Eval Loop
# =============================================================================


@draccus.wrap()
def eval(cfg: EvalConfig):
    """
    Main entry point for real-robot policy evaluation.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # -------------------------------------------------------------------------
    # 1. Initialize Robot Hardware
    # -------------------------------------------------------------------------
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    # -------------------------------------------------------------------------
    # 2. Initialize Policy Wrapper + Client
    # -------------------------------------------------------------------------
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = So100Adapter(policy_client)

    log_say(
        f'Policy ready with instruction: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )

    # -------------------------------------------------------------------------
    # 3. Main real-time control loop
    # -------------------------------------------------------------------------
    while True:
        obs = robot.get_observation()
        obs["lang"] = cfg.lang_instruction  # insert language

        # obs = {
        #     "front": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "shoulder_pan.pos": 0.0,
        #     "shoulder_lift.pos": 0.0,
        #     "elbow_flex.pos": 0.0,
        #     "wrist_flex.pos": 0.0,
        #     "wrist_roll.pos": 0.0,
        #     "gripper.pos": 0.0,
        #     "lang": cfg.lang_instruction,
        # }

        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                print(f"obs[{k}]: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"obs[{k}]: {v}")
        for k, v in compress_obs_images(obs).items():
            if isinstance(v, np.ndarray):
                print(f"c_obs[{k}]: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"c_obs[{k}]: {type(v)}")

        actions = policy.get_action(compress_obs_images(obs))

        for i, action_dict in enumerate(actions[: cfg.action_horizon]):
            tic = time.time()
            print(f"action[{i}]: {action_dict}")
            # action_dict = {
            #     "shoulder_pan.pos":    5.038022994995117,
            #     "shoulder_lift.pos":  17.09104347229004,
            #     "elbow_flex.pos":    -18.519847869873047,
            #     "wrist_flex.pos":     86.86847686767578,
            #     "wrist_roll.pos":      1.0669738054275513,
            #     "gripper.pos":        36.83877944946289,
            # }
            robot.send_action(action_dict)
            toc = time.time()
            if toc - tic < 1.0 / 30:
                time.sleep(1.0 / 30 - (toc - tic))


if __name__ == "__main__":
    eval()

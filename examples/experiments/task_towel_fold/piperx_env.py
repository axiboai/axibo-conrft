"""Gym interface for bimanual PiperX using the existing ZMQ rollout stack.

This env is the ConRFT actor-side adapter for the same streams used by your
OpenPI scripts:
    state JSON      tcp://localhost:3335
    cam_front       tcp://localhost:5560
    cam_left_wrist  tcp://localhost:5556
    cam_right_wrist tcp://localhost:5558
    target PUB      tcp://0.0.0.0:3336

Observations match converted demo schema:
    state["joint_pos"] = (14,)
    images["side_policy_256"] = (256,256,3)
    images["wrist_1"] = (128,128,3)
"""

import copy
import json
import queue
import struct
import threading
import time
from typing import Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
import zmq


class PiperXEnvConfig:
    """Fill these in for your PiperX workspace."""

    # Legacy HTTP URL (unused in ZMQ mode; kept for compatibility).
    SERVER_URL: str = "http://127.0.0.1:5000/"

    # ZMQ endpoints (aligned with your OpenPI/HIL stack).
    STATE_ADDR: str = "tcp://localhost:3335"
    TARGET_ADDR: str = "tcp://0.0.0.0:3336"
    CAMERAS: Dict[str, str] = {
        "side_policy_256": "tcp://localhost:5560",  # cam_front
        "wrist_1": "tcp://localhost:5558",          # right wrist by default
    }
    # Optional per-camera crop callables applied before resize.
    IMAGE_CROP: Dict[str, callable] = {}

    STATE_DIM: int = 14
    ACTION_DIM: int = 14

    # Reset joint configuration (absolute joint targets) the robot returns to.
    RESET_JOINTS: np.ndarray = np.zeros((14,), dtype=np.float32)

    # Normalized action -> joint delta scaling (per joint). Actions are clipped
    # to [-1, 1] then multiplied by ACTION_SCALE and added to current joints.
    ACTION_SCALE: np.ndarray = np.full((14,), 0.05, dtype=np.float32)

    # Hard joint limits for safety clipping (absolute targets).
    JOINT_LIMIT_LOW: np.ndarray = np.full((14,), -np.pi, dtype=np.float32)
    JOINT_LIMIT_HIGH: np.ndarray = np.full((14,), np.pi, dtype=np.float32)

    DISPLAY_IMAGE: bool = True
    MAX_EPISODE_LENGTH: int = 200
    CONTROL_HZ: float = 10.0


class ImageDisplayer(threading.Thread):
    def __init__(self, q, name):
        threading.Thread.__init__(self)
        self.queue = q
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for v in img_array.values()], axis=1)
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


class PiperXEnv(gym.Env):
    def __init__(self, fake_env=False, save_video=False, config: PiperXEnvConfig = None):
        self.config = config or PiperXEnvConfig()
        self.url = self.config.SERVER_URL
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.hz = self.config.CONTROL_HZ
        self.display_image = self.config.DISPLAY_IMAGE
        self.save_video = save_video
        self.latest_state_msg = None
        self._command_seq = 0

        self.action_space = gym.spaces.Box(
            -np.ones((self.config.ACTION_DIM,), dtype=np.float32),
            np.ones((self.config.ACTION_DIM,), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "joint_pos": gym.spaces.Box(-np.inf, np.inf,
                                            shape=(self.config.STATE_DIM,)),
            }),
            "images": gym.spaces.Dict({
                "side_policy_256": gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8),
                "wrist_1": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            }),
        })

        self.currpos = self.config.RESET_JOINTS.copy()
        self.curr_path_length = 0
        self.terminate = False
        self.recording_frames = []

        if fake_env:
            return

        self.ctx = zmq.Context.instance()
        self.state_sub = ConflateSub(self.ctx, self.config.STATE_ADDR)
        self.cam_sub = {k: CameraSub(self.ctx, v) for k, v in self.config.CAMERAS.items()}
        self.target_pub = self.ctx.socket(zmq.PUB)
        self.target_pub.setsockopt(zmq.SNDHWM, 1)
        self.target_pub.setsockopt(zmq.LINGER, 0)
        self.target_pub.bind(self.config.TARGET_ADDR)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.config.TARGET_ADDR)
            self.displayer.start()

        from pynput import keyboard
        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True
        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        print("Initialized PiperX env")

    def _get_state(self):
        msg = self.state_sub.latest()
        if msg is None:
            return self.currpos
        try:
            state = json.loads(msg)
            self.latest_state_msg = state
            self.currpos = pack_follower_state_14(state)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return self.currpos

    def _send_command(self, joint_targets):
        joint_targets = np.clip(joint_targets,
                                self.config.JOINT_LIMIT_LOW,
                                self.config.JOINT_LIMIT_HIGH)
        joint_targets = np.asarray(joint_targets, dtype=np.float32)
        payload = {
            "t_mono": time.monotonic(),
            "seq": self._command_seq,
            "source": "policy",
            "left": {
                "q": joint_targets[:6].tolist(),
                "gripper": float(joint_targets[6]),
            },
            "right": {
                "q": joint_targets[7:13].tolist(),
                "gripper": float(joint_targets[13]),
            },
        }
        try:
            self.target_pub.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            self._command_seq += 1
        except zmq.Again:
            pass

    def _server_reset(self):
        # Publish reset target for a short ramp window so follower sinks receive it.
        for _ in range(max(1, int(self.hz * 0.5))):
            self._send_command(self.config.RESET_JOINTS)
            time.sleep(1.0 / self.hz)

    def get_im(self) -> Dict[str, np.ndarray]:
        images, display_images = {}, {}
        for key, sub in self.cam_sub.items():
            sub.poll()
            rgb = sub.last_rgb
            if rgb is None:
                # Missing stream -> black placeholder; keeps actor loop alive.
                shape = self.observation_space["images"][key].shape
                rgb = np.zeros(shape, dtype=np.uint8)
            if key in self.config.IMAGE_CROP:
                rgb = self.config.IMAGE_CROP[key](rgb)
            target = self.observation_space["images"][key].shape[:2][::-1]
            resized = cv2.resize(rgb, target)
            images[key] = resized.astype(np.uint8)
            display_images[key] = resized
        self.recording_frames.append(copy.deepcopy(images))
        if self.display_image:
            self.img_queue.put(display_images)
        return images

    # ----------------------------------------------------------------- gym API
    def _get_obs(self):
        images = self.get_im()
        return copy.deepcopy(dict(
            images=images,
            state={"joint_pos": self.currpos.astype(np.float32)},
        ))

    def compute_reward(self, obs) -> bool:
        # Real success signal comes from the reward-classifier wrapper.
        return False

    def step(self, action: np.ndarray):
        start = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target = self.currpos + action * self.config.ACTION_SCALE
        self._send_command(target)

        self.curr_path_length += 1
        time.sleep(max(0.0, (1.0 / self.hz) - (time.time() - start)))

        self._get_state()
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = (self.curr_path_length >= self.max_episode_length
                or bool(reward) or self.terminate)
        return obs, int(reward), done, False, {"succeed": bool(reward)}

    def reset(self, **kwargs):
        self._server_reset()
        time.sleep(1.0)
        self.curr_path_length = 0
        self.terminate = False
        self._get_state()
        obs = self._get_obs()
        return obs, {"succeed": False}

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        if hasattr(self, "state_sub"):
            self.state_sub.close()
        if hasattr(self, "cam_sub"):
            for sub in self.cam_sub.values():
                sub.close()
        if hasattr(self, "target_pub"):
            self.target_pub.close(linger=0)
        if self.display_image and hasattr(self, "img_queue"):
            self.img_queue.put(None)
            cv2.destroyAllWindows()


def pack_follower_state_14(state_msg: dict) -> np.ndarray:
    f_left = state_msg["follower_left"]
    f_right = state_msg["follower_right"]
    out = np.zeros((14,), dtype=np.float32)
    out[:6] = np.asarray(f_left["q"][:6], dtype=np.float32)
    out[6] = float(f_left["gripper"])
    out[7:13] = np.asarray(f_right["q"][:6], dtype=np.float32)
    out[13] = float(f_right["gripper"])
    return out


class ConflateSub:
    """Single-part text SUB with latest-only semantics."""

    def __init__(self, ctx: zmq.Context, addr: str):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.RCVHWM, 1)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(addr)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        self.s = s
        self._last = None

    def poll(self) -> Optional[str]:
        latest = None
        while True:
            try:
                latest = self.s.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is not None:
            self._last = latest
        return latest

    def latest(self) -> Optional[str]:
        self.poll()
        return self._last

    def close(self):
        self.s.close(linger=0)


class CameraSub:
    """JPEG RGB camera subscriber (multipart format: topic, ts_ns, jpeg)."""

    def __init__(self, ctx: zmq.Context, addr: str, topic: bytes = b"image"):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 1)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(addr)
        s.setsockopt(zmq.SUBSCRIBE, topic)
        self.s = s
        self.topic = topic
        self.last_rgb = None

    def poll(self):
        while True:
            try:
                parts = self.s.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) < 3 or parts[0] != self.topic or len(parts[1]) != 8:
                continue
            try:
                struct.unpack(">Q", parts[1])[0]
            except struct.error:
                continue
            jpeg = parts[2]
            if len(jpeg) < 2 or jpeg[:2] != b"\xff\xd8":
                continue
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            self.last_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.s.close(linger=0)

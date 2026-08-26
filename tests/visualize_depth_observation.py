import gymnasium as gym
import mani_skill.envs
import matplotlib.pyplot as plt
import numpy as np


env = gym.make(
    "PushCube-v1",
    num_envs=1,
    obs_mode="rgb+depth",
    sim_backend="physx_cpu",
    render_backend="sapien_cpu",
)

obs, _ = env.reset(seed=1)

rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
depth = obs["sensor_data"]["base_camera"]["depth"][0, ..., 0].cpu().numpy()

print("depth min:", depth.min())
print("depth max:", depth.max())
print("depth mean:", depth.mean())

# 为了显示，把毫米转成米
depth_m = depth.astype(np.float32) / 1000.0

# 显示时只保留近距离区域，提高对比度
depth_vis = np.clip(depth_m, 0.2, 2.0)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("RGB")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_vis, cmap="viridis")
plt.colorbar(label="Depth (m)")
plt.title("Depth")
plt.axis("off")

plt.tight_layout()
plt.savefig("depth_observation.png", dpi=200)
plt.show()

env.close()
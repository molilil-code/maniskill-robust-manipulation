\# PushCube PPO CPU Baseline



\## Purpose



验证 Windows CPU 条件下 ManiSkill PPO 的完整训练流程：

rollout → GAE → PPO update → evaluation → checkpoint。



\## Environment



\- OS: Windows

\- Task: PushCube-v1

\- Observation: state

\- Backend: physx\_cpu

\- Reward mode: normalized\_dense

\- num\_envs: 1

\- num\_eval\_envs: 1



\## PPO Configuration



\- total\_timesteps: 300000

\- num\_steps: 50

\- num\_minibatches: 2

\- update\_epochs: 4

\- learning\_rate: 3e-4

\- gamma: 0.8

\- gae\_lambda: 0.9

\- clip\_coef: 0.2

\- target\_kl: 0.1



\## Command



python ppo\_cpu.py --total\_timesteps=300000



\## Results



\- Training time: about 18.8 min

\- SPS: about 300

\- eval/success\_at\_end: 0

\- value loss decreased significantly

\- Actor and Critic were updated normally

\- Policy did not solve PushCube



\## Analysis



The complete PPO training pipeline worked successfully.



However, the CPU setup used only:



1 env × 50 steps = 50 samples per rollout



which is substantially smaller and less diverse than the official GPU parallel setting.



\## Conclusion



This experiment is retained as a CPU pipeline verification experiment,

not as the final PPO baseline.



Next step: reproduce the official PushCube PPO baseline with Linux,

NVIDIA GPU and physx\_cuda.


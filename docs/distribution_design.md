# Distribution Design

## Baseline Training Distribution

Official ManiSkill PushCube-v1.

| Parameter | Distribution |
|---|---|
| Cube x/y | U(-0.10, 0.10) m |
| Goal offset | 0.20 m |
| Robot qpos noise | sigma = 0.02 rad |

## DR V1 Training Distribution

PushCubeDR-v1.

| Parameter | Distribution |
|---|---|
| Cube x/y | U(-0.13, 0.13) m |
| Goal offset | U(0.17, 0.23) m |
| Robot qpos noise | sigma = 0.04 rad |

## OOD Evaluation Conditions

### Normal
Same as baseline training distribution.

### Cube Position OOD
At least one of x/y lies in:

[-0.16, -0.13] or [0.13, 0.16]

### Goal Near OOD
goal offset ~ U(0.14, 0.16) m

### Goal Far OOD
goal offset ~ U(0.24, 0.26) m

### Qpos Shift
robot qpos noise sigma = 0.06 rad

### Combined OOD
- Cube position: outer band
- Goal offset: U(0.24, 0.26)
- Robot qpos noise: sigma = 0.06

| condition    | Cube     | Goal         | Qpos        |
| ------------ | -------- | ------------ | ----------- |
| `normal`     | baseline | baseline     | baseline    |
| `cube_ood`   | **OOD**  | baseline     | baseline    |
| `goal_near`  | baseline | **near OOD** | baseline    |
| `goal_far`   | baseline | **far OOD**  | baseline    |
| `qpos_shift` | baseline | baseline     | **shifted** |
| `combined`   | **OOD**  | **far OOD**  | **shifted** |

## V2 Physics Distribution

Nominal PushCube physics parameters:

| Parameter | Nominal |
|---|---:|
| Mass | 0.064 kg |
| Static friction | 0.30 |
| Dynamic friction | 0.30 |
| Restitution | 0.00 |

### Physics DR Training Distribution

| Parameter | Distribution |
|---|---|
| Mass | U(0.0448, 0.0832) kg |
| Static friction | U(0.21, 0.39) |
| Dynamic friction | Same as static friction |
| Restitution | 0 |

### Physics OOD Evaluation

| Condition | Mass | Friction |
|---|---:|---:|
| mass_low | 0.032 kg | 0.30 |
| mass_high | 0.096 kg | 0.30 |
| friction_low | 0.064 kg | 0.15 |
| friction_high | 0.064 kg | 0.45 |
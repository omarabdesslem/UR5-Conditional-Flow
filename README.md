# Conditional Flow Matching for Berkeley UR5 Demonstrations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

A research baseline for **conditional, stochastic trajectory generation from real Berkeley Autolab UR5 demonstrations** using rectified flow matching, specialist temporal-convolution teachers, and on-policy ensemble-to-student distillation.

Given a recorded **start state**, **goal state**, and **task ID**, the model generates a 32-step, 8-channel state trajectory. A Gaussian residual trajectory is transported through a learned velocity field using numerical ODE integration. The first and last states are inserted exactly, while the 30 interior states are generated.

> **Scope:** This experiment uses the numeric portion of `lerobot/berkeley_autolab_ur5`. **It does not load occupancy maps, obstacle geometry, images, camera video, or RRT plans.** It therefore does not establish collision-free robot execution. The eight channels are *recorded state features*, not eight grippers or eight independently controlled robot arms.

## Research questions

1. Can conditional rectified flow matching generate diverse motions close to held-out UR5 demonstrations while respecting exact endpoints?
2. What do specialist velocity predictors contribute when combined as a weighted ensemble?
3. Can a single temporal ConvNet student imitate the ensemble at **states reached by the student itself**, while retaining direct supervision from demonstrations?
4. What trade-offs emerge among flow MSE, rollout imitation RMSE, discrete smoothness, sample diversity, parameter count, and inference latency?

## Dataset and preprocessing

The notebook downloads and reads the numeric Berkeley Autolab UR5 dataset from Hugging Face:

```bash
hf download lerobot/berkeley_autolab_ur5 --repo-type dataset \
  --include "data/*/*.parquet" --include "meta/*.json" \
  --local-dir Data/berkeley_ur5
```

It loads Parquet files under `Data/berkeley_ur5/data/`, concatenates and sorts their frames by episode and frame index, and reads `observation.state` plus `task_index`.

| Property | Notebook observation |
| --- | --- |
| Frames | 97,939 |
| Episodes | 1,000 |
| Task IDs | 5 (`0`–`4`) |
| Dataset sampling rate | 5 FPS |
| Recorded state shape | `(8,)` |
| Recorded action shape | `(7,)` (not used as the prediction target) |
| Resampled trajectory | `(32, 8)` |
| Fixed-length dataset | `(1000, 32, 8)` |
| Task representation | Five-element one-hot vector |
| Validation split | 10% of episodes, seeded random split |

Each complete valid episode is **linearly interpolated in time to 32 waypoints**. State channels are standardized using per-channel means and standard deviations. The training input contains these normalized states and one-hot task IDs.

**Important validation caveat:** As currently written, the notebook calculates the normalization statistics over all resampled episodes **before** the train/validation split. This leaks aggregate validation-set statistics into preprocessing. A stricter evaluation should split episodes first and fit the scaler on training episodes alone.

### What are the eight channels?

The Parquet fields name the eight dimensions only through `observation.state`; the notebook labels them `motor_0` through `motor_7` when plotting. Their value ranges suggest this *working hypothesis*:

| Channels | Assumed interpretation for visualization |
| --- | --- |
| 0–2 | End-effector Cartesian position `(x, y, z)` |
| 3–6 | End-effector orientation quaternion (assumed `xyzw`) |
| 7 | Gripper opening / gripper state |

This mapping **has not been verified from authoritative state metadata**. It is used for the optional UR5 3D reconstruction; the flow network itself treats the data as **eight numeric channels** and learns all eight jointly. There is **one hypothesized gripper channel**, not eight grippers. The code does not multiply the input by eight: the observed trajectory tensor has shape **`[batch, 32, 8]`**, and the generated interior residual has shape **`[batch, 30, 8]`**.

## Model: conditional rectified flow

Let the normalized demonstration be `q ∈ R^(32×8)` with endpoints `q_start` and `q_goal`. The reference path is straight interpolation **in normalized recorded-state space**:

```text
q_ref(s) = (1-s) q_start + s q_goal,     s ∈ [0,1]
r_data(s) = q_demo(s) - q_ref(s),       for interior waypoints
```

The flow model generates only the **30 interior residuals**, ensuring that reconstruction inserts the start and goal states unchanged. Unlike a six-DOF joint-angle planner, this notebook **does not implement shortest-angle interpolation or quaternion-aware interpolation**; the reference is componentwise linear interpolation of eight standardized features.

For each training batch:

```text
r_noise ~ Normal(0, I)
t ~ Uniform(0, 1)
x_t = (1-t) r_noise + t r_data
v_target = r_data - r_noise
L_flow = MSE(v_theta(t, x_t, start, goal, task), v_target)
```

The network predicts a velocity for every channel at each interior waypoint. Its condition vector is:

```text
[sinusoidal flow-time embedding (32), start (8), goal (8), task one-hot (5)]
```

The condition is projected and broadcast across the 30 temporal positions. Temporal `Conv1d` layers and dilated residual blocks integrate information from neighboring and more distant waypoints.

### Sampling

The sampler begins with a new Gaussian tensor of shape `(samples, 30, 8)` and integrates the learned velocity field using **explicit Euler updates**:

```python
x = torch.randn(samples, 30, 8, device=device)
for k in range(ode_steps):
    t = k / ode_steps
    x = x + model(t, x, start, goal, task) / ode_steps
trajectory = decode_residual(x, start, goal)
```

The notebook's default inference sampler uses **200 ODE steps**; its batch benchmark uses **50 steps** for rollout metrics. Different noise draws can produce different outputs for the same start, goal, and task. These are stochastic **state trajectories**, not directly validated actuator commands.

## Training stages

### 1. Initial flow baseline

A temporal ConvNet is initially trained with the standard flow-matching loss. This first baseline uses `HIDDEN_DIM=256`, temporal dilations `(1, 2, 4, 8)`, and **500 epochs**, with AdamW, gradient clipping, and cosine learning-rate decay. It has **1,073,416 parameters**. This initial experiment is separate from the four 905,672-parameter specialists in the later benchmark.

### 2. Four specialist teachers

Four independent ConvNets share the same architecture: **192 channels**, six dilated residual blocks with dilations `(1, 2, 4, 8, 4, 2)`, and **905,672 parameters per teacher**. Each trains for **300 epochs**, with an independent optimizer and best-validation-objective checkpoint.

| Teacher | Objective in the code | Intended emphasis |
| --- | --- | --- |
| `rmse` | `L_flow + 0.35 × clean-state reconstruction RMSE` | Better reconstruction |
| `acceleration` | `L_flow + 0.30 × squared second-difference penalty` | Smoother predicted clean trajectories |
| `validation` | `L_flow` | Standard flow performance |
| `latency` / base | `L_flow` | Same-architecture controlled replicate; latency measured separately |

**Clarification:** Runtime is *not* part of the latency teacher's differentiable training objective. Nor does this experiment train an MLP or Transformer as part of the final ensemble; the final ensemble uses four temporal ConvNets.

### 3. Frozen weighted ensemble

After training, all four teachers are placed in evaluation mode and frozen. For the **same** flow state, time, and condition, each predicts an 8-channel velocity over the 30 interior waypoints. Those velocities are combined as:

```text
v_ensemble = 0.30 v_rmse
           + 0.30 v_acceleration
           + 0.30 v_validation
           + 0.10 v_latency
```

The ensemble averages **velocity predictions at every ODE step**, not four finished trajectories after sampling. Weights are manually specified; they are not learned by a gating network. The total ensemble contains **3,622,688 parameters**.

### 4. On-policy distillation into a student

The student uses the **same architecture and parameter count as any one specialist**: 905,672 parameters. It is initialized independently and trained for **300 epochs** using three complementary terms:

1. **On-policy velocity imitation:** Draw Gaussian residual noise, roll the current student forward a randomly chosen number of steps (0–15) on a 16-step Euler grid, and ask the *frozen teacher ensemble* to label that student-reached state. Penalize the difference between student and ensemble velocities.
2. **Demonstration-supervised flow matching:** Draw a fresh rectified-flow interpolation between Gaussian noise and real demonstration residuals, then train against its known target velocity. This retains direct supervision from observed motion.
3. **Smoothness regularization:** Decode a one-step student prediction into a full trajectory and penalize squared second finite differences along the waypoint dimension.

The notebook configures:

```text
L_student = L_on_policy_imitation
          + 1.00 L_demonstration_flow
          + 0.10 L_smoothness
```

Teacher predictions and on-policy state collection use `no_grad`: gradients update **the student only**; they are not propagated through the student rollout history or through the teachers. The checkpoint is selected using held-out standard flow loss.

## Evaluation

The notebook evaluates multiple aspects of trajectory generation rather than assigning a single success score:

- **Validation flow MSE:** Local error in the predicted rectified-flow velocity.
- **Rollout RMSE:** Generated trajectory versus a held-out demonstration in **normalized state space**. A valid alternative trajectory can still have nonzero RMSE.
- **Acceleration RMS:** Root-mean-squared second finite difference over waypoints, a smoothness proxy—not measured physical acceleration.
- **Endpoint RMSE:** Confirms that start/goal states are inserted exactly.
- **Sample diversity / predictive variance:** Differences between generated paths under independently sampled noise.
- **Latency:** Mean time for **one forward velocity-field evaluation**, after warm-up and CUDA synchronization where applicable. This is **not total 200-step trajectory-generation time**.
- **3D reconstruction:** An assumption-based inverse-kinematics visualization, not validation of the recorded robot's original joint configuration.

### Specialist and student benchmark

The following values are from the **attached notebook's printed comparison table** (rounded as shown there), not from the earlier obstacle-based README. Lower is better for the metrics below.

| Model | Parameters | Validation flow MSE ↓ | Rollout RMSE ↓ | Acceleration RMS ↓ | Forward pass latency (ms) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `latency` / base | 905,672 | **0.29846** | 0.5668 | 0.2635 | **0.706** |
| Ensemble | 3,622,688 | 0.29974 | 0.5593 | 0.2474 | 2.871 |
| `validation` | 905,672 | 0.30387 | 0.5651 | 0.2650 | 0.708 |
| `rmse` | 905,672 | 0.30406 | 0.5594 | 0.2656 | 0.726 |
| `acceleration` | 905,672 | 0.31533 | 0.5692 | **0.2335** | **0.706** |
| Student | 905,672 | 0.31862 | **0.5571** | 0.2401 | 0.713 |

The distilled student has **one-quarter of the ensemble's parameters**, with a roughly **4× shorter measured forward-pass latency** in this run. It recorded the lowest rollout RMSE in this table, although its validation flow MSE is worse than every teacher and the ensemble. The acceleration-specialist teacher gave the lowest acceleration RMS. These are complementary metrics, not interchangeable measures of robot task success.

### Additional held-out comparison in recorded units

On **one held-out example** with eight generated samples, using the same initial noise for teacher and student, the notebook reports:

| Model | Demo RMSE ↓ | Acceleration RMS ↓ | Jerk RMS ↓ | Endpoint RMSE |
| --- | ---: | ---: | ---: | ---: |
| Teacher ensemble | 0.09188 | 0.10308 | 0.17883 | 0.00000000 |
| Student | **0.08715** | 0.10920 | 0.19070 | 0.00000000 |
| Straight interpolation | 0.24705 | **0.00000** | **0.00000** | 0.00000000 |

This shows why smoothness must be interpreted carefully: straight interpolation has zero discrete acceleration and jerk by construction, yet is much less similar to the demonstration. These values are in **original recorded feature units**; because the eight channels may represent different quantities, aggregating them into a single RMSE has interpretability limitations.

## Visualizations

The notebook saves plots and interactive demonstrations under `Results/`, including:

| Output | What it shows |
| --- | --- |
| `01_recorded_trajectories.png` | Raw trajectories over time |
| `02_initial_flow_training.png` | Baseline training/validation loss |
| `03_initial_flow_samples.png` | Six samples and a held-out demonstration, for all eight channels |
| `04_initial_flow_diagnostics.png` | Imitation, endpoints, smoothness |
| `05_robot_reconstruction_animation.html` | Real vs generated UR5 3D reconstruction using assumed state semantics and IK |
| `06_robot_ik_residuals.png` | IK position/orientation fit diagnostics |
| `07_specialist_training_curves.png` | Teacher training and validation curves |
| `08_student_training_curves.png` | Student loss components |
| `09_ensemble_student_rollout.png` | Paired ensemble/student trajectories |
| `10_model_comparison.csv` | Model-level benchmark |
| `11_heldout_comparison.csv` | Held-out sample statistics |
| `13_student_3d.html` | Interactive student reconstruction |

The reconstruction assumes channels 0–2 encode XYZ, 3–6 a quaternion, and 7 the gripper state. Inverse kinematics recovers one plausible six-joint UR5 pose at each waypoint, warm-started from the preceding waypoint. The notebook reports 32/32 IK-accepted frames for one real and one generated trajectory under its configured thresholds, but this **does not prove ground-truth joint recovery, joint-limit safety, or collision-free execution**.

## Setup and reproduction

The attached notebook installs the required packages in its kernel:

```python
%pip install -q numpy pandas pyarrow matplotlib plotly scipy tqdm torch
```

For a typical local environment:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install numpy pandas pyarrow matplotlib plotly scipy tqdm torch huggingface_hub
```

Download the dataset using the command above, then open and run the Berkeley UR5 flow-matching notebook from the project root or `Code/` directory. The notebook resolves its project root based on whether the current directory is named `Code`, writes model checkpoints to `runs/`, and exports plots/tables to `Results/`.

A CUDA-enabled PyTorch environment is recommended for teacher and student training. The notebook uses a fixed random seed of `7` for Python, NumPy, and PyTorch and a seeded 90/10 episode split. Experiment results still depend on hardware and stochastic execution; reproduce latency on your own deployment device.

## What this experiment does **not** demonstrate

- **No occupancy-map conditioning:** The notebook does not read or encode any obstacle occupancy grid.
- **No RRT demonstrations:** Data come from recorded Berkeley robot episodes, not RRT-generated collision-free paths.
- **No collision-free rate:** Scene meshes, obstacles, and simulator contacts are absent; the notebook reports `N/A` instead of inventing a collision success rate.
- **No verified six-joint training target:** The model learns eight anonymous recorded channels. Six UR5 joint configurations appear only in the optional assumption-based IK reconstruction.
- **No eight grippers:** Eight features should not be interpreted as eight gripper devices.
- **No shortest-angle/quaternion-constrained flow:** The residual reference is linear in normalized feature coordinates; generated quaternions are normalized for visualization only.
- **No comprehensive memorization test:** A held-out episode split, visual comparisons, and train/validation curves offer initial generalization diagnostics, but the notebook does not run nearest-neighbor memorization checks, train-versus-test generation comparisons, duplicate detection, or independently repeated train/test evaluations. The preprocessing leakage noted above also limits a strict out-of-sample claim.
- **No physical robot deployment:** Demonstration similarity, IK fit, and inference timing do not establish real-world task success or safety.

## Suggested next experiments

To strengthen the results, verify channel semantics from dataset documentation; split episodes before fitting normalization; quantify generated-vs-training nearest-neighbor distances; evaluate multiple random seeds and unseen start/goal combinations; add task-success and physically grounded collision checks if scene geometry becomes available; and benchmark **full ODE rollout time**, not just one neural-network call. A genuinely obstacle-conditioned variant would require obstacle representations in the dataset or a simulator, an occupancy-map encoder, and appropriate collision-aware evaluation.

## License

MIT — see the repository's `LICENSE` file.

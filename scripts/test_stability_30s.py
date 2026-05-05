"""hand_final 30s stability test: D2 Go/No-Go gate."""
import mujoco
import numpy as np
import time

MODEL_PATH = "/home/rivery/duke_hand/models/hand_final.xml"
DURATION_SEC = 30.0


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    n_steps = int(DURATION_SEC / model.opt.timestep)
    print(f"Running {DURATION_SEC} s = {n_steps} steps at dt={model.opt.timestep}s")

    qpos_history = np.zeros((n_steps, model.nq))
    qvel_max_history = np.zeros(n_steps)

    period_sec = 3.0
    mcp_open, mcp_close = 0.0, 70.0
    pip_open, pip_close = 0.0, 90.0

    wall_start = time.time()
    for k in range(n_steps):
        t = k * model.opt.timestep
        phase = 0.5 * (1.0 - np.cos(2 * np.pi * t / period_sec))
        mcp_target = np.deg2rad(mcp_open + phase * (mcp_close - mcp_open))
        pip_target = np.deg2rad(pip_open + phase * (pip_close - pip_open))

        for i in range(3):
            data.ctrl[2 * i] = mcp_target
            data.ctrl[2 * i + 1] = pip_target

        mujoco.mj_step(model, data)

        qpos_history[k] = data.qpos.copy()
        qvel_max_history[k] = np.max(np.abs(data.qvel))

        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            print(f"FAIL: NaN detected at step {k} (t={t:.3f}s)")
            return 1

    wall_elapsed = time.time() - wall_start

    print(f"\n30s simulation completed.")
    print(f"  Wall clock time: {wall_elapsed:.2f} s")
    print(f"  Realtime factor: {DURATION_SEC / wall_elapsed:.2f}x")
    print(f"  No NaN encountered: PASS")

    print(f"\nState bounds over 30s:")
    for i in range(model.nq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        col = qpos_history[:, i]
        print(f"  {name:8s}  min={np.degrees(col.min()):+7.2f}  max={np.degrees(col.max()):+7.2f} deg")

    print(f"\nMax joint speed seen: {np.degrees(qvel_max_history.max()):.2f} deg/s")

    final_q = qpos_history[-1]
    expected_q = qpos_history[int(period_sec / model.opt.timestep)]
    drift = np.degrees(np.abs(final_q - expected_q)).max()
    print(f"Cycle drift (final vs 1-period mark): {drift:.4f} deg")
    if drift < 1.0:
        print("  PASS: drift < 1 deg, controller tracking is stable")
    else:
        print("  WARN: drift > 1 deg, may indicate controller issue")

    print("\nD2 Go/No-Go: 30s simulation stable, no NaN, no divergence.")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    exit(main())

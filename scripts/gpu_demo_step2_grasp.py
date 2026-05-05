"""GPU multi-env grasp test: 64 hands grasp in parallel, dump qpos history."""
import time
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

MODEL_PATH = "/home/rivery/duke_hand/models/hand_final.xml"
NWORLD = 64
N_STEPS = 1500
PERIOD_SEC = 3.0
DT = 0.002


def main():
    mjm = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mjd = mujoco.MjData(mjm)
    print(f"CPU model: nq={mjm.nq}, nu={mjm.nu}")

    print(f"\nPutting on GPU with nworld={NWORLD}...")
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=NWORLD)
    print(f"GPU ctrl shape: {d.ctrl.shape}, qpos shape: {d.qpos.shape}")

    qpos_history = np.zeros((N_STEPS, NWORLD, mjm.nq), dtype=np.float32)
    ctrl_buffer = np.zeros((NWORLD, mjm.nu), dtype=np.float32)

    print(f"\nRunning {N_STEPS} steps ({N_STEPS * DT:.1f} s sim time)...")
    t_start = time.time()
    for k in range(N_STEPS):
        t = k * DT
        phase = 0.5 * (1.0 - np.cos(2 * np.pi * t / PERIOD_SEC))
        mcp_target = np.deg2rad(0.0 + phase * 70.0)
        pip_target = np.deg2rad(0.0 + phase * 90.0)

        for i in range(3):
            ctrl_buffer[:, 2 * i] = mcp_target
            ctrl_buffer[:, 2 * i + 1] = pip_target
        d.ctrl = wp.from_numpy(ctrl_buffer, dtype=wp.float32, device="cuda:0")

        mjw.step(m, d)

        qpos_history[k] = d.qpos.numpy()

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.2f} s wall time")
    print(f"Sim time: {N_STEPS * DT:.1f} s, throughput: {N_STEPS * NWORLD / elapsed:.0f} steps/s")

    final = qpos_history[-1]
    print(f"\nFinal qpos hand 0 (deg): {np.degrees(final[0])}")
    print(f"Final qpos hand 63 (deg): {np.degrees(final[63])}")

    np.save("/home/rivery/duke_hand/scripts/qpos_history.npy", qpos_history)
    print(f"\nSaved qpos_history to scripts/qpos_history.npy")
    print(f"Shape: {qpos_history.shape} ({qpos_history.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

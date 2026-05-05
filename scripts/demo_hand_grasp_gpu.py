"""GPU multi-hand grasp demo: 64 hands grasping in parallel, rendered live.

Physics is computed by mujoco-warp on GPU.
Visualization is done by a tiled display model on CPU.
The display model has no actuators and no gravity—it's just an animation skeleton
driven by GPU-computed qpos.
"""
import time
import mujoco
import mujoco.viewer
import mujoco_warp as mjw
import numpy as np
import warp as wp

PHYSICS_MODEL = "/home/rivery/duke_hand/models/hand_final.xml"
DISPLAY_MODEL = "/home/rivery/duke_hand/models/hand_grid_64.xml"
NWORLD = 64
PERIOD_SEC = 3.0
DT = 0.002


def main():
    print("=" * 60)
    print("GPU multi-hand grasp demo")
    print("=" * 60)

    print("\nLoading physics model (single hand) for GPU...")
    mjm = mujoco.MjModel.from_xml_path(PHYSICS_MODEL)
    mjd = mujoco.MjData(mjm)
    print(f"  nq={mjm.nq}, nu={mjm.nu}")

    print(f"\nPutting model on GPU with nworld={NWORLD}...")
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=NWORLD)
    print(f"  GPU ctrl: {d.ctrl.shape}, qpos: {d.qpos.shape}")

    print("\nLoading display model (64-hand grid) for CPU rendering...")
    disp_model = mujoco.MjModel.from_xml_path(DISPLAY_MODEL)
    disp_data = mujoco.MjData(disp_model)
    assert disp_model.nq == NWORLD * mjm.nq, \
        f"display nq={disp_model.nq} != {NWORLD}*{mjm.nq}"
    print(f"  Display nq={disp_model.nq}")

    ctrl_buffer = np.zeros((NWORLD, mjm.nu), dtype=np.float32)

    print("\nLaunching viewer. Close window or Ctrl+C to exit.")
    print("Physics on GPU (cuda:0). Visualization on CPU.")

    with mujoco.viewer.launch_passive(disp_model, disp_data) as viewer:
        sim_start = time.time()
        step_count = 0
        last_print = sim_start

        while viewer.is_running():
            t = time.time() - sim_start

            phase = 0.5 * (1.0 - np.cos(2 * np.pi * t / PERIOD_SEC))
            mcp_target = np.deg2rad(0.0 + phase * 70.0)
            pip_target = np.deg2rad(0.0 + phase * 90.0)
            for i in range(3):
                ctrl_buffer[:, 2 * i] = mcp_target
                ctrl_buffer[:, 2 * i + 1] = pip_target
            d.ctrl = wp.from_numpy(ctrl_buffer, dtype=wp.float32, device="cuda:0")

            mjw.step(m, d)
            step_count += 1

            qpos_gpu = d.qpos.numpy()
            disp_data.qpos[:] = qpos_gpu.reshape(-1)
            mujoco.mj_forward(disp_model, disp_data)
            viewer.sync()

            now = time.time()
            if now - last_print > 2.0:
                throughput = step_count * NWORLD / (now - sim_start)
                print(f"  t={t:.1f}s, GPU throughput: {throughput:.0f} steps/s "
                      f"({step_count} steps, {NWORLD} hands)")
                last_print = now

            sim_target_time = step_count * DT
            sleep_for = sim_target_time - (time.time() - sim_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()

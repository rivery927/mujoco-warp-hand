"""GPU multi-hand grasp display: replay qpos_history on 64 tiled hands."""
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "/home/rivery/duke_hand/models/hand_grid_64.xml"
QPOS_PATH  = "/home/rivery/duke_hand/scripts/qpos_history.npy"
DT = 0.002


def main():
    print("Loading qpos history...")
    history = np.load(QPOS_PATH)
    n_steps, n_hands, nq_per_hand = history.shape
    print(f"  shape: {history.shape}")
    print(f"  {n_steps} steps, {n_hands} hands, {nq_per_hand} DoF/hand")

    print("\nLoading display model...")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print(f"  Display model: nq={model.nq} (expect {n_hands * nq_per_hand})")
    assert model.nq == n_hands * nq_per_hand, "nq mismatch"

    print("\nLaunching viewer. Close window or Ctrl+C to exit.")
    print("Replay loops indefinitely. Each loop = 3.0 s sim time.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        loop_start = time.time()
        while viewer.is_running():
            elapsed = time.time() - loop_start
            k = int(elapsed / DT) % n_steps

            data.qpos[:] = history[k].reshape(-1)
            mujoco.mj_forward(model, data)
            viewer.sync()

            next_step_time = (k + 1) * DT
            sleep_for = next_step_time - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()

"""hand_final coordinated grasp demo: open-close cycle on all 3 fingers."""
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH =  "/home/rivery/duke_hand/models/hand_final.xml"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("Starting viewer. Close window or Ctrl+C to exit.")
    print("Coordinated grasp: 3 fingers open/close in sync, period = 3 sec.")

    mcp_open_deg, mcp_close_deg = 0.0, 70.0
    pip_open_deg, pip_close_deg = 0.0, 90.0
    period_sec = 3.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start = time.time()
        while viewer.is_running():
            t = time.time() - sim_start

            phase = 0.5 * (1.0 - np.cos(2 * np.pi * t / period_sec))
            mcp_target_deg = mcp_open_deg + phase * (mcp_close_deg - mcp_open_deg)
            pip_target_deg = pip_open_deg + phase * (pip_close_deg - pip_open_deg)

            mcp_target_rad = np.deg2rad(mcp_target_deg)
            pip_target_rad = np.deg2rad(pip_target_deg)

            for i in range(3):
                data.ctrl[2 * i] = mcp_target_rad
                data.ctrl[2 * i + 1] = pip_target_rad

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next = data.time - (time.time() - sim_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()

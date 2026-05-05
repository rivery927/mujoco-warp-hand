"""finger_v1 dynamic demo: sin-wave drive both joints, visualize in viewer."""
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "/home/rivery/duke_hand/models/finger_v1.xml"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("Starting viewer. Close window or Ctrl+C to exit.")
    print("MCP: 0..60 deg @ 0.5 Hz   PIP: 10..90 deg @ 1.0 Hz")
    print("(Targets are converted to radians internally.)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start = time.time()
        while viewer.is_running():
            t = time.time() - sim_start

            mcp_target_deg = 30 + 30 * np.sin(2 * np.pi * 0.5 * t)
            pip_target_deg = 50 + 40 * np.sin(2 * np.pi * 1.0 * t)

            data.ctrl[0] = np.deg2rad(mcp_target_deg)
            data.ctrl[1] = np.deg2rad(pip_target_deg)

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next = data.time - (time.time() - sim_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()

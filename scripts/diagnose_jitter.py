"""量化诊断关节抖动幅度."""
import mujoco
import numpy as np

MODEL_PATH = "/home/rivery/duke_hand/models/finger_v1.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

data.ctrl[:] = 0
for _ in range(1000):
    mujoco.mj_step(model, data)

N = 1000
mcp_history = np.zeros(N)
pip_history = np.zeros(N)
for i in range(N):
    mujoco.mj_step(model, data)
    mcp_history[i] = np.degrees(data.qpos[0])
    pip_history[i] = np.degrees(data.qpos[1])

print("Stable-state joint angle statistics over 2s:")
print(f"  MCP  mean={mcp_history.mean():+.6f} deg  std={mcp_history.std():.6f}  ptp={np.ptp(mcp_history):.6f}")
print(f"  PIP  mean={pip_history.mean():+.6f} deg  std={pip_history.std():.6f}  ptp={np.ptp(pip_history):.6f}")

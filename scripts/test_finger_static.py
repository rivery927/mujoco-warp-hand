"""finger_v1 static test: model loading, mass check, static holding torque."""
import mujoco
import numpy as np

MODEL_PATH = "/home/rivery/duke_hand/models/finger_v1.xml"


def main():
    print("=" * 60)
    print("Load model")
    print("=" * 60)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"  nq    = {model.nq}        (expect 2: MCP + PIP)")
    print(f"  nv    = {model.nv}        (expect 2)")
    print(f"  nbody = {model.nbody}     (expect 4)")
    print(f"  nu    = {model.nu}        (expect 2)")
    print(f"  timestep = {model.opt.timestep} s")

    print("\n" + "=" * 60)
    print("Mass check (HL-3915 = 35.8 g/servo, design = 40 g/segment)")
    print("=" * 60)
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  body[{i}] {name:12s} mass = {model.body_mass[i]*1000:.1f} g")

    print("\n" + "=" * 60)
    print("Joint range check")
    print("=" * 60)
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = np.degrees(model.jnt_range[i])
        print(f"  joint[{i}] {name:6s} range = [{lo:6.1f}, {hi:6.1f}] deg")

    print("\n" + "=" * 60)
    print("Static holding torque test (finger horizontal, target=0)")
    print("=" * 60)
    data.ctrl[:] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)

    print(f"  steady-state joint angles:")
    print(f"    MCP = {np.degrees(data.qpos[0]):+.4f} deg")
    print(f"    PIP = {np.degrees(data.qpos[1]):+.4f} deg")
    print(f"  actuator force required:")
    print(f"    MCP = {data.actuator_force[0]:+.4f} N.m")
    print(f"    PIP = {data.actuator_force[1]:+.4f} N.m")

    max_torque = max(abs(data.actuator_force[0]), abs(data.actuator_force[1]))
    margin = 1.39 / max_torque if max_torque > 1e-6 else float("inf")
    print(f"\n  HL-3915 stall torque = 1.39 N.m")
    print(f"  Max torque used      = {max_torque:.4f} N.m")
    print(f"  Safety margin        = {margin:.1f}x  (design predicted ~115x)")

    assert not np.any(np.isnan(data.qpos)), "qpos NaN, check inertial"
    assert not np.any(np.isnan(data.qvel)), "qvel NaN"
    print("\nOK: physics state stable, no NaN")


if __name__ == "__main__":
    main()

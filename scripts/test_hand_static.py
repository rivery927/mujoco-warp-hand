"""hand_final static test: 6 DoF model verification."""
import mujoco
import numpy as np

MODEL_PATH = "/home/rivery/duke_hand/models/hand_final.xml"


def main():
    print("=" * 60)
    print("Load hand_final")
    print("=" * 60)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"  nq    = {model.nq}        (expect 6)")
    print(f"  nv    = {model.nv}        (expect 6)")
    print(f"  nbody = {model.nbody}     (expect 8)")
    print(f"  nu    = {model.nu}        (expect 6)")

    print("\nJoint inventory:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = np.degrees(model.jnt_range[i])
        print(f"  joint[{i}] {name:8s} range = [{lo:6.1f}, {hi:6.1f}] deg")

    print("\nMass inventory:")
    total_mass = 0.0
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        m = model.body_mass[i]
        total_mass += m
        print(f"  body[{i}] {name:14s} mass = {m*1000:6.1f} g")
    print(f"  TOTAL hand mass = {total_mass*1000:.1f} g")

    print("\nStatic holding torque test (all joints at 0):")
    data.ctrl[:] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)

    print("  Steady-state errors (deg):")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"    {name:8s} = {np.degrees(data.qpos[i]):+.4f}")

    print("\n  Actuator forces (N.m):")
    max_torque = 0.0
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        f = abs(data.actuator_force[i])
        max_torque = max(max_torque, f)
        print(f"    {name:12s} = {data.actuator_force[i]:+.4f}")

    margin = 1.39 / max_torque if max_torque > 1e-6 else float("inf")
    print(f"\n  Max torque used = {max_torque:.4f} N.m")
    print(f"  Safety margin   = {margin:.1f}x  (HL-3915 stall = 1.39 N.m)")

    assert not np.any(np.isnan(data.qpos)), "qpos NaN"
    assert not np.any(np.isnan(data.qvel)), "qvel NaN"
    print("\nOK: 6-DoF physics state stable, no NaN")


if __name__ == "__main__":
    main()

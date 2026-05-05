"""GPU multi-env smoke test: put hand on GPU, step 100 times, fetch qpos."""
import mujoco
import mujoco_warp as mjw
import numpy as np

MODEL_PATH = "/home/rivery/duke_hand/models/hand_final.xml"
NWORLD = 64


def main():
    mjm = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mjd = mujoco.MjData(mjm)
    print(f"CPU model loaded: nq={mjm.nq}, nu={mjm.nu}")

    print(f"\nPutting model + {NWORLD} data copies on GPU...")
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=NWORLD)
    print(f"GPU data shape: qpos={d.qpos.shape}, ctrl={d.ctrl.shape}")

    print(f"\nStepping {NWORLD} hands x 100 steps on GPU...")
    for _ in range(100):
        mjw.step(m, d)

    qpos_all = d.qpos.numpy()
    print(f"\nAfter 100 steps:")
    print(f"  qpos shape (CPU): {qpos_all.shape}")
    print(f"  hand 0 qpos (rad): {qpos_all[0]}")
    print(f"  hand 0 qpos (deg): {np.degrees(qpos_all[0])}")
    print(f"  all hands identical? {np.allclose(qpos_all[0], qpos_all[1])}")


if __name__ == "__main__":
    main()

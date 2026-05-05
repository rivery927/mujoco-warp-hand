"""Generate a multi-hand display XML by tiling hand_final 8x8."""
import os

GRID_N = 8                    # 8x8 = 64 hands
SPACING = 0.30                # 30 cm between hand centers
OUT_PATH = "/home/rivery/duke_hand/models/hand_grid_64.xml"

HEADER = """<mujoco model="hand_grid_64">
  <compiler angle="degree" autolimits="true" inertiafromgeom="true"/>
  <option gravity="0 0 0" timestep="0.002"/>

  <default>
    <joint type="hinge" axis="0 1 0" damping="0.1"/>
    <geom type="capsule" rgba="0.7 0.75 0.85 1"/>
  </default>

  <worldbody>
    <light pos="0.0 0.0 3.0" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2.0 2.0 0.05" rgba="0.85 0.9 0.85 1"/>
"""

FOOTER = """  </worldbody>
</mujoco>
"""


def hand_block(hid, x, y):
    """Generate one hand's XML at offset (x, y)."""
    tag = f"h{hid}"
    return f"""    <body name="{tag}_palm" pos="{x:.3f} {y:.3f} 0.15">
      <geom name="{tag}_palm_g" type="box" size="0.05 0.05 0.012" rgba="0.4 0.4 0.45 1" mass="0.12"/>
      <body name="{tag}_prox_0" pos="0.05 0.030 0">
        <joint name="{tag}_mcp_0" range="-10 90"/>
        <geom name="{tag}_prox_0_g" fromto="0 0 0 0.05 0 0" size="0.008" mass="0.04"/>
        <body name="{tag}_dist_0" pos="0.05 0 0">
          <joint name="{tag}_pip_0" range="0 110"/>
          <geom name="{tag}_dist_0_g" fromto="0 0 0 0.035 0 0" size="0.007" mass="0.04"/>
        </body>
      </body>
      <body name="{tag}_prox_1" pos="0.05 0 0">
        <joint name="{tag}_mcp_1" range="-10 90"/>
        <geom name="{tag}_prox_1_g" fromto="0 0 0 0.05 0 0" size="0.008" mass="0.04"/>
        <body name="{tag}_dist_1" pos="0.05 0 0">
          <joint name="{tag}_pip_1" range="0 110"/>
          <geom name="{tag}_dist_1_g" fromto="0 0 0 0.035 0 0" size="0.007" mass="0.04"/>
        </body>
      </body>
      <body name="{tag}_prox_2" pos="0.05 -0.030 0">
        <joint name="{tag}_mcp_2" range="-10 90"/>
        <geom name="{tag}_prox_2_g" fromto="0 0 0 0.05 0 0" size="0.008" mass="0.04"/>
        <body name="{tag}_dist_2" pos="0.05 0 0">
          <joint name="{tag}_pip_2" range="0 110"/>
          <geom name="{tag}_dist_2_g" fromto="0 0 0 0.035 0 0" size="0.007" mass="0.04"/>
        </body>
      </body>
    </body>
"""


def main():
    parts = [HEADER]
    half = (GRID_N - 1) * SPACING / 2.0
    for row in range(GRID_N):
        for col in range(GRID_N):
            hid = row * GRID_N + col
            x = -half + col * SPACING
            y = -half + row * SPACING
            parts.append(hand_block(hid, x, y))
    parts.append(FOOTER)

    with open(OUT_PATH, "w") as f:
        f.write("".join(parts))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB)")
    print(f"Grid: {GRID_N}x{GRID_N} = {GRID_N**2} hands")
    print(f"Spacing: {SPACING} m, total span: {(GRID_N-1)*SPACING:.2f} m")


if __name__ == "__main__":
    main()

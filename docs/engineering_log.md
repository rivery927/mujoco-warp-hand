# Engineering Log

This file records design decisions, bugs encountered, and fixes during development.
Each entry is potential material for the final presentation.

## D1 (2026-05-04)

### Story 1: Joint oscillation tuning
- Initial parameters: kp=250, damping=0.01
- Symptom: visible jitter, PIP joint had peak-to-peak amplitude of 28 degrees
- Diagnosis: ran `diagnose_jitter.py`, recorded MCP ptp=14.6 deg, PIP ptp=28.1 deg
- Root cause: underdamped second-order system; controller too stiff vs. damping
- Fix: kp 250 -> 100, damping 0.01 -> 0.1
- Result: PIP ptp 28.1 deg -> 0.0 deg (limit cycle eliminated)
- Lesson: HL-3915 internal PID gains are factory-tuned for hardware mass / inertia.
  Simulation-side kp / damping are the *equivalent model* of the servo loop, not
  copies of the hardware PID values. They must be tuned against the simulated
  inertia.

### Story 2: Static holding torque - simulation vs. hand calculation
- Hand-calculated margin (one-segment, simplified): ~115x
- Simulated margin (full open-chain dynamics): 38.3x
- Difference explanation: simulation includes torque transmitted from distal
  segment through PIP, plus damping coupling. The simulated number is the more
  accurate engineering value.
- Conclusion: 38.3x margin still confirms direct-drive feasibility for HL-3915.

### Story 3: ctrlrange units inconsistency
- Symptom: in dynamic demo, finger always pinned at MCP=90 deg instead of
  oscillating
- Diagnosis: printed raw ctrlrange and jnt_range from compiled model
  - jnt_range: stored in radians (auto-converted from degrees)
  - ctrlrange: stored as raw XML value (in degrees as written)
- Root cause: the position actuator interprets ctrl signal in the joint's native
  unit (radians). Writing ctrl=30 means "drive to 30 rad" (~1718 deg), clipped
  by joint range to 90 deg.
- Fix: convert all targets to radians via np.deg2rad() in the control script,
  and update XML ctrlrange to radian values for consistency.
- Lesson: MuJoCo's `compiler angle="degree"` is a parsing-time convenience; the
  runtime is unit-agnostic and treats all signals in radians.

Mon May  4 09:27:23 PM EDT 2026 - Project functionally complete. Email requirements 6/7 (#7 = in-person).

## D2 verification — kinematic connectivity check

After visual inspection raised doubt about whether the three fingers
were truly children of the palm body or just visually overlapping
floating bodies, ran two diagnostic checks:

1. body_parentid array confirmed:
   - palm.parent = world
   - proximal_{0,1,2}.parent = palm
   - distal_{i}.parent = proximal_{i}

2. Moved palm.body_pos from (0,0,0.15) to (0.5,0,0.5) and re-ran
   forward kinematics. All three fingertip positions translated by
   exactly (+0.5, 0, +0.35), matching the palm displacement to
   floating-point precision.

Conclusion: The hand is one rigid kinematic tree, not three independent
floating fingers. Visual gap at palm-finger interface is a rendering
artifact, not a structural disconnect.

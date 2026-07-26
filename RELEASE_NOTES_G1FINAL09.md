# G1FINAL-09: Language-Routed G1 Pick-and-Place

`G1FINAL-09` is the reproducible task-level demo release for this project. A
learned `G1LANG-34` language router selects the requested object from the
rendered MuJoCo scene. A classical inverse-kinematics, interpolation, and
assisted-grasp controller then executes the pick-and-place trajectory.

## Results

- 9/9 correct language routes.
- 9/9 controller successes.
- 9/9 strict hybrid task successes.
- Three object-slot permutations with fresh position jitter.
- Mean joint-limit violation fraction: `0.000441`.
- Minimum router top-1 probability: `0.999798`.
- Every released video has 182 frames at 640x480 and 15 FPS.

## Released Evidence

- `g1final09_red_triangle.mp4`
- `g1final09_yellow_rod.mp4`
- `g1final09_green_cube.mp4`
- `g1final09_summary.json`

The JSON summary contains all nine episode routes and task outcomes. The three
videos provide one representative execution for each target object.

## Claim Boundary

This release demonstrates learned language routing followed by a privileged
classical controller with assisted grasp. It is not end-to-end learned SmolVLA
control, a physical friction-only grasp, or a successful Isaac Sim replay. The
learned specialist plus World Model/MPC task gate remains a documented negative
result, and the hybrid Isaac replay remains blocked by an RTX
`ERROR_DEVICE_LOST` environment failure.

## Reproduce

```bash
bash /home/ubuntu/G1-UpperBody/scripts/run_g1_final09.sh
```

The command refuses to overwrite an existing result directory.

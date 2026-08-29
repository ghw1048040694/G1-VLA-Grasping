# G1 VLA Architecture

G1 is currently paused. The repository retains historical manipulation,
world-model, and sim2sim experiments, but they share one lifecycle boundary:

```text
doctor -> prepare -> smoke -> collect -> audit
                           -> train-vla -> train-world-model -> eval -> play -> report
```

`src/g1_vla/` owns the installable lifecycle package. `g1-vla doctor`,
`prepare`, `smoke`, and `report` are package-native and do not launch a
simulator. The other commands record a run context and invoke the selected
historical Python entry point through `runtime.compat`; stdout/stderr and the
exit code are preserved under the same run directory.

The existing scripts are not deleted in this stage because they are the only
reproducibility boundary for historical G1 results. New functionality should
be moved into `src/g1_vla/` first, then exposed through `g1_vla.cli`. The single
experiment journal remains `../../notes/Lerobot.md`.

## Runtime Contract

```text
.runtime/runs/G1-<timestamp>/
  config.json
  command.txt
  status.json
  summary.json
  datasets/
  checkpoints/
  videos/
  logs/
```

No command in this lifecycle layer changes policy inputs, adds privileged
object state, or claims a grasp result. The G1 end-to-end target remains
unmet, as recorded in the workspace journal.

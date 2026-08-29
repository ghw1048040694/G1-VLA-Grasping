# G1 Script Compatibility

The user-facing entry point is `g1-vla`. Install the lightweight lifecycle
package in the G1 environment:

```bash
python -m pip install -e . --no-deps
```

Use the unified commands:

```bash
g1-vla doctor
g1-vla prepare
g1-vla smoke
g1-vla collect --help
g1-vla train-vla --help
g1-vla train-world-model --help
g1-vla eval --help
g1-vla play --help
g1-vla report
```

Every non-dry run writes `.runtime/runs/G1-*/` with the command, configuration,
status, summary, and streamed logs. The mapped Python scripts remain
compatibility implementations while the G1 project is paused; this migration
does not start Genesis, IsaacGym, training, or evaluation automatically.

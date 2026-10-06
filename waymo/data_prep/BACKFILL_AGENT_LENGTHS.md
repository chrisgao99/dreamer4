# Incremental measured-length enrichment

`backfill_waymo_agent_lengths.py` preserves an existing dataset's train/val
membership, filenames, agent ordering and every existing NPZ feature. It reads
the source manifest's TFRecord shard/record index, seeks over unused records,
and parses only selected scenarios. Each selected record is shared across all
of its OOI-centered samples. It verifies scenario ID, source record/path,
selected source indices, track IDs, agent types and current validity.

The added `agent_lengths` is float32, in metres, in selected-agent order, from
raw `state/current/length`. Padding gets zero. An active vehicle/cyclist without
a finite positive measured length stops the job; no proxy sizes are invented.
Inactive selected tracks preserve the raw current-frame value (which may be
invalid); downstream execution validates lengths only for active wheeled agents.

An output NPZ is a byte copy of the original ZIP, with `agent_lengths.npy`
appended. Old compressed array payloads are not recomputed or recompressed.
ZIP member metadata and the appended length array are checked before atomic
publication. `--resume` rechecks existing outputs against the source and raw
lengths; it does not blindly skip existing filenames. Inputs are read-only and
the output must be a separate directory. A lock prevents concurrent writers.

```
python waymo/data_prep/backfill_waymo_agent_lengths.py \
  --source_dir data/waymo_vector_dataset_ooi_centered_50k \
  --output_dir data/waymo_vector_dataset_ooi_centered_50k_with_lengths \
  --workers 4 --resume
```

`--max_shards 1` is an explicit smoke-test option; do not use it for the complete
production dataset. Output status/config record the selected subset.

Output contains `enrichment_config.json`, `enrichment_status.json`, train/val
NPZs and (only upon successful processing of all requested samples) `manifest.csv`.
The rewritten manifest preserves every original column, changes `npz_path` to
the new output, and adds `source_npz_path`. No old holonomic action statistics are
copied. No GPU work, calibration, neural training or evaluation is launched by
this tool. A later type-aware training run still needs rho calibration and fresh
action statistics.

Prepared background wrapper: `runs/agent_lengths_20260924/run.sh`, using four CPU
workers, a lower scheduling priority and no visible CUDA devices. It writes
`run.log` and an `exit_code` file alongside the wrapper. A tmux session keeps it
running after terminal disconnect/screen lock. The dataset status JSON reports
completed shards, written/verified file counts, elapsed time and failures.

Validation before launch: 5 automated CPU tests passed, including exact old NPY
payload preservation, source immutability, alignment failures, corrupt-record
handling, resume verification and failure status. A real first-shard smoke test
processed 56 NPZs (50 train, 6 val); all 1,848 original NPY payloads were
byte-identical. Full 50,000-sample processing was launched in tmux session
`waymo_lengths_20260924`; consult the status JSON/log for current completion.

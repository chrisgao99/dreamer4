# Waymo Vector Tokenizer Commands

Run commands from the Dreamer 4 repo root:

```bash
cd /p/yufeng/tri30/dreamer4
```

Use the project Python:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python
```

## 1. Inspect One Raw Waymo TFRecord

This reads one scenario from a Waymo motion tf.Example TFRecord and prints the filtered tensor shapes.

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/inspect_waymo_vector_filter.py \
  /p/liverobotics/waymo_open_dataset_motion/tf_example/training/training_tfexample.tfrecord-00082-of-01000 \
  --max_records 1 \
  --num_agents 16
```

Useful options:

```bash
--num_agents 16
--num_agents 32
--agent_distance_threshold 80
--map_distance_threshold 100
--max_map_polylines 256
--max_points_per_polyline 20
--history_only_selection
```

## 2. Convert Raw Waymo TFRecord To Filtered NPZ Files

This creates one `.npz` file per scenario.

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/waymo_vector_filter.py \
  /p/liverobotics/waymo_open_dataset_motion/tf_example/training/training_tfexample.tfrecord-00082-of-01000 \
  --max_records 10 \
  --num_agents 16 \
  --agent_distance_threshold 80 \
  --map_distance_threshold 100 \
  --max_map_polylines 256 \
  --max_points_per_polyline 20 \
  --output_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset
```

Prepare the first 5,000-scene training set in one command:

```bash
cd /p/yufeng/tri30/dreamer4
bash waymo/submit_prepare_waymo_vector_5k.sh
```

This writes to:

```text
/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_5k
```

Watch progress:

```bash
tail -f /p/yufeng/tri30/dreamer4/waymo/prepare_waymo_vector_5k.log
```

Check the final count:

```bash
find /p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_5k -name '*.npz' | wc -l
```

Expected output per NPZ:

```text
agents:        (K, 91, 8)
agent_mask:    (K,)
map_polylines: (256, 20, 6)
map_mask:      (256, 20)
lights:        (91, 16, 4)
light_mask:    (91, 16)
```

## 3. Visualize A Filtered NPZ

This writes an MP4 and a VSCode-friendly PNG contact sheet.

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/visualize_waymo_vector_npz.py \
  /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.npz \
  --output /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.mp4
```

Default extra output:

```text
/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e_preview.png
```

Make a denser preview image:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/visualize_waymo_vector_npz.py \
  /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.npz \
  --output /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.mp4 \
  --preview_frames 20 \
  --preview_cols 5
```

Skip the PNG preview:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/visualize_waymo_vector_npz.py \
  /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.npz \
  --output /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.mp4 \
  --no_preview_png
```

Optionally create a GIF too:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/visualize_waymo_vector_npz.py \
  /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.npz \
  --output /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.mp4 \
  --gif_output /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset/3e55e88e46dac74e.gif
```

## 4. Smoke-Test The Vector Encoder

Run the encoder on the filtered NPZ dataset using a 32-step training-style window.

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_encoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 32
```

Run the encoder on an 11-step observed-prefix window.

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_encoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 11
```

Expected smoke-test output shape for `--time_window 32` with current test settings:

```text
batch agents: (2, 16, 32, 8)
batch map: (2, 256, 20, 6)
batch lights: (2, 32, 16, 4)
z: (2, 32, 8, 32)
agent_tokens: (2, 32, 16, 128)
map_tokens: (2, 32, 256, 128)
light_tokens: (2, 32, 16, 128)
```

## 5. Smoke-Test The Vector Encoder + Decoder

This runs the latent-only decoder from `z`, computes first-pass agent/light reconstruction losses, and checks backward by default.

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_decoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 11
```

Run a longer forward-only training-window check:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_decoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 32 \
  --no_backward
```

Expected output shape with current smoke-test settings:

```text
encoder z: (2, T, 8, 32)
decoder agent_continuous: (2, T, 16, 7)
decoder agent_valid_logits: (2, T, 16)
decoder light_state_logits: (2, T, 16, 16)
decoder light_valid_logits: (2, T, 16)
```

Tokenizer pretraining intentionally has no learned scene/task/policy-agent token. Following the Dreamer 4 phase split, these learned control or summary tokens should be inserted only when finetuning the dynamics model into a policy, not during tokenizer reconstruction training.

## 6. Train The Waymo Vector Tokenizer

This trains Encoder A + Decoder 1 on filtered NPZ files. The default bottleneck is the current first baseline:

```text
n_latents = 8
d_bottleneck = 32
z scalars per timestep = 256
```

The loss is:

```text
1.0 * agent xy SmoothL1
0.5 * agent speed/vx/vy SmoothL1
0.5 * agent yaw sin/cos SmoothL1
0.2 * agent valid BCE
0.5 * traffic-light state CE
0.1 * traffic-light valid BCE
```

Run a tiny debug check:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/train_waymo_vector_tokenizer.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --ckpt_dir /p/yufeng/tri30/dreamer4/waymo/checkpoints/vector_tokenizer_debug \
  --batch_size 2 \
  --time_window 11 \
  --max_steps 5 \
  --log_every 1 \
  --eval_every 5 \
  --save_every 5
```

Run the first baseline on 32-step windows:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/train_waymo_vector_tokenizer.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --ckpt_dir /p/yufeng/tri30/dreamer4/waymo/checkpoints/vector_tokenizer_baseline \
  --batch_size 2 \
  --time_window 32 \
  --epochs 50 \
  --d_model 128 \
  --depth 3 \
  --decoder_depth 3 \
  --n_latents 8 \
  --d_bottleneck 32
```

Run the same trainer on multiple GPUs with `torchrun`:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/torchrun --nproc_per_node=4 waymo/train_waymo_vector_tokenizer.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --ckpt_dir /p/yufeng/tri30/dreamer4/waymo/checkpoints/vector_tokenizer_baseline \
  --batch_size 2 \
  --time_window 32 \
  --epochs 50 \
  --d_model 128 \
  --depth 3 \
  --decoder_depth 3 \
  --n_latents 8 \
  --d_bottleneck 32
```

`--batch_size` is per GPU, so the effective global batch size is `batch_size * nproc_per_node`. With `torchrun`, each rank gets a `DistributedSampler`; only rank 0 writes checkpoints, prints metrics, and initializes wandb. Validation metrics are reduced across ranks before logging.

Resume from the latest checkpoint:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/train_waymo_vector_tokenizer.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --ckpt_dir /p/yufeng/tri30/dreamer4/waymo/checkpoints/vector_tokenizer_baseline \
  --resume /p/yufeng/tri30/dreamer4/waymo/checkpoints/vector_tokenizer_baseline/latest.pt
```

Add `--wandb --wandb_run_name <name>` if you want online logging.

## 7. Load Filtered NPZ Files From Python

```python
from waymo_vector_dataset import WaymoVectorDataset

ds = WaymoVectorDataset("/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset")
item = ds[0]

print(item["agents"].shape)
print(item["map_polylines"].shape)
print(item["lights"].shape)
```

When running this interactively, make sure the current directory is:

```bash
cd /p/yufeng/tri30/dreamer4/waymo
```

or add `/p/yufeng/tri30/dreamer4/waymo` to `PYTHONPATH`.

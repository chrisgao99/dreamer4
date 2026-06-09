# Agent Context

Last updated: 2026-05-13

## Source

- Combined boss-meeting Google Slides deck covering 2026-04-21, 2026-04-09, and 2026-03-24:
  [Google Slides](https://docs.google.com/presentation/d/14FdjmNIs-73L64M3qDy_P66qVXiMy3ILl8cpyiaAvPg/edit?usp=sharing)

## Research Focus

Yufeng is working on learning interaction-aware representations of driving scenarios, primarily from Waymo motion data. The goal is to build representations that capture multi-agent driving dynamics well enough to support downstream planning, policy learning, and analysis of special traffic interactions.

The current implementation priority is **Dreamer 4**. This is the first major direction to execute, ahead of alternative pathways.

## Current Thesis

The current working belief is:

1. A world-model-based approach is a strong way to learn driving interaction structure because predicting future world state can force the model to capture causality that hand-designed similarity rules may miss.
2. Dreamer 4 is especially promising because its transformer-style space-time structure can preserve richer multi-agent information than Dreamer 3 / RSSM-style compression.
3. Tokenizer pretraining should focus on bottleneck latents `z` and ordinary per-agent state tokens for selected tracks. Learned scene/task/policy-agent tokens should not be part of tokenizer pretraining; add them only when finetuning the dynamics model into a policy.
4. Contrastive learning is still relevant, but likely as an enhancement on top of pooled latents, selected-agent tokens, or downstream policy finetuning tokens, not as the first standalone path.

## What Changed Across the Three Meetings

### 2026-03-24

The problem was framed as **offline representation learning for interaction dynamics**. Two main pathways were analyzed:

- **Trajectory sequence alignment (CLASS-style)** using DTW
- **Dynamic system representation (MATS-style)** using affine time-varying systems

Key takeaway:

- The hard part in contrastive learning is not only the loss, but defining mathematically meaningful positive pairs for "similar driving dynamics."
- DTW can align trajectories that look spatially similar while ignoring meaningful speed differences unless richer state variables are included.
- MATS offers an interpretable dynamical-system view, but raises unresolved choices about which timesteps or interaction blocks should define similarity.

This meeting ended with two next directions:

- investigate other pathways such as world models and PINNs
- filter and visualize special interaction scenes such as highway merge, multi-lane to two lanes, and unprotected left turn

### 2026-04-09

The focus shifted toward a **Dreamer world model** as a way to learn driving dynamics directly.

Key ideas introduced:

- Use all agents' positions as input
- Encode multi-agent state with self-attention
- Handle variable numbers of agents with fixed max size, zero-padding, and masking
- Add map information through a separate map encoder plus cross-attention
- Consider whether discrete latent state `z` can represent meaningful driving-interaction concepts

Important tension:

- Dreamer-style latent state may naturally compress high-level concepts
- But it is unclear whether RSSM-style latent variables are expressive enough for rich multi-agent interaction behavior

Tooling and environment notes:

- GPUDrive is attractive for tutorials and rendering, but CUDA/NVRTC compilation is a blocker
- PufferDrive scales well on CPU but seems harder to render and inspect visually

### 2026-04-21

This meeting made the direction much more concrete: **start from Dreamer 4**.

The main shift was from RSSM compression to a **block-causal transformer world model** with both space and time processing:

- Space layers process interactions among tokens within a timestep
- Time layers process the same token/entity across timesteps
- For driving, the causal time layer is attractive because it lets the model attend to the same agent over time

The key representation idea is:

- Dreamer 4 inserts task/policy tokens during downstream agent finetuning, after tokenizer/world-model pretraining.
- For this project, tokenizer pretraining should not include a learned scene token or learned policy-agent token.
- Downstream scene/task/policy-agent tokens can be inserted later when finetuning the dynamics model into a policy, where they can gather global interaction information for policy, reward, value, probing, or retrieval.

Another important benefit is that the model output can retain **selected-agent state representations**, which may be used for agent-level dynamics analysis. Learned policy-agent tokens are reserved for later dynamics-to-policy finetuning.

## Why Dreamer 4 Is the First Implementation Target

Dreamer 4 is currently favored because it appears to solve several limitations of the Dreamer 3 / RSSM route:

- It preserves structure across both **time** and **agents**, rather than compressing all history into a small latent state
- It gives more direct access to **token-level representations**, including possible agent-specific dynamics tokens
- It is more naturally compatible with variable agent counts through transformer masking/padding
- It aligns well with the actual research need: representing rich interaction dynamics rather than only reconstructing single-step latent summaries

Known tradeoffs:

- higher compute and memory cost
- still operates with a fixed history window
- no official public codebase was noted in the slides

## Current Proposed Implementation Direction

The most important near-term plan, based on the latest meeting, is:

1. Start from Dreamer 4
2. Train the tokenizer without learned scene/task/policy-agent tokens, using bottleneck latents plus the ordinary selected-agent, traffic-light, and map tokens needed to encode the scene state.
3. Train the dynamics/world model on tokenizer latents.
4. Add downstream scene/task/policy-agent tokens only when finetuning the dynamics model into a policy.
5. Explore whether bottleneck latents, selected-agent state tokens, and downstream policy finetuning tokens serve as useful interaction representations.

## World Model vs. Contrastive Learning

The current stance is not "one replaces the other," but rather:

- **World model first** to learn causally grounded structure
- **Contrastive refinement second** if needed to sharpen representation geometry

Working comparison:

- Contrastive learning can emphasize distinctions between scenes, but depends heavily on how positive and negative pairs are defined
- World-model learning can capture detailed structure and causal regularities, but may also encode noise without additional pressure

Practical interpretation:

- Dreamer 4-style tokenizer/dynamics first; downstream scene/task/policy-agent tokens only during policy finetuning
- contrastive loss is a likely add-on, not the first milestone

## Open Technical Questions

- How exactly should downstream scene/task/policy-agent tokens be trained during dynamics-to-policy finetuning: policy/reward/value losses, auxiliary probing, contrastive loss, or some combination?
- What should count as the best scene-level objective for driving interaction representation?
- How useful are the per-agent output tokens in practice for representing agent dynamics?
- How much map information should enter the world model, and at what stage?
- Which simulator or environment path is most practical for downstream RL experiments given current GPUDrive and PufferDrive limitations?
- What evaluation will best show that the learned representation is truly interaction-aware rather than only predictive?

## Environment and Data Notes

- Waymo motion data is the central dataset mentioned in the slides
- Special interaction scenes of interest include:
  - highway merge
  - multi-lane to two lanes
  - unprotected left turn
- GPUDrive has been used for visualizing special scenes
- GPUDrive currently has CUDA/NVRTC-related compilation issues
- PufferDrive is still of interest for scalable RL experimentation

## What A Future Codex Session Should Know Immediately

If a future session starts from this file, the most important facts are:

- Yufeng is studying **interaction-aware driving representations**
- The active implementation priority is **Dreamer 4**
- The tokenizer should not include learned scene/task/policy-agent tokens; add them only when finetuning the dynamics model into a policy
- Dreamer 3 / RSSM and pure contrastive approaches are important baselines or comparisons, but not the first implementation target
- The session should treat the latest plan from **2026-04-21** as the current source of truth unless newer notes override it

## Suggested Immediate Next Steps

- Gather the exact Dreamer 4 paper/code references that match the transformer world-model design described in the slides
- Translate the downstream scene/task/policy-agent-token idea into a concrete dynamics-to-policy finetuning spec for driving data
- Define the input tensor structure for multi-agent trajectories and map features
- Decide the first downstream objective for validating representation quality
- Record implementation choices and unresolved questions in a separate running note as work begins

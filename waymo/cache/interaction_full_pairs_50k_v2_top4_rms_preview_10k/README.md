# RMS preview (10k per split)

This is a parameter/visual sanity check built from the completed v2 dataset
that still used `non_ooi_top_k_per_focus=4`. It is not the final no-top-k
result.

- 10,000 train samples and 10,000 val samples were loaded.
- The final chosen retrieval setting is 3 DCT coefficients, 1,024 coarse
  candidates, exact mask-aware RMS reranking, and top 32 storage.
- `visual_audit/index.html` samples 12 train top-1 matches across the RMS
  distance distribution.
- The no-top-k full result will be written separately under
  `../interaction_full_pairs_50k_v2_no_topk_rms_v0/`.

Fixed-window overfit manifests generated from reconstruction_fulltraj_random10_staticmap_v2_chunk32_threeckpts.

seed=20260622
dataset_root=/scratch/baz7dy/tri30/dreamer4/waymo/data/waymo_vector_dataset_ooi_centered_50k

true_fixedwin_report10.txt: the 10 report scenes, in sorted report-directory order.
true_fixedwin_report100.txt: first the 10 report scenes, then 90 random dataset NPZs.
true_fixedwin_report1000.txt: first the 100-list, then 900 more random dataset NPZs.

All paths are absolute /scratch paths.

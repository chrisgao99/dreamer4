import os,json,pathlib,subprocess,sys,torch
root=pathlib.Path('/p/yufeng/tri30/dreamer4')
out=root/'waymo/eval_results/world_model/minade_four_models_intersection10_n10_h80_b5_20260924'
ref=root/'waymo/eval_results/world_model/h90_intersection10_two_agent_n10_ctx11_h80_20260907'
items=[('cuda1_mon_detach0_step2500','waymo_daf_h40_95k_mon8_gt_unroll80_b5_detach0_10k_20260922','best_ade.pt',2500),('cuda2_mon_detach5_step3000','waymo_daf_h40_95k_mon8_gt_unroll80_b5_detach5_10k_20260922','step_0003000.pt',3000),('cuda3_mon_detach20_step2500','waymo_daf_h40_95k_mon8_gt_unroll80_b5_detach20_10k_20260922','best_ade.pt',2500)]
if '--catk' in sys.argv:
 items=[('cuda0_catk_step1000','waymo_daf_h40_95k_catk_sample8_recovery_flow_unroll80_b5_10k_20260922','step_0001000.pt',1000)]
for label,folder,file,step in items:
 src=root/'waymo/checkpoints'/folder/file
 c=torch.load(src,map_location='cpu',weights_only=False)
 assert c['step']==step,(src,c['step'])
 rows=[json.loads(l) for l in (src.parent/'metrics.jsonl').read_text().splitlines()]
 val=next(r for r in rows if r.get('kind')=='validation' and r['step']==step)
 manifest=dict(source_checkpoint=str(src),step=step,validation=val,historical_minade=min((r for r in rows if r.get('kind')=='validation'),key=lambda r:r['val_minade_m']),reference_dir=str(ref),seed=20260907,num_scenes=10,num_rollouts=10,rollout_steps=80,commitment_steps=5,solver_steps=8,weights='ema')
 # Preserve selected EMA weights so ongoing training cannot overwrite this selection.
 frozen=out/(label+'.pt')
 torch.save({k:c[k] for k in ('ema_model','args','action_stats','step')},frozen)
 del c
 (out/(label+'_selection.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 cmd=[sys.executable,str(root/'waymo/evaluation/visualize_direct_action_flow_intersection_two_agent_rollouts.py'),'--checkpoint',str(frozen),'--val_data_dir',str(root/'data/waymo_vector_dataset_ooi_centered_50k/val'),'--selected_manifest',str(ref/'selected_intersection10_manifest.json'),'--reference_h90_dir',str(ref),'--output_dir',str(out),'--device','cuda','--weights','ema','--focus_mode','generate_all','--num_workers','4','--num_scenes','10','--num_rollouts','10','--rollout_steps','80','--commitment_steps','5','--solver_steps','8','--seed','20260907','--model_label',label+' (EMA)','--images_only','--filename_prefix',label+'_']
 print('START',label,flush=True)
 with (out/(label+'.log')).open('w') as log: subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',label,flush=True)

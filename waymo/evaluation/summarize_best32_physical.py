#!/usr/bin/env python3
import csv,json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for p in sorted(root.glob('*/result.json')):
 d=json.loads(p.read_text())
 rows.append(dict(experiment=p.parent.name,step=d['checkpoint_step'],scenes=d['scene_count'],
   mean_ade_m=d['mean_ade_m'],minade_m=d['minade_m'],cpd=d['flow_erd_cpd'],**d.get('physical_metrics',{})))
if rows:
 tmp=root/f'summary.{os.getpid()}.tmp'
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(root/'summary.csv')
print(f'Completed results: {len(rows)}/4; summary: {root / "summary.csv"}')

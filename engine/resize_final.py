"""Final local entry point: v0.4 + relations-v2, alpha-aware selection."""
import argparse,json,shutil
from pathlib import Path
from .resize_selector_v2 import resize,run,ROOT
VERSION='final-1.0'
def main():
 p=argparse.ArgumentParser(description='Generate two candidates, validate, render and select.');p.add_argument('--input',type=Path,default=ROOT/'input');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 paths=sorted(a.input.glob('*.json')) if a.input.is_dir() else [a.input]
 if not paths:raise ValueError('No input JSON tasks')
 a.output.mkdir(parents=True,exist_ok=False);summary=[]
 for path in paths:
  task=json.loads(path.read_text(encoding='utf-8-sig'));folder=a.output/'candidates'/path.stem;r=run(task,folder)
  if r['selected']:
   for ext in ['json','png']:shutil.copy2(folder/('selected.'+ext),a.output/(path.stem+'_output.'+ext))
  summary.append({'task':path.name,'selected':r['selected'],'low_margin':r.get('low_margin'),'details':str(folder.relative_to(a.output))})
  print(path.name,'->',r['selected'],flush=True)
 (a.output/'summary.json').write_text(json.dumps({'version':VERSION,'tasks':summary,'needs_visual_review':True},ensure_ascii=False,indent=2),encoding='utf-8')
 if any(r['selected'] is None for r in summary):raise SystemExit(1)
if __name__=='__main__':main()

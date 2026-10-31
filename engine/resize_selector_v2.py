"""Offline two-engine candidate selector; no engine is silently substituted."""
import argparse,json,subprocess,sys,shutil,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .candidate_quality_v2 import score,choose
from renderer.converter.convert import render_design_to_png
ROOT=Path(__file__).resolve().parents[1]
VERSION='selector-v2-alpha'
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def generate(name,task):
 module={'v04':'engine.resize_v0_4','relations_v2':'engine.resize_relations_v2'}[name]
 code=f'import json,sys,contextlib;from {module} import resize;t=json.load(sys.stdin);\nwith contextlib.redirect_stdout(sys.stderr): d=resize(t["source"],t["target_canvas"])\njson.dump(d,sys.stdout,ensure_ascii=False,allow_nan=False)'
 r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],input=json.dumps(task,ensure_ascii=False),text=True,encoding='utf-8',capture_output=True,cwd=ROOT,timeout=300)
 if r.returncode:raise RuntimeError(r.stderr[-3000:])
 return json.loads(r.stdout)
def validation_python():
 override=os.environ.get('CHAMELEON_TORCH_PYTHON')
 if override:return override
 local=ROOT/'torch-py/.venv'/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
 return str(local) if local.is_file() else sys.executable
def validate(source,design):
 code='import json,sys;from chameleon_torch.data import validate_pair;s,d=json.load(sys.stdin);print(json.dumps(validate_pair(s,d)))'
 r=subprocess.run([validation_python(),'-B','-X','utf8','-c',code],input=json.dumps([source,design]),text=True,encoding='utf-8',capture_output=True,cwd=ROOT/'engine',timeout=60)
 if r.returncode:raise ValueError(r.stderr[-2000:])
 return json.loads(r.stdout)
def run(task,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);dump(out/'input.json',task);dump(out/'source.json',task['source']);reports={};start=time.monotonic()
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures={name:pool.submit(generate,name,task) for name in ['v04','relations_v2']}
  for name,future in futures.items():
   try:
    d=future.result();dump(out/(name+'.json'),d)
    if d['canvas']['width']!=task['target_canvas']['width'] or d['canvas']['height']!=task['target_canvas']['height']:raise ValueError('Wrong canvas')
    warnings=validate(task['source'],d)
    rw=render_design_to_png(d,ROOT/'brand_kit/assets',out/(name+'.png'),ROOT/'brand_kit/fonts',max_size=max(task['target_canvas'].values()))
    reports[name]={'valid':True,'validation_warnings':warnings,'render_warnings':rw,'score':score(task['source'],d)}
   except Exception as exc:reports[name]={'valid':False,'error':str(exc)}
 selection=choose(reports);report={'version':VERSION,'candidates':reports,**selection,'seconds':time.monotonic()-start}
 dump(out/'selection.json',report)
 if selection['selected']:
  for ext in ['json','png']:shutil.copy2(out/(selection['selected']+'.'+ext),out/('selected.'+ext))
 return report

def resize(source,target_canvas):
 import tempfile
 with tempfile.TemporaryDirectory(prefix='chameleon_selector_') as folder:
  out=Path(folder)/'result';r=run({'source':source,'target_canvas':target_canvas},out)
  if not r['selected']:raise RuntimeError(json.dumps(r,ensure_ascii=False))
  return json.loads((out/'selected.json').read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();task=json.loads(a.input.read_text(encoding='utf-8-sig'));r=run(task,a.output);print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['selected'] else 1)
if __name__=='__main__':main()

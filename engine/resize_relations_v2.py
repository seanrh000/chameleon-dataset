"""Experimental relation-trained engine. Run with the renderer Python environment."""
from copy import deepcopy
import argparse,json,math
from pathlib import Path
from .inference_bridge import resize_many as predict_many
from .relation_refiner_v3 import refine
from chameleon_torch.geometry import EDITABLE,box_bounds,locked_background
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'engine/models/relations_v1.pt'
VERSION='relations-v2'

def check(source,design):
    src={e['id']:e for e in source['elements']};dst={e['id']:e for e in design['elements']}
    if src.keys()!=dst.keys() or len(dst)!=len(design['elements']):raise ValueError('IDs changed')
    w,h=design['canvas']['width'],design['canvas']['height']
    for eid,e in dst.items():
        s=src[eid]
        if {k:v for k,v in e.items() if k not in EDITABLE}!={k:v for k,v in s.items() if k not in EDITABLE}:raise ValueError('Immutable attributes changed: '+eid)
        if not all(math.isfinite(e.get(k,0)) for k in EDITABLE) or min(e['width'],e['height'])<=0:raise ValueError('Invalid geometry: '+eid)
        x,y,bw,bh=box_bounds(e)
        if min(x,y)<-.05 or x+bw>w+.05 or y+bh>h+.05:raise ValueError('Outside canvas: '+eid)
        if e['type']=='image' and not locked_background(s,source,'cover') and not math.isclose(e['width']/e['height'],s['width']/s['height'],rel_tol=1e-4):raise ValueError('Image aspect changed: '+eid)

def resize_many(tasks,*,device='cpu',repair=True):
    results=predict_many(tasks,checkpoint=MODEL_PATH,device=device)
    for task,result in zip(tasks,results):
        raw=deepcopy(result['design'])
        report={}
        if repair:
            result['design'],report=refine(task['source'],raw)
        check(task['source'],result['design'])
        result['report'].update(engine_version=VERSION,checkpoint=str(MODEL_PATH),relation_repair=report,repair_enabled=repair,needs_visual_review=True)
    return results

def resize(source,target_canvas):
    return resize_many([{'source':source,'target_canvas':target_canvas}])[0]['design']

def main():
    from renderer.converter.convert import render_design_to_png
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'input');p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--no-repair',action='store_true');a=p.parse_args()
    paths=sorted(a.input.glob('*.json')) if a.input.is_dir() else [a.input]
    if not paths:raise ValueError('No task JSON')
    if a.output.exists():raise FileExistsError('Use a new output folder')
    tasks=[json.loads(f.read_text(encoding='utf-8-sig')) for f in paths]
    results=resize_many(tasks,device=a.device,repair=not a.no_repair);a.output.mkdir(parents=True)
    for path,result in zip(paths,results):
        d=result['design'];name=path.stem+'_output'
        (a.output/(name+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
        warnings=render_design_to_png(d,ROOT/'brand_kit/assets',a.output/(name+'.png'),ROOT/'brand_kit/fonts',max_size=max(d['canvas'].values()))
        result['report'].update(official_render_performed=True,render_warnings=warnings)
        (a.output/(name+'_report.json')).write_text(json.dumps(result['report'],ensure_ascii=False,indent=2),encoding='utf-8')
        print(path.name,'OK; visual review required')
if __name__=='__main__':main()

"""Relation repair v2: rotation-aware containment and group-local transforms. Experimental."""
from copy import deepcopy
from pathlib import Path
import math,sys,re,io
from PIL import Image
from functools import lru_cache
from renderer.converter.convert import convert_chameleon_to_crello
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
from chameleon_torch.geometry import box_bounds,locked_background
from renderer.converter.convert import _get_renderer
from engine.resize_v0_4 import _Typography

def bounds(es):
 b=[box_bounds(e) for e in es];x=min(z[0] for z in b);y=min(z[1] for z in b)
 return x,y,max(z[0]+z[2] for z in b)-x,max(z[1]+z[3] for z in b)-y

def overlap(a,b):return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))

def refine(source,prediction):
 d=deepcopy(prediction);src={e['id']:e for e in source['elements']};dst={e['id']:e for e in d['elements']};w,h=d['canvas']['width'],d['canvas']['height'];groups={};notes=[]
 assert set(src)==set(dst)
 for e in source['elements']:groups.setdefault(('group',e['group_id']) if e.get('group_id') else ('single',e['id']),[]).append(e['id'])
 fm=_get_renderer(str(ROOT/'brand_kit/fonts')).font_manager
 renderer=_get_renderer(str(ROOT/'brand_kit/fonts'))
 inferred=[]
 # Only high-confidence day/month pairs; no source IDs or poster-specific positions.
 months=set('JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC JANUARY FEBRUARY MARCH APRIL JUNE JULY AUGUST SEPTEMBER OCTOBER NOVEMBER DECEMBER'.split())
 free=[e for e in source['elements'] if e['type']=='text' and not e.get('group_id')]
 for day in free:
  value=day.get('content','').strip()
  if not re.fullmatch(r'\d{1,2}',value) or not 1<=int(value)<=31:continue
  candidates=[e for e in free if e.get('content','').strip().upper() in months and abs(e['y']+e['height']/2-day['y']-day['height']/2)<source['canvas']['height']*.12 and 0<=e['x']-day['x']-day['width']<source['canvas']['width']*.06]
  if len(candidates)==1:
   month=candidates[0];ids=[day['id'],month['id']]
   if all(('single',i) in groups for i in ids):
    for i in ids:del groups['single',i]
    groups['inferred_date',day['id']]=ids;inferred.append(ids)
 # Fit before measuring visible ink. No changes to text, fonts or source metadata.
 meter=_Typography(d['elements'],fm)
 for idx,e in enumerate(d['elements']):
  if e['type']=='text':e['font_size']=meter.fit(idx,e['width'],e['height'],e['font_size'])
 @lru_cache(maxsize=4096)
 def ink_local(eid,width,height,font):
  e=deepcopy(dst.get(eid,src[eid]));e.update(x=font*2+4,y=font*2+4,width=width,height=height,font_size=font,rotation=0)
  pad=font*2+4;cw=math.ceil(width+2*pad);ch=math.ceil(height+2*pad)
  example,_=convert_chameleon_to_crello({'canvas':{'width':cw,'height':ch},'elements':[e]},fm,str(ROOT/'brand_kit/assets'))
  img=Image.open(io.BytesIO(renderer.render(example,max_size=min(4096,max(cw,ch)),format='png',canvas_color=None))).convert('RGBA')
  bb=img.getchannel('A').point(lambda v:255 if v>=8 else 0).getbbox()
  if bb is None:return 0,0,0,0
  sx,sy=cw/img.width,ch/img.height
  return bb[0]*sx-pad,bb[1]*sy-pad,(bb[2]-bb[0])*sx,(bb[3]-bb[1])*sy
 def visible(e):
  if e['type']!='text':return box_bounds(e)
  ox,oy,iw,ih=ink_local(e['id'],e['width'],e['height'],e['font_size']);cx=e['width']/2;cy=e['height']/2
  a=math.radians(e.get('rotation',0));c,t=math.cos(a),math.sin(a)
  pts=[(e['x']+cx+(x-cx)*c-(y-cy)*t,e['y']+cy+(x-cx)*t+(y-cy)*c) for x in [ox,ox+iw] for y in [oy,oy+ih]]
  left=min(p[0] for p in pts);top=min(p[1] for p in pts)
  return left,top,max(p[0] for p in pts)-left,max(p[1] for p in pts)-top
 def visual_bounds(es):
  bs=[visible(e) for e in es];x=min(b[0] for b in bs);y=min(b[1] for b in bs)
  return x,y,max(b[0]+b[2] for b in bs)-x,max(b[1]+b[3] for b in bs)-y
 # Date remains a single movable unit; source/output group_id are untouched.
 for day,month in inferred:
  a,b=visible(dst[day]),visible(dst[month]);gap=min(w,h)*.035
  dst[month]['x']+=a[0]+a[2]+gap-b[0]
  dst[month]['y']+=a[1]+a[3]/2-b[1]-b[3]/2
 separators=[]
 for line in source['elements']:
  if line['type']!='shape' or line.get('group_id') or line.get('rotation',0) or line['width']/max(line['height'],.01)<30:continue
  candidates=[e for e in free if ('single',e['id']) in groups and abs(e['x']-line['x'])<source['canvas']['width']*.025 and abs(e['width']-line['width'])<source['canvas']['width']*.05 and 0<e['y']-line['y']-line['height']<source['canvas']['height']*.12]
  if len(candidates)!=1:continue
  text=candidates[0];anchor=dst[text['id']];e=dst[line['id']]
  if ('single',line['id']) not in groups:continue
  e['x']=anchor['x'];e['width']=anchor['width'];e['height']=max(.1,line['height']*anchor['width']/line['width'])
  e['y']=anchor['y']-e['height']-min(w,h)*.035
  del groups['single',line['id']];del groups['single',text['id']]
  groups['inferred_separator',line['id']]=[line['id'],text['id']];separators.append([line['id'],text['id']])
 repaired=[]
 for key,ids in groups.items():
  if key[0]!='group':continue
  shapes=[src[i] for i in ids if src[i]['type']=='shape' and src[i].get('shape_type')!='line']
  if not shapes:continue
  anchor=max(shapes,key=lambda e:e['width']*e['height']);a=dst[anchor['id']]
  # Only restore groups that actually contain their members in the source.
  def rotate(x,y,angle):
   rad=math.radians(angle);return x*math.cos(rad)-y*math.sin(rad),x*math.sin(rad)+y*math.cos(rad)
  def local_center(e):
   return rotate(e['x']+e['width']/2-anchor['x']-anchor['width']/2,e['y']+e['height']/2-anchor['y']-anchor['height']/2,-anchor.get('rotation',0))
  def contained(e):
   cx,cy=local_center(e);angle=e.get('rotation',0)-anchor.get('rotation',0)
   for dx,dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
    rx,ry=rotate(dx*e['width']/2,dy*e['height']/2,angle)
    if abs(cx+rx)>anchor['width']/2+.1 or abs(cy+ry)>anchor['height']/2+.1:return False
   return True
  kids=[i for i in ids if i!=anchor['id'] and contained(src[i])]
  for eid in kids:
   s,e=src[eid],dst[eid];sx=a['width']/anchor['width'];sy=a['height']/anchor['height']
   rel=s.get('rotation',0)-anchor.get('rotation',0)
   # A rotated child under anisotropic scaling would require shear. Keep a rectangle.
   if e['type']=='image' or abs(math.sin(math.radians(rel)))>1e-6:
    ww=s['width']*min(sx,sy);hh=s['height']*min(sx,sy)
   else:ww=s['width']*sx;hh=s['height']*sy
   lx,ly=local_center(s);rx,ry=rotate(lx*sx,ly*sy,a.get('rotation',0))
   cx=a['x']+a['width']/2+rx;cy=a['y']+a['height']/2+ry
   e.update(x=cx-ww/2,y=cy-hh/2,width=ww,height=hh,rotation=(a.get('rotation',0)+rel+180)%360-180)
   if e['type']=='text':e['font_size']=max(1,s['font_size']*min(sx,sy))
  if kids:repaired.append(key[1])
 # Foreground blocks move as whole groups. Decorative ungrouped shapes are excluded.
 blocks=[]
 for key,ids in groups.items():
  if any(locked_background(src[i],source,'cover') for i in ids):continue
  if not any(src[i]['type']=='text' or src[i]['type']=='image' and (ROOT/'brand_kit/assets'/src[i]['asset_id']).exists() for i in ids):continue
  blocks.append(ids)
 margin=max(2,min(w,h)*.025)
 def clamp_move(ids,dx,dy):
  x,y,bw,bh=bounds([dst[i] for i in ids]);mx=min(margin,max(0,(w-bw)/2));my=min(margin,max(0,(h-bh)/2));dx=min(max(dx,mx-x),w-mx-x-bw);dy=min(max(dy,my-y),h-my-y-bh)
  for i in ids:dst[i]['x']+=dx;dst[i]['y']+=dy
 for ids in blocks:clamp_move(ids,0,0)
 def relevant(i,j):
  a=bounds([src[k] for k in blocks[i]]);b=bounds([src[k] for k in blocks[j]])
  return overlap(a,b)<=.01*min(a[2]*a[3],b[2]*b[3])
 pairs=[(i,j) for i in range(len(blocks)) for j in range(i+1,len(blocks)) if relevant(i,j)]
 def cost():return sum(overlap(visual_bounds([dst[k] for k in blocks[i]]),visual_bounds([dst[k] for k in blocks[j]])) for i,j in pairs)
 initial=cost();moves=0;scales={i:1.0 for i in range(len(blocks))}
 for iteration in range(30):
  best=None;base=cost()
  if base<.01:break
  for i,j in pairs:
   a=visual_bounds([dst[k] for k in blocks[i]]);b=visual_bounds([dst[k] for k in blocks[j]])
   if overlap(a,b)<=0:continue
   for idx,other in [(i,b),(j,a)]:
    ids=blocks[idx];x,y,bw,bh=visual_bounds([dst[k] for k in ids]);ox,oy,ow,oh=other
    for dx,dy in [(ox-margin-x-bw,0),(ox+ow+margin-x,0),(0,oy-margin-y-bh),(0,oy+oh+margin-y)]:
     saved=[(dst[k]['x'],dst[k]['y']) for k in ids];clamp_move(ids,dx,dy);c=cost();dist=math.hypot(dst[ids[0]]['x']-saved[0][0],dst[ids[0]]['y']-saved[0][1]);actual=(dst[ids[0]]['x']-saved[0][0],dst[ids[0]]['y']-saved[0][1])
     for k,(px,py) in zip(ids,saved):dst[k]['x']=px;dst[k]['y']=py
     score=c+dist*.01
     if c<base-.01 and (best is None or score<best[0]):best=(score,ids,actual)
  if best is None:
   # If translation is trapped, shrink an entire colliding group uniformly.
   # A bounded 0.65 floor prevents progressively collapsing everything to points.
   candidates=[]
   for idx,ids in enumerate(blocks):
    if len(ids)<2 or scales[idx]*.9<.65:continue
    if not any((idx==i or idx==j) and overlap(visual_bounds([dst[k] for k in blocks[i]]),visual_bounds([dst[k] for k in blocks[j]]))>0 for i,j in pairs):continue
    saved={k:deepcopy(dst[k]) for k in ids};x,y,bw,bh=bounds([dst[k] for k in ids]);cx=x+bw/2;cy=y+bh/2
    for k in ids:
     e=dst[k];e['x']=cx+(e['x']-cx)*.9;e['y']=cy+(e['y']-cy)*.9;e['width']*=.9;e['height']*=.9
     if e['type']=='text':e['font_size']*=.9
    candidate={k:deepcopy(dst[k]) for k in ids};c=cost()
    for k in ids:dst[k].clear();dst[k].update(saved[k])
    if c<base-.01:candidates.append((c,idx,candidate))
   if not candidates:break
   _,idx,candidate=min(candidates,key=lambda x:x[0])
   for k in candidate:dst[k].clear();dst[k].update(candidate[k])
   scales[idx]*=.9
   continue
  clamp_move(best[1],*best[2]);moves+=1
 fm=_get_renderer(str(ROOT/'brand_kit/fonts')).font_manager;meter=_Typography(d['elements'],fm);textissues=[]
 for i,e in enumerate(d['elements']):
  if e['type']!='text':continue
  lo,hi=1,max(1,int(e['font_size']));best=1
  while lo<=hi:
   mid=(lo+hi)//2;tw,th,lines=meter.measure(i,mid,e['width'])
   if tw<=e['width']+.1 and th<=e['height']+.1 and lines<=(e.get('max_lines') or lines):best=mid;lo=mid+1
   else:hi=mid-1
  e['font_size']=best
  tw,th,lines=meter.measure(i,best,e['width'])
  if tw>e['width']+.1 or th>e['height']+.1 or lines>(e.get('max_lines') or lines):
   textissues.append({'id':e['id'],'issue':'text_does_not_fit_at_minimum_font'})
  if best<8:textissues.append({'id':e['id'],'issue':'small_text_under_8px_warning_only'})
 return d,{'version':'relations-v2','collision_geometry':'official rendered text alpha bounds (threshold 8/255), rotated around JSON box center','inferred_date_groups':inferred,'inferred_separator_groups':separators,'safe_margin':margin,'groups_restored':repaired,'foreground_overlap_area_before':initial,'foreground_overlap_area_after':cost(),'block_moves':moves,'group_scales':{str(blocks[i]):v for i,v in scales.items() if v<1},'warnings':textissues,'limitations':['Source non-overlap heuristics; intentional overlap may be ambiguous','No guarantee of optimal packing or readability; small text is reported, not auto-rejected']}

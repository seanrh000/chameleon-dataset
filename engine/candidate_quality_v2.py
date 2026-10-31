"""Independent, source-aware heuristic scorer. Not the official competition score."""
import io,json,math,itertools
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from PIL import Image
from renderer.converter.assets import load_asset_image
from .resize_v0_4 import _Typography
from .relation_refiner_v3 import bounds,overlap
from .resize_relations_v2 import check
from chameleon_torch.geometry import box_bounds,is_background
from renderer.converter.convert import _get_renderer,convert_chameleon_to_crello
ROOT=Path(__file__).resolve().parents[1]
WEIGHTS={'text_fit':25,'readability':20,'collision':20,'relationships':15,'hierarchy':10,'margin':5,'coverage':5}
def avg(xs):return sum(xs)/len(xs) if xs else 0.
def clip(x):return max(0.,min(1.,x))
@lru_cache(maxsize=4096)
def local_ink(raw):
 e=json.loads(raw);font=e['font_size'];pad=2*font+4;e.update(x=pad,y=pad,rotation=0)
 cw=math.ceil(e['width']+2*pad);ch=math.ceil(e['height']+2*pad)
 renderer=_get_renderer(str(ROOT/'brand_kit/fonts'));ex,_=convert_chameleon_to_crello({'canvas':{'width':cw,'height':ch},'elements':[e]},renderer.font_manager,str(ROOT/'brand_kit/assets'))
 im=Image.open(io.BytesIO(renderer.render(ex,max_size=min(4096,max(cw,ch)),format='png',canvas_color=None))).convert('RGBA');b=im.getchannel('A').point(lambda x:255 if x>=8 else 0).getbbox()
 if b is None:return (0.,0.,0.,0.)
 sx,sy=cw/im.width,ch/im.height
 return b[0]*sx-pad,b[1]*sy-pad,(b[2]-b[0])*sx,(b[3]-b[1])*sy
@lru_cache(maxsize=4096)
def image_ink(asset_id,width,height,fit,opacity):
 im=load_asset_image(str(ROOT/'brand_kit/assets'),asset_id,width,height,fit)
 bb=im.getchannel('A').point(lambda a:255 if a*opacity>=8 else 0).getbbox()
 if bb is None:return (0.,0.,0.,0.)
 sx,sy=width/im.width,height/im.height
 return bb[0]*sx,bb[1]*sy,(bb[2]-bb[0])*sx,(bb[3]-bb[1])*sy

def ink(e):
 if e['type']=='image':
  x,y,w,h=image_ink(e.get('asset_id'),e['width'],e['height'],e.get('fit','contain'),e.get('opacity',1))
  return rotated_ink(e,x,y,w,h)
 if e['type']!='text':return box_bounds(e)
 local={k:v for k,v in e.items() if k not in ('x','y','rotation')};x,y,w,h=local_ink(json.dumps(local,sort_keys=True))
 return rotated_ink(e,x,y,w,h)
def rotated_ink(e,x,y,w,h):
 cx,cy=e['width']/2,e['height']/2;a=math.radians(e.get('rotation',0));c,s=math.cos(a),math.sin(a)
 pts=[(e['x']+cx+(u-cx)*c-(v-cy)*s,e['y']+cy+(u-cx)*s+(v-cy)*c) for u in (x,x+w) for v in (y,y+h)]
 left=min(p[0] for p in pts);top=min(p[1] for p in pts)
 return left,top,max(p[0] for p in pts)-left,max(p[1] for p in pts)-top

def score(source,design):
 check(source,design)
 src={e['id']:e for e in source['elements']};dst={e['id']:e for e in design['elements']};w,h=design['canvas']['width'],design['canvas']['height']
 fg=[e['id'] for e in source['elements'] if not is_background(e,source) and (e['type']=='text' or (e['type']=='image' and (ROOT/'brand_kit/assets'/e['asset_id']).is_file()))]
 si={i:ink(src[i]) for i in fg};di={i:ink(dst[i]) for i in fg}
 texts=[e for e in design['elements'] if e['type']=='text'];meter=_Typography(texts,_get_renderer(str(ROOT/'brand_kit/fonts')).font_manager)
 fits=[];read=[];warnings=[]
 for idx,e in enumerate(texts):
  tw,th,lines=meter.measure(idx,max(1,int(e['font_size'])),e['width'])
  err=max(0,tw/e['width']-1,th/e['height']-1,(lines/e['max_lines']-1) if e.get('max_lines') else 0)
  fits.append(clip(err));read.append(clip((10-e['font_size'])/10))
  if err>.01:warnings.append({'id':e['id'],'issue':'text_fit','overflow_ratio':err})
  if e['font_size']<8:warnings.append({'id':e['id'],'issue':'small_text','font_px':e['font_size']})
 collisions=[]
 for i,j in itertools.combinations(fg,2):
  if src[i].get('group_id') and src[i].get('group_id')==src[j].get('group_id'):continue
  if overlap(si[i],si[j])>.01*min(si[i][2]*si[i][3],si[j][2]*si[j][3]):continue
  collisions.append(clip(overlap(di[i],di[j])/max(1,min(di[i][2]*di[i][3],di[j][2]*di[j][3]))))
 relations=[]
 for i in fg:
  child=src[i];gid=child.get('group_id')
  if not gid:continue
  anchors=[e for e in source['elements'] if e['type']=='shape' and e.get('group_id')==gid and e.get('shape_type')!='line']
  for anchor in anchors:
   sb=box_bounds(anchor)
   if overlap(si[i],sb)>=.95*max(1,si[i][2]*si[i][3]):
    db=box_bounds(dst[anchor['id']]);relations.append(clip(1-overlap(di[i],db)/max(1,di[i][2]*di[i][3])))
 # Only explicit unambiguous day/month combinations inferred from the source.
 months=set('JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC JANUARY FEBRUARY MARCH APRIL JUNE JULY AUGUST SEPTEMBER OCTOBER NOVEMBER DECEMBER'.split())
 for i in fg:
  value=src[i].get('content','').strip()
  if not value.isdigit() or not 1<=int(value)<=31:continue
  matches=[j for j in fg if src[j].get('content','').strip().upper() in months and abs((src[j]['y']+src[j]['height']/2)-(src[i]['y']+src[i]['height']/2))<source['canvas']['height']*.12 and 0<=src[j]['x']-src[i]['x']-src[i]['width']<source['canvas']['width']*.06]
  if len(matches)==1:
   a,b=di[i],di[matches[0]];gap=max(0,b[0]-a[0]-a[2],a[0]-b[0]-b[2],b[1]-a[1]-a[3],a[1]-b[1]-b[3]);relations.append(clip(gap/max(1,min(w,h)*.15)))
 ratios=[math.log(max(e['font_size'],.01)/src[e['id']]['font_size']) for e in texts]
 mean=avg(ratios);hierarchy=avg([clip(abs(x-mean)/1.5) for x in ratios])
 margin=min(w,h)*.025;edge=[clip(1-min(b[0],b[1],w-b[0]-b[2],h-b[1]-b[3])/max(1,margin)) for b in di.values()]
 sc=avg([b[2]*b[3] for b in si.values()])/(source['canvas']['width']*source['canvas']['height'])
 dc=avg([b[2]*b[3] for b in di.values()])/(w*h)
 coverage=clip(abs(math.log(max(dc,1e-8)/max(sc,1e-8)))/3) if fg else 0
 penalties={'text_fit':avg(fits),'readability':avg(read),'collision':avg(sorted(collisions,reverse=True)[:max(1,len(fg))]),'relationships':avg(relations),'hierarchy':hierarchy,'margin':avg(edge),'coverage':coverage}
 points={k:WEIGHTS[k]*(1-v) for k,v in penalties.items()}
 return {'total':sum(points.values()),'points':points,'maximum':WEIGHTS,'penalties':penalties,'warnings':warnings,'label':'heuristic quality score; not official or calibrated human preference','limits':['AABB approximates rotated overlap','No semantic/image aesthetics model','10px readability reference is a heuristic, not a rule','Grouped intentional overlap is handled conservatively','Coverage does not measure whitespace aesthetics']}

def choose(reports):
 valid=[(name,r) for name,r in reports.items() if r.get('valid') and 'score' in r]
 if not valid:return {'selected':None,'status':'no_valid_candidate','needs_review':True}
 valid.sort(key=lambda v:(-v[1]['score']['total'],v[0]))
 margin=valid[0][1]['score']['total']-valid[1][1]['score']['total'] if len(valid)>1 else None
 return {'selected':valid[0][0],'score_margin':margin,'status':'selected','low_margin':margin is not None and margin<5,'needs_review':True,'confidence_note':'Score margin is not a probability; all results need visual review.'}

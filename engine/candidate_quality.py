"""Source-aware heuristic scoring and candidate selection.

Includes alpha-aware bounds, source-relative prominence, and conservative
text-flow relationships. Not the official competition score.
"""
import io,json,math,itertools
from functools import lru_cache
from pathlib import Path
from PIL import Image
from renderer.converter.assets import load_asset_image
from .resize_v0_4 import _Typography
from .relation_refiner_v3 import overlap
from .resize_relations_v2 import check
from chameleon_torch.geometry import box_bounds,is_background
from renderer.converter.convert import _get_renderer,convert_chameleon_to_crello
ROOT=Path(__file__).resolve().parents[1]
VERSION = 'quality-v3-source-relations'
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

def _base_score(source,design):
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


def area_scale(element):
    if element['type'] == 'text':
        return max(.01, element['font_size'])
    _, _, w, h = image_ink(element.get('asset_id'), element['width'], element['height'],
                              element.get('fit', 'contain'), element.get('opacity', 1))
    return max(.01, math.sqrt(w*h))


def prominence(source_elements, target_elements):
    """Compare relative linear scale changes, invariant to uniform resizing."""
    dst = {e['id']: e for e in target_elements}
    scales = {e['id']: math.log(area_scale(dst[e['id']])/area_scale(e)) for e in source_elements}
    pairs = []
    for idx, a in enumerate(source_elements):
        for b in source_elements[idx+1:]:
            if a['type'] != 'image' and b['type'] != 'image':
                continue
            # Allow a twofold relative scale change during aspect-ratio reflow.
            drift = abs(scales[a['id']]-scales[b['id']])
            penalty = clip((drift-math.log(2))/math.log(4))
            pairs.append({'ids': [a['id'], b['id']], 'penalty': penalty})
    return avg([p['penalty'] for p in pairs]), pairs


def text_flow(source, design):
    """Infer only nearby, aligned, unequal-size text pairs with no intervening text.

    A vertical pair may become a horizontal pair. Absolute positions and empty
    canvas area are not scored. Ambiguous or rotated source pairs are ignored.
    """
    texts = [e for e in source['elements'] if e['type'] == 'text'
             and abs(e.get('rotation', 0)) < 5 and e.get('content', '').strip()]
    target = {e['id']: e for e in design['elements']}
    edges = []
    sw = source['canvas']['width']; sh = source['canvas']['height']
    for a in texts:
        options = []
        for b in texts:
            if a['id'] == b['id'] or a['font_size'] < b['font_size']*1.25:
                continue
            if a.get('group_id') != b.get('group_id'):
                continue
            ab, bb = ink(a), ink(b)
            gap = bb[1]-ab[1]-ab[3]
            alignment = abs(a['x']+a['width']/2-b['x']-b['width']/2)
            if gap < 0 or gap > min(sh*.12, max(ab[3],bb[3])*3) or alignment > sw*.025:
                continue
            if a.get('text_align','left') != b.get('text_align','left'):
                continue
            between = [c for c in texts if c['id'] not in (a['id'],b['id'])
                       and ab[1]+ab[3] < c['y']+c['height']/2 < bb[1]
                       and abs(c['x']+c['width']/2-a['x']-a['width']/2) < sw*.025]
            if not between:
                options.append((gap,b))
        if not options:
            continue
        options.sort(key=lambda x:(x[0],x[1]['id']))
        # Near-equal candidates are ambiguous, so do not invent a relationship.
        if len(options)>1 and options[1][0]-options[0][0] < .01*sh:
            continue
        b=options[0][1]; ab=ink(target[a['id']]);bb=ink(target[b['id']])
        # Permit top-to-bottom or left-to-right reflow; reject only clear reversal.
        forward_y=bb[1] >= ab[1]+ab[3]-.5
        forward_x=bb[0] >= ab[0]+ab[2]-.5
        reversed_y=ab[1] >= bb[1]+bb[3]-.5
        reversed_x=ab[0] >= bb[0]+bb[2]-.5
        order_penalty=float((reversed_y or reversed_x) and not (forward_y or forward_x))
        edges.append({'ids':[a['id'],b['id']], 'relation':'primary_text_before_secondary_text',
                      'penalty':order_penalty, 'axis_change_allowed':True})
    return avg([e['penalty'] for e in edges]),edges


def score(source, design):
    result=_base_score(source, design)
    fg=[e for e in source['elements'] if not is_background(e,source)
        and (e['type']=='text' or (e['type']=='image' and (ROOT/'brand_kit/assets'/e['asset_id']).is_file()))]
    p,pairs=prominence(fg,design['elements'])
    flow,edges=text_flow(source,design)
    old=dict(result['penalties'])
    # Do not dilute explicit containment/date constraints with easy new relations.
    result['penalties']['hierarchy']=max(old['hierarchy'],p)
    # Inferred text-flow is a soft signal, capped at half the relationship budget.
    result['penalties']['relationships']=max(old['relationships'],.5*flow)
    result['points']={k:WEIGHTS[k]*(1-v) for k,v in result['penalties'].items()}
    result['total']=sum(result['points'].values())
    result['version']=VERSION
    result['diagnostics']={'legacy_penalties':old,'cross_type_prominence':{'penalty':p,'pairs':pairs,'relative_scale_tolerance':2},
                           'inferred_text_flow':{'penalty':flow,'edges':edges,'budget_fraction':.5}}
    result['limits']+=['Prominence uses image alpha bounds and font size, not semantic saliency',
                       'Text-flow inference is conservative and may miss valid or broken relationships',
                       'Weights and inferred-relation strength are heuristic, not preference-trained']
    return result

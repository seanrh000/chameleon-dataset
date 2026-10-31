"""Dependency-free geometry encoding, feature extraction and JSON reconstruction."""
from copy import deepcopy
import hashlib
import math
import re

FEATURE_VERSION = 1
HASH_DIM = 32
EDITABLE = {'x', 'y', 'width', 'height', 'font_size', 'rotation'}

def fingerprint(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

def canonical(value):
    # Editor JSON.stringify writes 90.0 as 90. They identify the same design.
    def normalize(v):
        if isinstance(v, float) and v.is_integer(): return int(v)
        if isinstance(v, dict): return {k:normalize(x) for k,x in v.items()}
        if isinstance(v, list): return [normalize(x) for x in v]
        return v
    return __import__('json').dumps(normalize(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def valid_canvas(canvas):
    if not isinstance(canvas, dict) or any(type(canvas.get(k)) is not int or canvas[k] < 1 for k in ('width', 'height')):
        raise ValueError('Canvas width/height must be positive integers.')

def is_background(e, source):
    c=source['canvas']; eps=.01
    return (e['type']!='text' and e.get('group_id') is None and not e.get('rotation',0)
            and e.get('z_index',0)==min(x.get('z_index',0) for x in source['elements'])
            and abs(e['x'])<eps and abs(e['y'])<eps
            and abs(e['width']-c['width'])<eps and abs(e['height']-c['height'])<eps)

def locked_background(e, source, policy):
    return is_background(e,source) and (e['type']=='shape' or policy=='cover' and e.get('fit')=='cover')

def box_bounds(e):
    a=math.radians(e.get('rotation',0));w,h=e['width'],e['height']
    rw=abs(math.cos(a))*w+abs(math.sin(a))*h;rh=abs(math.sin(a))*w+abs(math.cos(a))*h
    return e['x']+(w-rw)/2,e['y']+(h-rh)/2,rw,rh

def layout_rows(design, ids=None):
    w,h=design['canvas']['width'],design['canvas']['height']; es={e['id']:e for e in design['elements']}
    rows=[]
    for eid in ids or list(es):
        e=es[eid];a=math.radians(e.get('rotation',0))
        rows.append([(e['x']+e['width']/2)/w,(e['y']+e['height']/2)/h,
                     e['width']/w,e['height']/h,e.get('font_size',1)/min(w,h),math.sin(a),math.cos(a)])
    return rows

def baseline_rows(source, size, policy='cover'):
    valid_canvas(size); sw,sh=source['canvas']['width'],source['canvas']['height']
    tw,th=size['width'],size['height'];scale=min(tw/sw,th/sh)
    d=deepcopy(source);d['canvas']={**source['canvas'],**size}
    for e,s in zip(d['elements'],source['elements']):
        e.update(x=s['x']*scale+(tw-sw*scale)/2,y=s['y']*scale+(th-sh*scale)/2,width=s['width']*scale,height=s['height']*scale)
        if e['type']=='text':e['font_size']=max(1,s['font_size']*scale)
        if locked_background(s,source,policy):e.update(x=0,y=0,width=tw,height=th)
    return layout_rows(d)

def color_features(value):
    s=str(value or '')
    try:
        if s.startswith('#'):
            code=s[1:];code=''.join(c*2 for c in code) if len(code)==3 else code
            return [int(code[i:i+2],16)/255 for i in (0,2,4)]+[1.]
        nums=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',s)]
        if len(nums)>=3:return [max(0.,min(1.,x/255)) for x in nums[:3]]+[nums[3] if len(nums)>3 else 1.]
    except (ValueError,IndexError):pass
    return [0.,0.,0.,0.]

def features(source, target_canvas):
    """Only source and requested size enter features; IDs and annotations never do."""
    valid_canvas(source['canvas']);valid_canvas(target_canvas)
    sw,sh=source['canvas']['width'],source['canvas']['height'];tw,th=target_canvas['width'],target_canvas['height']
    groups={}; membership=[]
    for i,e in enumerate(source['elements']):
        key=('group',e['group_id']) if e.get('group_id') is not None else ('single',i)
        if key not in groups:groups[key]=len(groups)
        membership.append(groups[key])
    result=[];geo=layout_rows(source)
    for i,e in enumerate(source['elements']):
        text=str(e.get('content','')); font=e.get('font_family','')
        words=re.findall(r'\w+',text.lower()); hashed=[0.]*HASH_DIM
        for word in words:
            # Stable hashing without a learned vocabulary fitted on validation/test text.
            digest=hashlib.sha256(word.encode()).digest();idx=int.from_bytes(digest[:4],'little')%HASH_DIM
            hashed[idx]+=1. if digest[4]&1 else -1.
        norm=max(1.,math.sqrt(sum(x*x for x in hashed)));hashed=[x/norm for x in hashed]
        same=[other for j,other in enumerate(source['elements']) if membership[j]==membership[i]]
        gx=min(x['x'] for x in same);gy=min(x['y'] for x in same)
        gw=max(x['x']+x['width'] for x in same)-gx;gh=max(x['y']+x['height'] for x in same)-gy
        numeric=geo[i]+[
            math.log(max(1e-8,e['width']/e['height'])),e.get('opacity',1),e.get('z_index',0)/100,
            e.get('line_height',1.2),e.get('letter_spacing',0)/max(1,e.get('font_size',1)),
            e.get('stroke_width',0)/min(sw,sh),e.get('corner_radius',0)/min(sw,sh),
            math.log1p(len(text))/10,math.log1p(len(words))/5,math.log1p(text.count('\n'))/5,
            float(e.get('font_weight')=='bold'),float(e.get('font_style')=='italic'),
            float(font=='Solstice Sans'),float(font=='Solstice Serif'),float(bool(e.get('style_runs'))),
            float(is_background(e,source)),len(same)/len(source['elements']),gx/sw,gy/sh,gw/sw,gh/sh,
        ]
        for key,choices in [('type',['text','image','shape']),('shape_type',['rectangle','ellipse','line']),
                            ('fit',['contain','cover','fill']),('text_align',['left','center','right'])]:
            numeric.extend(float(e.get(key, 'contain' if key=='fit' else 'left' if key=='text_align' else '')==v) for v in choices)
        numeric+=color_features(e.get('color',e.get('fill_color')))+color_features(e.get('stroke_color'))+hashed
        if not all(math.isfinite(x) for x in numeric):raise ValueError('Nonfinite input feature')
        result.append(numeric)
    canvas=[math.log(sw)/10,math.log(sh)/10,math.log(tw)/10,math.log(th)/10,
            math.log(sw/sh),math.log(tw/th),math.log(tw/sw),math.log(th/sh)]
    return result,canvas,membership

_EXAMPLE={'canvas':{'width':100,'height':100},'elements':[{'id':'x','type':'shape','x':0,'y':0,'width':10,'height':10}]}
FEATURE_DIM=len(features(_EXAMPLE, _EXAMPLE['canvas'])[0][0])

def reconstruct(source, size, rows, policy='cover'):
    """Return official design structure; project boxes to rotated canvas bounds.

    This is geometry repair, not typography/contrast/intent validation.
    """
    if policy not in ('strict','cover'):raise ValueError('Unknown background policy')
    valid_canvas(size)
    if len(rows)!=len(source['elements']):raise ValueError('Prediction count does not match source elements')
    out=deepcopy(source);out['canvas']={**source['canvas'],**size};tw,th=size['width'],size['height'];warnings=[]
    for e,s,row in zip(out['elements'],source['elements'],rows):
        if len(row)!=7 or not all(math.isfinite(float(v)) for v in row):raise ValueError('Invalid model geometry')
        cx,cy,w,h,font,sin,cos=map(float,row);cx*=tw;cy*=th
        w=max(1e-4,min(w,1e4)*tw);h=max(1e-4,min(h,1e4)*th)
        rotation=math.degrees(math.atan2(sin,cos)) if abs(sin)+abs(cos)>1e-8 else s.get('rotation',0)
        if e['type']=='image':
            scale=min(w/s['width'],h/s['height']);w,h=s['width']*scale,s['height']*scale
        a=math.radians(rotation);rw=abs(math.cos(a))*w+abs(math.sin(a))*h;rh=abs(math.sin(a))*w+abs(math.cos(a))*h
        shrink=min(1.,tw/max(rw,1e-8),th/max(rh,1e-8));w*=shrink;h*=shrink;rw*=shrink;rh*=shrink
        cx=min(max(cx,rw/2),tw-rw/2);cy=min(max(cy,rh/2),th-rh/2)
        e.update(x=cx-w/2,y=cy-h/2,width=w,height=h)
        if 'rotation' in s or abs(rotation)>1e-6:e['rotation']=rotation
        if e['type']=='text':e['font_size']=max(1.,min(font,1e4)*min(tw,th)*shrink)
        if locked_background(s,source,policy):
            e.update(x=0.,y=0.,width=float(tw),height=float(th))
            if 'rotation' in e:e['rotation']=0.
            if e['type']=='image' and not math.isclose(tw/th,s['width']/s['height'],rel_tol=1e-5):
                warnings.append(f"{e['id']}: cover background box changed; central crop. Competition policy needs confirmation.")
    return out,warnings

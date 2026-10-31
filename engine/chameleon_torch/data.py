"""Read editor ZIPs without extracting or modifying them; split by connected families."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import zipfile
import hashlib
import math
from .geometry import canonical, fingerprint, EDITABLE, box_bounds, locked_background, valid_canvas

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA=ROOT/'chameleon_design_schema_v0.9.json'
MAX_JSON_BYTES=8*1024*1024

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as file:json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)

def read_zip(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member names')
        data={}
        for name in ('source.json','target.json','metadata.json'):
            info=archive.getinfo(name)
            if info.file_size>MAX_JSON_BYTES:raise ValueError(f'{name} is too large')
            raw=archive.read(name);data[name[:-5]]=json.loads(raw.decode('utf-8-sig'))
            data[name+'_sha256']=hashlib.sha256(raw).hexdigest()
        # The original input.json may contain an OLD target size. Never use it for labels.
        review=data['metadata'].get('official_review',{})
        for kind in ('source','target'):
            record=review.get(kind,{})
            if record.get('saved_json_sha256') and record['saved_json_sha256']!=data[kind+'.json_sha256']:
                raise ValueError(f'{kind}: reviewed JSON hash does not match saved file')
            png=f'official/{kind}.png'
            if png in names and record.get('png_sha256'):
                if archive.getinfo(png).file_size>40*1024*1024:raise ValueError('Preview too large')
                if hashlib.sha256(archive.read(png)).hexdigest()!=record['png_sha256']:raise ValueError('Reviewed PNG hash mismatch')
        return data

def validate_design(design, schema_path=DEFAULT_SCHEMA, max_elements=500):
    import jsonschema
    validator=jsonschema.Draft202012Validator(read_json(schema_path))
    for label,d in [('design',design)]:
        issues=list(validator.iter_errors(d))
        if issues:raise ValueError(f'{label}: schema: {issues[0].message}')
        valid_canvas(d['canvas'])
        if len(d['elements'])>max_elements:raise ValueError(f'{label}: more than {max_elements} elements')
        if len({e['id'] for e in d['elements']})!=len(d['elements']):raise ValueError(f'{label}: duplicate IDs')
        for e in d['elements']:
            for key in ('x','y','width','height','rotation','font_size','opacity','z_index','line_height','letter_spacing','corner_radius','stroke_width'):
                if key in e and (isinstance(e[key],bool) or not isinstance(e[key],(int,float)) or not math.isfinite(e[key])):raise ValueError(f'{label}/{e["id"]}: invalid {key}')
            if min(e['width'],e['height'])<=0:raise ValueError(f'{label}: zero/negative element dimensions unsupported')


def validate_pair(source,target,policy='cover',schema_path=DEFAULT_SCHEMA,max_elements=500):
    if policy not in ('cover','strict'):raise ValueError('Invalid background policy')
    validate_design(source,schema_path,max_elements)
    validate_design(target,schema_path,max_elements)
    errors=[];warnings=[]
    src={e['id']:e for e in source['elements']};dst={e['id']:e for e in target['elements']}
    if src.keys()!=dst.keys():raise ValueError('Source/target IDs differ')
    for eid,s in src.items():
        e=dst[eid]
        if {k:v for k,v in s.items() if k not in EDITABLE}!={k:v for k,v in e.items() if k not in EDITABLE}:
            errors.append(f'{eid}: immutable attributes changed')
        x,y,w,h=box_bounds(e);tw,th=target['canvas']['width'],target['canvas']['height']
        if x<-.05 or y<-.05 or x+w>tw+.05 or y+h>th+.05:errors.append(f'{eid}: target rotated bounds outside canvas')
        if e['type']=='image' and not math.isclose(e['width']/e['height'],s['width']/s['height'],rel_tol=1e-4):
            if (locked_background(s,source,policy) and e.get('fit')=='cover'
                    and abs(e['x'])<.05 and abs(e['y'])<.05 and not e.get('rotation',0)
                    and abs(e['width']-tw)<.05 and abs(e['height']-th)<.05):warnings.append(f'{eid}: cover background box ratio differs')
            else:errors.append(f'{eid}: image aspect ratio changed')
    if errors:raise ValueError('; '.join(errors))
    return warnings

def catalog_families(paths):
    """Known synthetic recipe families, matched by SOURCE geometry/content only."""
    index={}
    for root in paths:
        root=Path(root)
        if not root.exists():continue
        for source_file in sorted(root.glob('samples/*/source.json')):
            ann=source_file.with_name('annotations.json')
            if ann.exists():
                family=read_json(ann).get('design_family_id')
                if family:index.setdefault(fingerprint(read_json(source_file)),set()).add(str(root.resolve())+'::'+family)
    return index

def inspect_pairs(data_dir,policy='cover',schema_path=DEFAULT_SCHEMA,allow_draft=False,catalogs=(),max_elements=500,family_map=None):
    catalog=catalog_families(catalogs);accepted=[];rejected=[];seen=set()
    family_map=family_map or {}
    if not isinstance(family_map,dict) or any(not isinstance(v,str) or not v.strip() for v in family_map.values()):
        raise ValueError('family-map must map ZIP filenames to nonempty family names')
    available={p.name for p in Path(data_dir).glob('*.zip')}
    if set(family_map)-available:raise ValueError('family-map contains unknown ZIP filenames')
    for path in sorted(Path(data_dir).glob('*.zip')):
        try:
            d=read_zip(path);s,t,m=d['source'],d['target'],d['metadata']
            if m.get('review_status')!='human_approved' and not allow_draft:raise ValueError('Not human_approved')
            warnings=validate_pair(s,t,policy,schema_path,max_elements)
            sf=fingerprint(s);tf=fingerprint(t);duplicate=(sf,tf)
            if duplicate in seen:raise ValueError('Duplicate source/target pair')
            seen.add(duplicate)
            family=m.get('design_family_id')
            if not isinstance(family,str) or not family.strip():raise ValueError('Missing design_family_id')
            tokens=['source:'+sf,'metadata:'+family]+['catalog:'+f for f in sorted(catalog.get(sf,[]))]
            if path.name in family_map:tokens.append('manual:'+family_map[path.name])
            if not catalog.get(sf) and path.name not in family_map:warnings.append('Recipe family not resolved; verify design_family_id before splitting variants.')
            if m.get('target_canvas') and m['target_canvas']!=t['canvas']:warnings.append('Metadata target canvas differs; target.json canvas is authoritative.')
            accepted.append({'path':str(path.resolve()),'zip_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'sample_id':m.get('sample_id',path.stem),'source_hash':sf,'target_hash':tf,'family_tokens':tokens,
                'source_canvas':s['canvas'],'target_canvas':t['canvas'],'element_count':len(s['elements']),
                'review_status':m.get('review_status'),'warnings':warnings})
        except (ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
            rejected.append({'path':str(path.resolve()),'reason':str(exc)})
    # Transitive family closure: metadata families, identical sources, and catalog recipes.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for row in accepted:
        tokens=row['family_tokens']
        for x in tokens[1:]:parent[find(x)]=find(tokens[0])
    components={}
    for row in accepted:components.setdefault(find(row['family_tokens'][0]),[]).append(row)
    for component in components.values():
        group=fingerprint(sorted({t for row in component for t in row['family_tokens']}))[:20]
        for row in component:row['split_group']=group
    return accepted,rejected

def split_manifest(rows,rejected,seed=20261026,val_fraction=.15,test_fraction=.15,policy='cover'):
    if not 0<val_fraction<1 or not 0<test_fraction<1 or val_fraction+test_fraction>=1:raise ValueError('Invalid split fractions')
    families=sorted({r['split_group'] for r in rows})
    if len(families)<3:raise ValueError('At least 3 independent design families are required for train/val/test.')
    random.Random(seed).shuffle(families);n=len(families)
    nv=max(1,round(n*val_fraction));nt=max(1,round(n*test_fraction))
    while nv+nt>=n:
        if nv>=nt and nv>1:nv-=1
        elif nt>1:nt-=1
        else:raise ValueError('Not enough families for the requested split')
    partition={f:('val' if i<nv else 'test' if i<nv+nt else 'train') for i,f in enumerate(families)}
    samples=[{**r,'split':partition[r['split_group']]} for r in rows]
    return {'format_version':1,'seed':seed,'background_policy':policy,'samples':samples,'rejected':rejected,
            'summary':{'accepted':len(rows),'rejected':len(rejected),'families':n,'pairs_per_split':dict(Counter(r['split'] for r in samples)),
                       'families_per_split':dict(Counter(partition.values()))}}

def load_manifest(path,split=None,verify=True):
    m=read_json(path);samples=m['samples'];family_splits={};sources={};seen=set()
    tokens={}
    if m.get('format_version')!=1:raise ValueError('Unsupported manifest version')
    for row in samples:
        if row['split'] not in ('train','val','test'):raise ValueError('Invalid split')
        for key,index in [('split_group',family_splits),('source_hash',sources)]:
            if row[key] in index and index[row[key]]!=row['split']:raise ValueError('Data leakage across splits')
            index[row[key]]=row['split']
        for token in row.get('family_tokens',[]):
            if token in tokens and tokens[token]!=row['split']:raise ValueError('Family token leakage across splits')
            tokens[token]=row['split']
        if (row['source_hash'],row['target_hash']) in seen:raise ValueError('Duplicate pair in manifest')
        seen.add((row['source_hash'],row['target_hash']))
        if verify and hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['zip_sha256']:
            raise ValueError(f'Dataset file changed since preparation: {row["path"]}')
    return m,[r for r in samples if split is None or r['split']==split]

"""Geometric effect relationships and low-resolution visibility estimates.

No asset pixels or output styles are changed. Masks are temporary scoring
data, not generated artwork. Relationships are hypotheses, not semantics.
"""
from dataclasses import dataclass
import math
from PIL import Image, ImageChops, ImageStat

from renderer.converter.assets import load_asset_image
from renderer.converter.colorutil import parse_rgba
from renderer.converter.shapes import render_shape


@dataclass(frozen=True)
class Effect:
    kind: str
    anchor: int | None
    attachment: float
    canvas: float


def infer_effects(elements, backgrounds, visible, width, height):
    """Use direct image relationships only; never merge transitive chains."""
    images = [i for i, e in enumerate(elements) if e['type'] == 'image'
              and i not in backgrounds and not e.get('rotation') and e.get('group_id') is None]
    result = {}
    for i, e in enumerate(elements):
        if e['type'] != 'shape' or i in backgrounds or e.get('group_id') is not None:
            continue
        x, y, w, h = (e[k] for k in ('x', 'y', 'width', 'height'))
        if min(w, h) <= 0:
            continue
        # Text backdrops are content, even when translucent and ungrouped.
        if any(t['type'] == 'text' and t.get('z_index', 0) > e.get('z_index', 0)
               and t['width']*t['height'] >= .2*w*h
               and t['x'] >= x and t['y'] >= y
               and t['x']+t['width'] <= x+w and t['y']+t['height'] <= y+h
               for t in elements):
            continue
        alpha = e.get('opacity', 1.)*parse_rgba(e.get('fill_color'))[3]
        area = w*h/(width*height)
        cx, cy = x+w/2, y+h/2
        edge = x <= 0 or y <= 0 or x+w >= width or y+h >= height
        peripheral = cx/width < .25 or cx/width > .75 or cy/height < .25 or cy/height > .75
        behind = any(t['type'] in ('image', 'text') and j not in backgrounds
                     and t.get('z_index', 0) > e.get('z_index', 0) for j, t in enumerate(elements))
        canvas = 0.
        if behind and area >= .035 and (edge or peripheral):
            canvas = .7 + .15*(alpha <= .35) + .1*edge
        best, anchor = 0., None
        for j in images:
            a = elements[j]
            ax, ay, aw, ah = visible(j, a)
            if min(aw, ah) <= 0:
                continue
            dx, dy = abs(cx-ax-aw/2)/aw, abs(cy-ay-ah/2)/ah
            overlap = (max(0., min(x+w, ax+aw)-max(x, ax)) *
                       max(0., min(y+h, ay+ah)-max(y, ay)))/(w*h)
            score = 0.
            if (e.get('z_index', 0) < a.get('z_index', 0) and alpha <= .6
                    and .35 <= w/aw <= 1.8 and h/ah <= .3
                    and dx <= .18 and abs(cy-ay-ah)/ah <= .14):
                score = .95  # aligned, shallow shadow immediately under image
            elif overlap >= .8 and dx <= .35 and dy <= .4 and .02 <= w*h/(aw*ah) <= 1.4:
                score = .88  # local component or reflection inside the image
            elif (alpha <= .35 and e.get('z_index', 0) < a.get('z_index', 0)
                  and dx <= .3 and dy <= .3 and .5 <= w*h/(aw*ah) <= 4):
                score = .65  # centered halo: intentionally left ambiguous
            if score > best:
                best, anchor = score, j
        if best >= .85 and best-canvas >= .2:
            result[i] = Effect('attached', anchor, best, canvas)
        elif best >= .55:
            result[i] = Effect('ambiguous', anchor, best, max(.5, canvas))
        elif canvas >= .7:
            result[i] = Effect('canvas', None, 0., canvas)
        # Uncertain ordinary shapes stay in the general layout search.
    return result


def alpha_mask(e, canvas, assets, longest=256):
    """Approximate drawn alpha, including rotation and image transparency.

    Text uses its measured box (supplied by caller), not a glyph mask.
    All operations are in memory, at bounded thumbnail resolution.
    """
    scale = longest/max(canvas['width'], canvas['height'])
    size = tuple(max(1, round(canvas[k]*scale)) for k in ('width', 'height'))
    w, h = max(1, round(e['width']*scale)), max(1, round(e['height']*scale))
    small = dict(e, width=w, height=h,
                 corner_radius=e.get('corner_radius', 0)*scale,
                 stroke_width=e.get('stroke_width', 1)*scale)
    if e['type'] == 'image':
        tile = load_asset_image(assets, e.get('asset_id'), w, h, e.get('fit', 'contain')).getchannel('A')
        alpha = e.get('opacity', 1.)
    elif e['type'] == 'shape':
        tile = render_shape(small).getchannel('A')
        alpha = e.get('opacity', 1.)*parse_rgba(e.get('fill_color'))[3]
    else:
        tile = Image.new('L', (w, h), 255)
        alpha = e.get('opacity', 1.)*parse_rgba(e.get('color'))[3]
    tile = tile.point(lambda p: round(p*min(1., max(0., alpha))))
    if e.get('rotation'):
        tile = tile.rotate(-e['rotation'], resample=Image.Resampling.BICUBIC, expand=True)
    mask = Image.new('L', size)
    cx, cy = (e['x']+e['width']/2)*scale, (e['y']+e['height']/2)*scale
    mask.paste(tile, (round(cx-tile.width/2), round(cy-tile.height/2)))
    return mask


class Visibility:
    """Cache foreground transmittance for a fixed layout and each z-layer."""
    def __init__(self, design, effects, backgrounds, assets, visible):
        self.design, self.assets = design, assets
        self.canvas = design['canvas']
        self.effects = effects
        self.masks = {}
        for i, e in enumerate(design['elements']):
            if i in effects or i in backgrounds:
                continue
            if e['type'] == 'text':
                x, y, w, h = visible(i, e)
                e = dict(e, x=x, y=y, width=w, height=h, rotation=0)
            self.masks[i] = alpha_mask(e, self.canvas, assets)
        self.transmission = {}

    def mask(self, index, e):
        raw = alpha_mask(e, self.canvas, self.assets)
        # Match renderer's stable z-order, including equal-z input order.
        order = (e.get('z_index', 0), index)
        if order not in self.transmission:
            free = Image.new('L', raw.size, 255)
            for j, mask in self.masks.items():
                other = self.design['elements'][j]
                if (other.get('z_index', 0), j) > order:
                    free = ImageChops.multiply(free, ImageChops.invert(mask))
            self.transmission[order] = free
        return ImageChops.multiply(raw, self.transmission[order])

    def features(self, index, e):
        mask = self.mask(index, e)
        amount = ImageStat.Stat(mask).mean[0]/255
        raw = ImageStat.Stat(alpha_mask(e, self.canvas, self.assets)).mean[0]/255
        # Broad spatial distribution, not a demand for identical pixel layout.
        w, h = mask.size
        cells = []
        for yy in range(2):
            for xx in range(3):
                region = mask.crop((xx*w//3, yy*h//2, (xx+1)*w//3, (yy+1)*h//2))
                cells.append(ImageStat.Stat(region).mean[0]/255 if region.width*region.height else 0.)
        total = sum(cells)
        return amount, tuple(v/max(1e-9, total) for v in cells), amount/max(1e-9, raw)


def visibility_loss(reference, current):
    area, distribution, exposed = reference
    actual, now, visible = current
    # Ignore tiny, almost invisible effects instead of magnifying raster noise.
    if area < .0002:
        return max(0., actual-area)*20
    ratio = max(1e-6, actual)/area
    amount = max(0., abs(math.log(ratio))-.18)**2
    spatial = sum((a-b)**2 for a, b in zip(distribution, now))
    occlusion = max(0., abs(visible-exposed)-.08)**2
    return 2.*amount + .8*spatial + 4.*occlusion

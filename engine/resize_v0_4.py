"""Offline layout search: grouping, measured typography, candidate refinement.

Only geometry and font_size change. The explicit scoring function is a
heuristic, not a trained aesthetic model or the official competition judge.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any

import skia
from PIL import Image
from engine.effects import infer_effects, Visibility, visibility_loss

from renderer.converter.colorutil import parse_rgba
from renderer.converter.assets import load_asset_image
from renderer.converter.convert import (
    _get_renderer, _normalize_weight, _weight_style_to_font_style,
    convert_chameleon_to_crello,
)
from renderer.converter.wrap import char_widths, wrap_text

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6
VERSION = "0.4"
Rect = tuple[float, float, float, float]


def _bounds(e: dict) -> Rect:
    """Union of declared and rotated boxes: both must stay on canvas."""
    x, y, w, h = (float(e[k]) for k in ("x", "y", "width", "height"))
    a = math.radians(e.get("rotation", 0))
    rw = max(w, abs(math.cos(a))*w + abs(math.sin(a))*h)
    rh = max(h, abs(math.sin(a))*w + abs(math.cos(a))*h)
    return x+(w-rw)/2, y+(h-rh)/2, rw, rh


def _union(rects: list[Rect]) -> Rect:
    x, y = min(r[0] for r in rects), min(r[1] for r in rects)
    return x, y, max(r[0]+r[2] for r in rects)-x, max(r[1]+r[3] for r in rects)-y


def _intersection(a: Rect, b: Rect) -> float:
    return max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))


def _contains(a: Rect, b: Rect, tolerance: float = 0.) -> bool:
    return (b[0] >= a[0]-tolerance and b[1] >= a[1]-tolerance
            and b[0]+b[2] <= a[0]+a[2]+tolerance
            and b[1]+b[3] <= a[1]+a[3]+tolerance)


@dataclass(frozen=True)
class Unit:
    indices: tuple[int, ...]
    box: Rect
    kind: str
    mass: float


@dataclass(frozen=True)
class Split:
    axis: str
    children: tuple[Any, ...]


def _split(axis: str, children: list) -> Any:
    return children[0] if len(children) == 1 else Split(axis, tuple(children))


class _Typography:
    """Match the converter's wrapping and cr-renderer's styled glyph advances.

    Font sizes are integers: cr-renderer casts fontSize to int when drawing,
    but not when wrapping. Caches are bounded and scoped to one resize call.
    """

    def __init__(self, elements: list[dict], font_manager):
        self.elements, self.fm = elements, font_manager
        self.glyphs = lru_cache(maxsize=4096)(self._glyphs)
        self.measure = lru_cache(maxsize=24000)(self._measure)

    def _glyphs(self, index: int, size: int):
        e = self.elements[index]
        text, family = e.get("content", ""), e.get("font_family", "Montserrat")
        weight = _normalize_weight(e.get("font_weight"))
        bold = weight in {"bold", "semibold", "extrabold", "black"}
        italic = (e.get("font_style") or e.get("italic")) in ("italic", True)
        style = _weight_style_to_font_style(bold, italic)
        wrapping = char_widths(self.fm, family, weight, style, size, text)
        flags = [(bold, italic)]*len(text)
        # The bundled converter treats end as inclusive. Match its actual
        # output without rewriting the original style-run metadata.
        for run in e.get("style_runs") or []:
            for j in range(max(0, run.get("start", 0)), min(len(text), run.get("end", len(text)-1)+1)):
                b, it = flags[j]
                if "font_weight" in run:
                    b = _normalize_weight(run["font_weight"]) in {"bold", "semibold", "extrabold", "black"}
                if "font_style" in run or "italic" in run:
                    it = (run.get("font_style") or run.get("italic")) in ("italic", True)
                flags[j] = b, it
        variants, height = {}, 0.
        for b, it in set(flags) or {(bold, italic)}:
            st, wt = _weight_style_to_font_style(b, it), "bold" if b else "regular"
            font = skia.Font(skia.Typeface.MakeFromData(self.fm.lookup(family, wt, st)), size)
            variants[b, it] = [
                sum(font.getWidths(font.textToGlyphs(c.upper() if e.get("capitalize") else c)))
                for c in text
            ]
            m = font.getMetrics()
            height = max(height, m.fDescent-m.fAscent)
        return wrapping, [variants[flags[j]][j] for j in range(len(text))], height

    def _measure(self, index: int, size: int, width: float) -> tuple[float, float, int]:
        e = self.elements[index]
        text = e.get("content", "")
        if not text:
            return 0., 0., 0
        wrapping, actual, glyph_height = self.glyphs(index, size)
        lines = wrap_text(text, wrapping, width).split("\n")
        start, max_width = 0, 0.
        for line in lines:
            # Conservative across style spans, where spacing restarts.
            spacing = max(0., float(e.get("letter_spacing", 0)))*max(0, len(line)-1)
            max_width = max(max_width, sum(actual[start:start+len(line)])+spacing)
            start += len(line)+1
        height = (len(lines)-1)*size*float(e.get("line_height", 1.2))+glyph_height
        return max_width, height, len(lines)

    def fit(self, index: int, width: float, height: float, cap: float) -> int:
        width = max(EPS, width)
        limit = self.elements[index].get("max_lines") or math.inf
        lo, hi, best = 1, max(1, int(cap)), 1
        while lo <= hi:
            mid = (lo+hi)//2
            w, h, lines = self.measure(index, mid, width)
            if w <= width*.985 and h <= height*.985 and lines <= limit:
                best, lo = mid, mid+1
            else:
                hi = mid-1
        return best


class _Search:
    def __init__(self, source: dict, target: dict):
        self.source, self.elements = source, source["elements"]
        self.canvas = {k: target[k] for k in ("width", "height")}
        self.w, self.h = (float(target[k]) for k in ("width", "height"))
        self.sw, self.sh = (float(source["canvas"][k]) for k in ("width", "height"))
        self.scale = math.sqrt(self.w*self.h/(self.sw*self.sh))
        self.aspect_change = abs(math.log((self.w/self.h)/(self.sw/self.sh)))
        self.renderer = _get_renderer(str(ROOT/"brand_kit"/"fonts"))
        self.type = _Typography(self.elements, self.renderer.font_manager)
        self.backgrounds, self.decorations = set(), set()
        self.ink = self._image_footprints()
        self.effects = {}
        self.units = self._units()
        self.membership = {i: n for n, u in enumerate(self.units) for i in u.indices}
        self.margin, self.gap = min(self.w, self.h)*.035, min(self.w, self.h)*.025
        self.rect = (self.margin, self.margin, self.w-2*self.margin, self.h-2*self.margin)
        fonts = [e["font_size"] for e in self.elements if e["type"] == "text"]
        self.median_font = median(fonts) if fonts else 1.
        self.text_relations = []
        text_units = [j for j, u in enumerate(self.units)
                      if any(self.elements[i]["type"] == "text" for i in u.indices)]
        for pos, j in enumerate(text_units):
            ax, ay, aw, ah = self.units[j].box
            for k in text_units[pos+1:]:
                bx, by, bw, bh = self.units[k].box
                # Adjacent, centered copy carries a stronger relationship
                # than mere reading order. Do not merge it into a rigid group.
                if (abs((ax+aw/2)-(bx+bw/2)) < .08*self.sw
                        and abs((ay+ah/2)-(by+bh/2)) < .2*self.sh):
                    self.text_relations.append((j, k))
        self.roles = self._roles()

    def _image_footprints(self) -> dict[int, Rect]:
        """Measure rendered alpha, including contain/cover padding.

        These bounds are for visual spacing only. The complete JSON box
        still has to fit the canvas. No source asset pixels are modified.
        """
        result = {}
        for i, e in enumerate(self.elements):
            if e["type"] != "image" or i in self.backgrounds:
                continue
            scale = min(1., 512/max(1., e["width"], e["height"]))
            im = load_asset_image(str(ROOT/"brand_kit/assets"), e.get("asset_id"),
                                  max(1., e["width"]*scale), max(1., e["height"]*scale), e.get("fit", "contain"))
            alpha = im.getchannel("A").point(lambda v: 255 if v >= 16 else 0)
            box = alpha.getbbox()
            if box:
                x, y, right, bottom = box
                result[i] = x/im.width, y/im.height, (right-x)/im.width, (bottom-y)/im.height
            else:
                result[i] = (0., 0., 1., 1.)
        return result

    def _visible(self, i: int, e: dict) -> Rect:
        if i in self.ink:
            fx, fy, fw, fh = self.ink[i]
            x, y, w, h = e["x"]+fx*e["width"], e["y"]+fy*e["height"], fw*e["width"], fh*e["height"]
        elif e["type"] == "text":
            w, h, _ = self.type.measure(i, max(1, int(e["font_size"])), e["width"])
            offset = {"left": 0., "center": .5, "right": 1.}.get(e.get("text_align"), 0.)
            x, y = e["x"]+offset*(e["width"]-w), e["y"]
        else:
            return _bounds(e)
        angle = math.radians(e.get("rotation", 0))
        cx, cy = e["x"]+e["width"]/2, e["y"]+e["height"]/2
        points = [(cx+(px-cx)*math.cos(angle)-(py-cy)*math.sin(angle),
                   cy+(px-cx)*math.sin(angle)+(py-cy)*math.cos(angle))
                  for px in (x, x+w) for py in (y, y+h)]
        left, top = min(p[0] for p in points), min(p[1] for p in points)
        return left, top, max(p[0] for p in points)-left, max(p[1] for p in points)-top

    def _unit_ink(self, j: int, elements: list[dict]) -> Rect:
        return _union([self._visible(i, elements[i]) for i in self.units[j].indices])

    def _roles(self) -> dict:
        """Conservative geometric role hypotheses; IDs/copy are opaque.

        These are proposal hints, not semantic ground truth. All units remain
        present; generic layouts provide a fallback for uncertain roles or
        infeasible campaign proposals.
        """
        text = [j for j, u in enumerate(self.units) if u.kind == "text"]
        images = [j for j, u in enumerate(self.units) if u.kind == "image"]
        groups = [j for j, u in enumerate(self.units) if u.kind == "group"]
        font = lambda j: max(self.elements[i].get("font_size", 0) for i in self.units[j].indices)
        area = lambda j: math.prod(self._unit_ink(j, self.elements)[2:])
        head = max(text, key=font) if text else None
        footer = [j for j in text if head is not None and j != head
                  and self.units[j].box[1] > .85*self.sh and font(j) < .7*self.median_font]
        body = sorted([j for j in text if j not in footer], key=lambda j: self.units[j].box[1])
        if head is not None:
            body = [head]+[j for j in body if j != head]
        # A dominant image may be the hero. Wide top images are branding
        # candidates; missing assets are kept as secondary imagery.
        existing = [j for j in images if all((ROOT/"brand_kit/assets"/self.elements[i].get("asset_id", "")).is_file() for i in self.units[j].indices)]
        hero = max(existing, key=area) if existing else None
        brand = [j for j in existing if j != hero and self.units[j].box[1] < .25*self.sh]
        secondary = [j for j in images if j != hero and j not in brand]
        corner = [j for j in groups if self.units[j].box[1] < .2*self.sh]
        lower = [j for j in groups if j not in corner and self.units[j].box[1] > .5*self.sh
                 and any(self.elements[i]["type"] == "text" for i in self.units[j].indices)]
        action = max(lower, key=lambda j: self.units[j].box[2]*self.units[j].box[3]) if lower else None
        badges = sorted([j for j in groups if j not in corner and j != action], key=lambda j: self.units[j].box[0])
        return dict(head=head, body=body, footer=footer, hero=hero, brand=brand,
                    secondary=secondary, corner=corner, action=action, badges=badges)

    def _units(self) -> list[Unit]:
        groups = {}
        min_z = min((e.get("z_index", 0) for e in self.elements), default=0)
        for i, e in enumerate(self.elements):
            box = tuple(e[k] for k in ("x", "y", "width", "height"))
            full = _contains(box, (0., 0., self.sw, self.sh), .5)
            # IDs are opaque. Classify only by geometry, type and layering.
            if (full and e["type"] != "text" and e.get("group_id") is None
                    and not e.get("rotation") and e.get("z_index", 0) == min_z):
                self.backgrounds.add(i)
        self.effects = infer_effects(self.elements, self.backgrounds, self._visible, self.sw, self.sh)
        self.decorations = set(self.effects)
        for i, e in enumerate(self.elements):
            if i in self.backgrounds or i in self.decorations:
                continue
            key = ("group", e["group_id"]) if e.get("group_id") is not None else ("element", i)
            groups.setdefault(key, []).append(i)
        # Infer an ungrouped label/backdrop pair conservatively.
        for i, e in enumerate(self.elements):
            key = ("element", i)
            if e["type"] != "text" or key not in groups:
                continue
            choices = []
            for j, shape in enumerate(self.elements):
                skey = ("element", j)
                if (shape["type"] == "shape" and skey in groups and not shape.get("rotation")
                        and not e.get("rotation") and shape.get("z_index", 0) < e.get("z_index", 0)
                        and shape["width"]*shape["height"] < .2*self.sw*self.sh
                        and _contains(_bounds(shape), _bounds(e))):
                    choices.append((shape["width"]*shape["height"], skey))
            if choices:
                groups[min(choices)[1]].extend(groups.pop(key))
        units = []
        for indices in groups.values():
            box = _union([_bounds(self.elements[i]) for i in indices])
            es = [self.elements[i] for i in indices]
            kind = "text" if all(e["type"] == "text" for e in es) else "image" if all(e["type"] == "image" for e in es) else "group"
            text_mass = sum(max(4, len(e.get("content", "")))*(e.get("font_size", 1)*self.scale)**2*.55 for e in es if e["type"] == "text")
            mass = max(text_mass, box[2]*box[3]*self.scale**2*(.65 if kind == "image" else .35), EPS)
            units.append(Unit(tuple(indices), box, kind, mass))
        return units

    def _place(self, unit: Unit, rect: Rect, font_scale: float | None = None) -> dict[int, dict]:
        x, y, w, h = rect
        bx, by, bw, bh = unit.box
        es = {i: dict(self.elements[i]) for i in unit.indices}
        if (self.aspect_change >= .45 and len(es) == 1 and unit.kind == "image"
                and not next(iter(es.values())).get("rotation")):
            i, e = next(iter(es.items()))
            fx, fy, fw, fh = self.ink.get(i, (0., 0., 1., 1.))
            scale = min(w/max(EPS, bw*fw), h/max(EPS, bh*fh),
                        self.w/max(EPS, bw), self.h/max(EPS, bh), self.scale*2.2)
            # Both the alpha footprint and the full asset must fit. Clamping
            # a large padded image afterwards can push its ink into a neighbor.
            def intervals(s):
                return (max(0., x-fx*bw*s), min(self.w-bw*s, x+w-(fx+fw)*bw*s),
                        max(0., y-fy*bh*s), min(self.h-bh*s, y+h-(fy+fh)*bh*s))
            lo, hi = 0., scale
            for _ in range(24):
                mid = (lo+hi)/2
                xl, xr, yt, yb = intervals(mid)
                if xl <= xr and yt <= yb:
                    lo = mid
                else:
                    hi = mid
            scale = max(EPS, lo)
            ww, hh = max(EPS, bw*scale), max(EPS, bh*scale)
            xx = x+(w-fw*ww)/2-fx*ww
            yy = y+(h-fh*hh)/2-fy*hh
            xl, xr, yt, yb = intervals(scale)
            e.update(x=min(max(xl, xx), xr), y=min(max(yt, yy), yb), width=ww, height=hh)
            return es
        if len(es) == 1 and unit.kind == "text" and not next(iter(es.values())).get("rotation"):
            i, e = next(iter(es.items()))
            growth = 1.05 if self.aspect_change < .45 else 2.2
            cap = e["font_size"]*(font_scale if font_scale is not None else self.scale*growth)
            fs = self.type.fit(i, w, h, cap)
            _, th, _ = self.type.measure(i, fs, w)
            e.update(x=x, y=y+max(0., (h-th)/2), width=w, height=min(h, max(th, EPS)), font_size=fs)
            return es
        shapes = [i for i in es if es[i]["type"] == "shape"]
        texts = [i for i in es if es[i]["type"] == "text"]
        flex = (len(shapes) == len(texts) == 1 and len(es) == 2
                and not any(e.get("rotation") for e in es.values())
                and _contains(_bounds(es[shapes[0]]), _bounds(es[texts[0]]), .5))
        if flex:
            si, ti = shapes[0], texts[0]
            shape, text = es[si], es[ti]
            if font_scale is not None:
                hh, ww = min(h, bh*font_scale), min(w, bw*font_scale*1.1)
            else:
                hh = min(h, bh*self.scale*1.8)
                ww = min(w, max(bw*self.scale*1.8, hh*bw/max(bh, EPS)))
            xx, yy = x+(w-ww)/2, y+(h-hh)/2
            ix = max(.06, min(.35, (text["x"]-bx)/max(bw, EPS)))
            iy = max(.08, min(.22, (text["y"]-by)/max(bh, EPS)))
            tw, th = max(EPS, ww*(1-2*ix)), max(EPS, hh*(1-2*iy))
            fs = self.type.fit(ti, tw, th, text["font_size"]*(font_scale if font_scale is not None else self.scale*2.2))
            _, used_h, _ = self.type.measure(ti, fs, tw)
            shape.update(x=xx, y=yy, width=ww, height=hh)
            text.update(x=xx+ww*ix, y=yy+(hh-min(th, used_h))/2, width=tw,
                        height=min(th, max(EPS, used_h)), font_size=fs)
            return es
        scale = min(w/max(bw, EPS), h/max(bh, EPS), self.scale*2.2)
        xx, yy = x+(w-bw*scale)/2, y+(h-bh*scale)/2
        for i, e in es.items():
            e.update(x=xx+(e["x"]-bx)*scale, y=yy+(e["y"]-by)*scale,
                     width=max(EPS, e["width"]*scale), height=max(EPS, e["height"]*scale))
            if e["type"] == "text":
                e["font_size"] = self.type.fit(i, e["width"], e["height"], self.elements[i]["font_size"]*scale)
        return es

    def _flow(self, ids: list[int], rect: Rect, font_scale: float, horizontal=False) -> dict:
        """Pack a related group using measured demand instead of equal cells."""
        if not ids:
            return {}
        x, y, w, h = rect
        gap = min(self.gap, (w if horizontal else h)/max(8, len(ids)*8))
        def demands_at(factor):
            demands = []
            for j in ids:
                unit = self.units[j]
                if horizontal:
                    demand = unit.box[2]*factor
                elif len(unit.indices) == 1 and unit.kind == "text":
                    i = unit.indices[0]
                    fs = max(1, int(self.elements[i]["font_size"]*factor))
                    demand = self.type.measure(i, fs, w)[1]*1.035
                else:
                    demand = self._unit_ink(j, self.elements)[3]*factor
                demands.append(max(EPS, demand))
            return demands
        extent = w if horizontal else h
        available = max(EPS, extent-gap*(len(ids)-1))
        demands = demands_at(font_scale)
        # Fit related typography with ONE shared scale. Shrinking boxes
        # independently collapses title/body hierarchy at wrap thresholds.
        if not horizontal and sum(demands) > available:
            lo, hi = 0., font_scale
            for _ in range(14):
                mid = (lo+hi)/2
                if sum(demands_at(mid)) <= available:
                    lo = mid
                else:
                    hi = mid
            font_scale = max(.001, lo)
            demands = demands_at(font_scale)
        scale = min(1., available/sum(demands))
        offset = (available-sum(demands)*scale)/2
        out = {}
        for j, demand in zip(ids, demands):
            size = demand*scale
            cell = (x+offset, y, size, h) if horizontal else (x, y+offset, w, size)
            out.update(self._place(self.units[j], cell, font_scale))
            offset += size+gap
        return out

    def _compositions(self) -> list[dict]:
        """Campaign proposals: visual anchor, copy, support, action, footer.

        Supports portrait/landscape by changing the outer split direction.
        Multiple proportions and a shared type scale are searched; none are
        keyed to sample IDs, text strings, or target resolution.
        """
        r = self.roles
        if r["hero"] is None or r["head"] is None or self.aspect_change < .45:
            return []
        candidates = []
        wide = self.w/self.h > 1.4
        rail_needed = len(r["badges"])+len(r["corner"]) > 2 and wide
        for hero_share in (.18, .24, .30, .38):
            for rail_share in ((.16, .22, .28) if rail_needed else (0.,)):
                for multiplier in (.85, 1.05, 1.3, 1.6, 1.9):
                    fs = self.scale*multiplier
                    x, y, w, h = self.rect
                    out = self._extras()
                    # Fine print remains a distinct footer with enough height
                    # for its own measured type; secondary assets accompany it.
                    foot = r["footer"]+r["secondary"]
                    fh = 0.
                    if foot:
                        fh = min(h*.18, max(6., h*.09))
                        out.update(self._flow(foot, (x, y+h-fh, w, fh), fs, horizontal=True))
                        h -= fh+self.gap
                    if wide:
                        hw, rw = w*hero_share, w*rail_share
                        visual = (x, y, hw, h)
                        copy = (x+hw+self.gap, y, w-hw-rw-self.gap*(2 if rw else 1), h)
                        rail = (x+w-rw, y, rw, h)
                    else:
                        hh = h*hero_share
                        visual = (x, y, w, hh)
                        copy = (x, y+hh+self.gap, w, h-hh-self.gap)
                        rail = None
                    # Top-left annotations stay near the imagery; top-right
                    # annotations retain a corner position in the support rail.
                    left = [j for j in r["corner"] if self.units[j].box[0] < self.sw/2]
                    right = [j for j in r["corner"] if j not in left]
                    vx, vy, vw, vh = visual
                    header = left+r["brand"]
                    header_h = min(vh*.28, max(vh*.18, min(self.units[j].box[3]*fs for j in header))) if header else 0.
                    if header:
                        if wide and left and r["brand"]:
                            # A corner annotation must not shift the brand
                            # away from the visual anchor's centerline.
                            out.update(self._flow(left, (vx, vy, vw*.38, header_h), fs, horizontal=True))
                            out.update(self._flow(r["brand"], (vx, vy, vw, header_h), fs, horizontal=True))
                        else:
                            out.update(self._flow(header, (vx, vy, vw, header_h), fs, horizontal=wide and len(header)>1))
                    out.update(self._place(self.units[r["hero"]], (vx, vy+header_h+self.gap, vw, max(EPS, vh-header_h-self.gap))))
                    action = [r["action"]] if r["action"] is not None else []
                    if rail_needed:
                        out.update(self._flow(right+r["badges"]+action, rail, fs))
                        out.update(self._flow(r["body"], copy, fs))
                        # Preserve a source badge cohort as a separate row,
                        # with action/ribbon in the side rail and legal below.
                        alternate = dict(out)
                        cx, cy, cw, ch = copy
                        badge_h = min(ch*.24, max(12., max((self.units[j].box[3]*fs for j in r["badges"]), default=0.)))
                        alternate.update(self._flow(r["body"], (cx, cy, cw, ch-badge_h-self.gap), fs))
                        alternate.update(self._flow(r["badges"], (cx, cy+ch-badge_h, cw, badge_h), fs, horizontal=True))
                        alternate.update(self._flow(right+action, rail, fs))
                        if len(alternate) == len(self.elements):
                            candidates.append(self._design(alternate))
                    else:
                        cx, cy, cw, ch = copy
                        support = right+r["badges"]
                        if support:
                            sh = min(ch*.22, max(self.units[j].box[3]*fs for j in support))
                            out.update(self._flow(support, (cx, cy, cw, sh), fs, horizontal=True))
                            cy, ch = cy+sh+self.gap, ch-sh-self.gap
                        out.update(self._flow(r["body"]+action, (cx, cy, cw, ch), fs))
                    if len(out) == len(self.elements):
                        candidates.append(self._design(out))
        return candidates

    def _confident_composition(self) -> bool:
        """Use relationship-constrained proposals only with clear evidence."""
        r = self.roles
        if r["hero"] is None or r["head"] is None or r["action"] is None or not r["brand"] or len(r["body"]) < 2:
            return False
        hero = self._unit_ink(r["hero"], self.elements)
        others = [self._unit_ink(j, self.elements) for j in r["brand"]]
        primary = max(self.elements[i]["font_size"] for i in self.units[r["head"]].indices)
        secondary = max(self.elements[i]["font_size"] for j in r["body"] if j != r["head"] for i in self.units[j].indices)
        return (hero[2]*hero[3] >= 1.5*max(b[2]*b[3] for b in others)
                and primary >= 1.3*secondary)

    def _feasible_composition(self, design: dict) -> bool:
        """Never prefer a role template at the expense of content/geometry."""
        es = design["elements"]
        for i, e in enumerate(es):
            if min(e["width"], e["height"]) <= 0 or not _contains((0., 0., self.w, self.h), _bounds(e), 1e-4):
                return False
            if e["type"] == "text":
                w, h, lines = self.type.measure(i, int(e["font_size"]), e["width"])
                if w > e["width"]+.05 or h > e["height"]+.05 or lines > (e.get("max_lines") or lines):
                    return False
        for i, a in enumerate(es):
            if i not in self.membership:
                continue
            ab = self._visible(i, a)
            for j in range(i+1, len(es)):
                if j not in self.membership or self.membership[i] == self.membership[j]:
                    continue
                bb = self._visible(j, es[j])
                if _intersection(ab, bb) > .02*max(EPS, min(ab[2]*ab[3], bb[2]*bb[3])):
                    return False
        return True

    def _extras(self) -> dict[int, dict]:
        out = {i: dict(self.elements[i], x=0., y=0., width=self.w, height=self.h) for i in self.backgrounds}
        for i in self.decorations:
            b = _bounds(self.elements[i])
            decor_scale = self.scale if self.aspect_change < .45 else min(self.w/self.sw, self.h/self.sh)
            ww, hh = min(b[2]*decor_scale, self.w), min(b[3]*decor_scale, self.h)
            x = min(max(0., (b[0]+b[2]/2)*self.w/self.sw-ww/2), self.w-ww)
            y = min(max(0., (b[1]+b[3]/2)*self.h/self.sh-hh/2), self.h-hh)
            out.update(self._place(Unit((i,), b, "group", 1.), (x, y, ww, hh)))
        return out

    def _design(self, out: dict[int, dict]) -> dict:
        # Strong local relationships follow their image, not the canvas.
        # Fit the whole image/effect cluster if an attachment crosses an edge.
        attached = {}
        for i, effect in self.effects.items():
            if effect.kind == "attached":
                attached.setdefault(effect.anchor, []).append(i)
        for anchor, ids in attached.items():
            for i in ids:
                out[i] = self._attached_effect(i, anchor, out[anchor])
            cluster = [anchor]+ids
            x, y, w, h = _union([_bounds(out[i]) for i in cluster])
            scale = min(1., self.w/max(EPS, w), self.h/max(EPS, h))
            xx, yy = min(max(0., x), self.w-w*scale), min(max(0., y), self.h-h*scale)
            for i in cluster:
                e = out[i]
                out[i] = dict(e, x=xx+(e["x"]-x)*scale, y=yy+(e["y"]-y)*scale,
                              width=e["width"]*scale, height=e["height"]*scale)
        return {"canvas": dict(self.canvas), "elements": [out[i] for i in range(len(self.elements))]}

    def _attached_effect(self, i, anchor, placed):
        source, original = self.elements[i], self.elements[anchor]
        factor = placed["width"]/max(EPS, original["width"])
        return dict(source, x=placed["x"]+(source["x"]-original["x"])*factor,
                    y=placed["y"]+(source["y"]-original["y"])*factor,
                    width=max(EPS, source["width"]*factor), height=max(EPS, source["height"]*factor))

    def _effect_candidates(self, i, design):
        """Bounded canvas hypotheses plus one direct attachment hypothesis."""
        original = self.elements[i]
        current = design["elements"][i]
        yield current
        effect = self.effects[i]
        if effect.kind == "ambiguous":
            attached = self._attached_effect(i, effect.anchor, design["elements"][effect.anchor])
            if _contains((0., 0., self.w, self.h), _bounds(attached), 1e-5):
                yield attached
        bx, by, bw, bh = _bounds(original)
        cx, cy = (bx+bw/2)*self.w/self.sw, (by+bh/2)*self.h/self.sh
        short = min(self.w/self.sw, self.h/self.sh)
        for factor in (short, (short+self.scale)/2, self.scale, self.scale*1.12):
            factor = min(factor, self.w/max(EPS, bw), self.h/max(EPS, bh))
            ww, hh = max(EPS, bw*factor), max(EPS, bh*factor)
            positions = [(cx, cy), (cx-self.w*.06, cy), (cx+self.w*.06, cy),
                         (cx, cy-self.h*.1), (cx, cy+self.h*.1)]
            # Preserve edge contact, while satisfying the no-off-canvas rule.
            if bx < 0 or by < 0 or bx+bw > self.sw or by+bh > self.sh:
                positions.append((ww/2 if bx < 0 else self.w-ww/2 if bx+bw > self.sw else cx,
                                  hh/2 if by < 0 else self.h-hh/2 if by+bh > self.sh else cy))
            for px, py in positions:
                rect = (min(max(0., px-ww/2), self.w-ww), min(max(0., py-hh/2), self.h-hh), ww, hh)
                yield self._place(Unit((i,), (bx, by, bw, bh), "group", 1.), rect)[i]

    def _refine_effects(self, design):
        """Freeze foreground, then search exposure/distribution of effects."""
        if self.aspect_change < .45:
            return design
        candidates = [i for i, effect in self.effects.items() if effect.kind != "attached"]
        candidates.sort(key=lambda i: (-self.elements[i]["width"]*self.elements[i]["height"], i))
        # Bound ambiguity exploration and overall cost on large input files.
        ambiguous = [i for i in candidates if self.effects[i].kind == "ambiguous"][:3]
        ids = [i for i in candidates if self.effects[i].kind == "canvas" or i in ambiguous][:8]
        if not ids:
            return design
        assets = str(ROOT/"brand_kit/assets")
        source_view = Visibility(self.source, self.effects, self.backgrounds, assets, self._visible)
        target_view = Visibility(design, self.effects, self.backgrounds, assets, self._visible)
        reference = {i: source_view.features(i, self.elements[i]) for i in ids}

        def score(i, e):
            loss = visibility_loss(reference[i], target_view.features(i, e))
            src = self.elements[i]
            # Preserve rough center/size without forcing cropped source art
            # to retain its exact pixels on a different aspect ratio.
            dx = (e["x"]+e["width"]/2)/self.w-(src["x"]+src["width"]/2)/self.sw
            dy = (e["y"]+e["height"]/2)/self.h-(src["y"]+src["height"]/2)/self.sh
            loss += .7*(dx*dx+dy*dy)
            sb, eb = _bounds(src), _bounds(e)
            for original_gap, new_gap, extent in (
                    (sb[0], eb[0], eb[2]), (sb[1], eb[1], eb[3]),
                    (self.sw-sb[0]-sb[2], self.w-eb[0]-eb[2], eb[2]),
                    (self.sh-sb[1]-sb[3], self.h-eb[1]-eb[3], eb[3])):
                if original_gap < 0:
                    loss += 2.*(new_gap/max(1., extent))**2
            ratio = (e["width"]*e["height"]/(self.w*self.h))/max(EPS, src["width"]*src["height"]/(self.sw*self.sh))
            loss += .2*max(0., abs(math.log(max(EPS, ratio)))-.35)**2
            effect = self.effects[i]
            if effect.kind == "ambiguous":
                attached = self._attached_effect(i, effect.anchor, design["elements"][effect.anchor])
                delta = sum(abs(e[k]-attached[k])/max(1., attached[k] if k in ("width", "height") else min(self.w, self.h))
                            for k in ("x", "y", "width", "height"))
                # A direct relationship is evidence, not a forced grouping.
                loss += min(1., delta)*effect.attachment*.6
            return loss

        base_score = sum(score(i, design["elements"][i]) for i in ids)
        beam = [(base_score, design)]
        for i in ids:
            proposals = []
            for total, current in beam:
                previous = score(i, current["elements"][i])
                seen = set()
                for e in self._effect_candidates(i, current):
                    key = tuple(round(e[k], 4) for k in ("x", "y", "width", "height"))
                    if key in seen:
                        continue
                    seen.add(key)
                    es = list(current["elements"])
                    es[i] = e
                    proposals.append((total-previous+score(i, e), dict(current, elements=es)))
            beam = sorted(proposals, key=lambda p: p[0])[:4]
        # The unchanged baseline remains eligible. Contrast may veto a denser
        # background; only foreground geometry is frozen, not legibility.
        base_contrast = self._render_penalty(design)
        ranked = [(base_score+8*base_contrast, design)]
        for loss, candidate in beam:
            contrast = self._render_penalty(candidate)
            if contrast <= base_contrast+.01:
                ranked.append((loss+8*contrast, candidate))
        return min(ranked, key=lambda pair: pair[0])[1]

    def _source_layout(self) -> dict:
        out = self._extras()
        for unit in self.units:
            x, y, w, h = unit.box
            ww, hh = min(self.w, w*self.w/self.sw), min(self.h, h*self.h/self.sh)
            xx, yy = min(max(0., x*self.w/self.sw), self.w-ww), min(max(0., y*self.h/self.sh), self.h-hh)
            out.update(self._place(unit, (xx, yy, max(EPS, ww), max(EPS, hh))))
        return self._design(out)

    def _trees(self) -> list:
        ids = list(range(len(self.units)))
        if not ids:
            return []
        orders = [sorted(ids, key=lambda j: (self.units[j].box[axis], self.units[j].box[1-axis], j)) for axis in (1, 0)]
        if self.aspect_change >= .45:
            # X-order in a centered poster is mostly box-width noise, not a
            # reading order. It can place the logo after the action button.
            orders = orders[:1]
        trees = []
        for order in orders:
            for axis in ("x", "y"):
                other = "y" if axis == "x" else "x"
                trees.append(_split(axis, order))
                for n in range(2, min(5, len(order))+1):
                    bands = [order[k*len(order)//n:(k+1)*len(order)//n] for k in range(n)]
                    trees.append(_split(axis, [_split(other, band) for band in bands]))
        by_kind = [[i for i in orders[0] if self.units[i].kind == kind] for kind in ("image", "text", "group")]
        images, texts, groups = by_kind
        for axis in ("x", "y"):
            other = "y" if axis == "x" else "x"
            trees.append(_split(axis, [_split(other, band) for band in by_kind if band]))
            if images and texts and groups:
                for group_axis in ("x", "y"):
                    copy = _split("y", [_split("y", texts), _split(group_axis, groups)])
                    trees.append(_split(axis, [_split(other, images), copy]))
                    # Optional fine-print footer, inferred from location.
                    last = texts[-1]
                    if len(texts) > 1 and self.units[last].box[1] > .8*self.sh:
                        body = _split(axis, [_split(other, images), _split("y", texts[:-1]), _split(group_axis, groups)])
                        trees.append(_split("y", [body, last]))
        return list(dict.fromkeys(trees))

    def _layout(self, tree, factors: tuple[float, ...]) -> dict:
        out = self._extras()

        def mass(node):
            if isinstance(node, int):
                return self.units[node].mass*factors[node]
            return sum(mass(c) for c in node.children)

        def visit(node, rect):
            if isinstance(node, int):
                out.update(self._place(self.units[node], rect))
                return
            x, y, w, h = rect
            # Concave allocation avoids starving small supporting units.
            weights = [mass(c)**.72 for c in node.children]
            extent = w if node.axis == "x" else h
            gap = min(self.gap, extent/(len(weights)*8))
            available, offset = max(EPS, extent-gap*(len(weights)-1)), 0.
            for child, weight in zip(node.children, weights):
                size = available*weight/sum(weights)
                cell = (x+offset, y, size, h) if node.axis == "x" else (x, y+offset, w, size)
                visit(child, cell)
                offset += size+gap
        visit(tree, self.rect)
        return self._design(out)

    def _legacy_score(self, design: dict) -> float:
        es = design["elements"]
        loss, total = 0., 0.
        for i, (src, e) in enumerate(zip(self.elements, es)):
            if not _contains((0., 0., self.w, self.h), _bounds(e), 1e-4):
                loss += 1000.
            if i in self.backgrounds or i in self.decorations:
                continue
            if e["type"] == "text":
                floor = 8. if self.aspect_change < .45 else max(4., 8*math.sqrt(src["font_size"]/max(v.get("font_size", 0) for v in self.elements)))
                fs, desired = e["font_size"], max(floor, src["font_size"]*self.scale)
                weight = .7+math.sqrt(src["font_size"]/self.median_font)
                tw, th, lines = self.type.measure(i, int(fs), e["width"])
                overflow = max(0., tw/max(EPS, e["width"])-1)+max(0., th/max(EPS, e["height"])-1)
                loss += weight*(max(0., math.log(desired/fs))**2*5
                                + max(0., (floor-fs)/floor)**2*4
                                - .25*math.log(max(1., min(fs/desired, 1.35)))
                                + .8*max(0., math.log(fs/(desired*1.4)))**2)
                loss += 100*overflow+20*max(0, lines-(src.get("max_lines") or lines))
                loss += .4*max(0, lines-4)**2
                total += weight
            elif e["type"] == "image":
                relative = e["width"]*e["height"]/max(EPS, src["width"]*src["height"]*self.scale**2)
                image_weight = .7
                if self.aspect_change >= .45 and self.membership[i] in self.roles["secondary"]:
                    image_weight = .15
                loss += image_weight*math.log(max(EPS, relative))**2
                total += image_weight
        for i, a in enumerate(es):
            if i not in self.membership:
                continue
            for j in range(i+1, len(es)):
                b = es[j]
                if j not in self.membership or self.membership[i] == self.membership[j]:
                    continue
                ab = _bounds(a) if self.aspect_change < .45 else self._visible(i, a)
                bb = _bounds(b) if self.aspect_change < .45 else self._visible(j, b)
                overlap = _intersection(ab, bb)
                if overlap:
                    denom = max(EPS, min(ab[2]*ab[3], bb[2]*bb[3]))
                    original = _intersection(_bounds(self.elements[i]), _bounds(self.elements[j]))
                    intentional = original > 0 and a["type"] != "text" and b["type"] != "text"
                    loss += (1. if intentional else 35.)*overlap/denom
        change = self.aspect_change
        for j, k in (self.text_relations if self.aspect_change < .45 else []):
            a = _union([_bounds(es[i]) for i in self.units[j].indices])
            b = _union([_bounds(es[i]) for i in self.units[k].indices])
            # Keep related copy in a coherent column. This is a soft prior;
            # a side-by-side arrangement can still win when space demands it.
            loss += 12*((a[0]+a[2]/2-b[0]-b[2]/2)/self.w)**2
        for unit in self.units:
            now = _union([_bounds(es[i]) for i in unit.indices])
            bx, by, bw, bh = unit.box
            loss += 3./(1+change)**2*((now[0]+now[2]/2)/self.w-(bx+bw/2)/self.sw)**2
            loss += 3./(1+change)**2*((now[1]+now[3]/2)/self.h-(by+bh/2)/self.sh)**2
            # Wide, small print along the original bottom edge carries an
            # anchoring relationship; avoid turning it into a narrow sidebar.
            if (unit.kind == "text" and by > .8*self.sh and bw > .6*self.sw
                    and max(self.elements[i]["font_size"] for i in unit.indices) < .7*self.median_font):
                loss += 10*max(0., bw/self.sw-now[2]/self.w)**2
                loss += 5*max(0., .85-(now[1]+now[3])/self.h)**2
        return loss/max(1., total)

    def _score(self, design: dict) -> float:
        score = self._legacy_score(design)
        if self.aspect_change < .45:
            return score
        es, r = design["elements"], self.roles
        head = r["head"]
        if head is None:
            return score
        hi = max(self.units[head].indices, key=lambda i: self.elements[i].get("font_size", 0))
        src_fs, fs = self.elements[hi]["font_size"], es[hi]["font_size"]
        # Match relative hierarchy, allowing some compression on tiny targets.
        errors = []
        for i, e in enumerate(es):
            if e["type"] == "text" and i != hi:
                original = self.elements[i]["font_size"]/src_fs
                actual = e["font_size"]/fs
                errors.append(math.log(actual/original)**2)
        score += 2.5*sum(errors)/max(1, len(errors))
        # Flexible backdrops may change shape, but a short badge/button
        # should not turn into a long bar with tiny lettering inside it.
        shape_errors = []
        for unit in self.units:
            if unit.kind != "group":
                continue
            for i in unit.indices:
                src, e = self.elements[i], es[i]
                if e["type"] == "shape" and min(src["width"], src["height"]) > 0:
                    ratio = (e["width"]/e["height"])/(src["width"]/src["height"])
                    shape_errors.append(max(0., abs(math.log(ratio))-.2)**2)
        score += .9*sum(shape_errors)/max(1, len(shape_errors))
        title = self._unit_ink(head, es)
        if r["hero"] is not None:
            hero = self._unit_ink(r["hero"], es)
            src_hero = self._unit_ink(r["hero"], self.elements)
            # Preserve the hero's visual height relative to the title, with
            # a tolerance for the required change in composition direction.
            ratio = (hero[3]/fs)/max(EPS, src_hero[3]/src_fs)
            score += 4*max(0., abs(math.log(max(EPS, ratio)))-.25)**2
            dx = max(0., title[0]-hero[0]-hero[2], hero[0]-title[0]-title[2])
            dy = max(0., title[1]-hero[1]-hero[3], hero[1]-title[1]-title[3])
            score += .07*max(0., math.hypot(dx, dy)/max(1., fs)-3)**2
            if r["action"] is not None:
                action = self._unit_ink(r["action"], es)
                src_action = self._unit_ink(r["action"], self.elements)
                original = src_action[2]*src_action[3]/max(EPS, src_hero[2]*src_hero[3])
                actual = action[2]*action[3]/max(EPS, hero[2]*hero[3])
                score += .7*max(0., abs(math.log(max(EPS, actual/original)))-.4)**2
        for j in r["corner"]:
            b = self._unit_ink(j, es)
            score += 3*max(0., (b[1]+b[3]/2)/self.h-.35)**2
            original = self.units[j].box
            anchor_x = (original[0]+original[2]/2)/self.sw
            position_x = (b[0]+b[2]/2)/self.w
            if anchor_x < .3:
                score += 4*max(0., position_x-.3)**2
            elif anchor_x > .7:
                score += 4*max(0., .7-position_x)**2
        if r["hero"] is not None:
            hero = self._unit_ink(r["hero"], es)
            for j in r["brand"]:
                b = self._unit_ink(j, es)
                score += 8*max(0., (b[1]+b[3]/2-hero[1]-hero[3]/2)/self.h)**2
        if r["action"] is not None and r["body"]:
            action = self._unit_ink(r["action"], es)
            last = self._unit_ink(r["body"][-1], es)
            score += 8*max(0., (last[1]+last[3]/2-action[1]-action[3]/2)/self.h)**2
        for j in r["footer"]:
            b = self._unit_ink(j, es)
            score += 4*max(0., .82-(b[1]+b[3]/2)/self.h)**2
        # Related copy may form rows OR columns, but should be close enough
        # to read as one group. No universal shared-center constraint.
        for j, k in self.text_relations:
            if j not in r["body"] or k not in r["body"]:
                continue
            a, b = self._unit_ink(j, es), self._unit_ink(k, es)
            dx = max(0., b[0]-a[0]-a[2], a[0]-b[0]-b[2])
            dy = max(0., b[1]-a[1]-a[3], a[1]-b[1]-b[3])
            score += .1*max(0., math.hypot(dx, dy)/max(1., fs)-2)**2
            # Preserve the internal alignment of a source copy group, even
            # when the entire group moves beside the imagery. This applies
            # to body copy only, not unrelated labels, action or legal text.
            score += 5*((a[0]+a[2]/2-b[0]-b[2]/2)/max(1., a[2], b[2]))**2
        return score

    def _brand_context(self, design):
        """Consider a shared header only when two clear regions leave room.

        No canvas-fraction destination: the interval comes from visible ink.
        Explicit groups, competing products/logos and cramped banners abstain.
        """
        r = self.roles
        if self.aspect_change < .45 or self.w/self.h <= 1.4 or len(r["brand"]) != 1 or r["hero"] is None or r["head"] is None:
            return None
        unit = self.units[r["brand"][0]]
        if len(unit.indices) != 1:
            return None
        i = unit.indices[0]
        src = self.elements[i]
        if src.get("group_id") is not None or src.get("rotation") or any(e.anchor == i for e in self.effects.values()):
            return None
        hero_source = self._unit_ink(r["hero"], self.elements)
        brand_source = self._visible(i, src)
        if (brand_source[0] < .08*self.sw or brand_source[0]+brand_source[2] > .92*self.sw
                or _intersection(brand_source, hero_source) > .02*brand_source[2]*brand_source[3]):
            # A source corner signature or overlapping product label carries
            # a stronger attachment cue than an independent centered header.
            return None
        if hero_source[2]*hero_source[3] < 2*brand_source[2]*brand_source[3]:
            return None
        if brand_source[1]+brand_source[3]/2 >= hero_source[1]+hero_source[3]/2:
            return None
        for j in r["secondary"]:
            box = self._unit_ink(j, self.elements)
            if box[2]*box[3] >= .45*hero_source[2]*hero_source[3]:
                return None
        es = design["elements"]
        hero = self._unit_ink(r["hero"], es)
        copy = _union([self._unit_ink(j, es) for j in r["body"]])
        brand = self._visible(i, es[i])
        if hero[0]+hero[2] < copy[0]:
            left, right = hero[0]+hero[2], copy[0]
        elif copy[0]+copy[2] < hero[0]:
            left, right = copy[0]+copy[2], hero[0]
        else:
            return None
        gap = right-left
        clearance = max(self.gap, brand[2]*.15)
        if gap < brand[2]*2+2*clearance or gap > min(self.w*.45, brand[2]*6):
            return None
        # Keep an existing top header; do not create a new row by moving copy.
        if brand[1]+brand[3] > min(hero[1]+hero[3]/2, copy[1]+copy[3]/2):
            return None
        return dict(index=i, left=left, right=right, gap=gap, brand=brand,
                    hero=hero, copy=copy, clearance=clearance)

    def _brand_candidates(self, design, context):
        """Translate only the logo. Keep its full geometry/style otherwise."""
        yield design
        i = context["index"]
        logo = design["elements"][i]
        box = context["brand"]
        for fraction in (.35, .5, .65):
            cx = context["left"]+context["gap"]*fraction
            e = dict(logo, x=logo["x"]+cx-box[0]-box[2]/2)
            if not _contains((0., 0., self.w, self.h), _bounds(e), 1e-5):
                continue
            ink = self._visible(i, e)
            if ink[0] < context["left"]+context["clearance"] or ink[0]+ink[2] > context["right"]-context["clearance"]:
                continue
            # Check all foreground, including badges/ribbons and secondary
            # images. Transparent file padding is not an occupied region.
            pad = min(self.gap, box[3]*.2)
            expanded = (ink[0]-pad, ink[1]-pad, ink[2]+2*pad, ink[3]+2*pad)
            if any(j != i and j in self.membership and _intersection(expanded, self._visible(j, other)) > EPS
                   for j, other in enumerate(design["elements"])):
                continue
            es = list(design["elements"])
            es[i] = e
            yield dict(design, elements=es)

    def _brand_position_loss(self, design, context):
        box = self._visible(context["index"], design["elements"][context["index"]])
        left_gap = box[0]-context["left"]
        right_gap = context["right"]-box[0]-box[2]
        balance = ((left_gap-right_gap)/context["gap"])**2
        old_center = context["brand"][0]+context["brand"][2]/2
        travel = abs(box[0]+box[2]/2-old_center)/context["gap"]
        # Existing alignment is meaningful: relocation has a fixed cost and
        # must improve the region-balance objective by a clear margin.
        movement = 0. if travel < 1e-5 else .28+.45*travel**2
        return balance+movement

    def _brand_contrast(self, design, context, background):
        """Sample actual opaque logo pixels against the underlying layers."""
        e = design["elements"][context["index"]]
        tile = load_asset_image(str(ROOT/"brand_kit/assets"), e.get("asset_id"),
                                128, max(1., 128*e["height"]/e["width"]), e.get("fit", "contain"))
        def luminance(rgb):
            c = [v/255 for v in rgb]
            c = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in c]
            return .2126*c[0]+.7152*c[1]+.0722*c[2]
        ratios = []
        for y in range(0, tile.height, 2):
            for x in range(0, tile.width, 2):
                rgba = tile.getpixel((x, y))
                if rgba[3] < 180:
                    continue
                px = min(background.width-1, max(0, int((e["x"]+(x+.5)/tile.width*e["width"])/self.w*background.width)))
                py = min(background.height-1, max(0, int((e["y"]+(y+.5)/tile.height*e["height"])/self.h*background.height)))
                under = background.getpixel((px, py))
                alpha = rgba[3]/255*e.get("opacity", 1.)
                a, b = luminance(under), luminance([alpha*v+(1-alpha)*u for v, u in zip(rgba[:3], under)])
                ratios.append((max(a, b)+.05)/(min(a, b)+.05))
        return median(ratios) if ratios else 0.

    def _refine_brand(self, design):
        context = self._brand_context(design)
        if context is None:
            return design
        candidates = list(self._brand_candidates(design, context))
        if len(candidates) == 1:
            return design
        i = context["index"]
        e = design["elements"][i]
        under = [v for j, v in enumerate(design["elements"]) if (v.get("z_index", 0), j) < (e.get("z_index", 0), i)]
        ex, _ = convert_chameleon_to_crello(dict(design, elements=under), self.renderer.font_manager, str(ROOT/"brand_kit/assets"))
        rendered = self.renderer.render(ex, max_size=800, format="png")
        with Image.open(BytesIO(rendered)) as im:
            background = im.convert("RGB")
        baseline = self._brand_position_loss(design, context)
        contrast = self._brand_contrast(design, context, background)
        ranked = [(baseline, design)]
        for candidate in candidates[1:]:
            new_contrast = self._brand_contrast(candidate, context, background)
            # Do not require an already high-contrast logo to keep 90% of
            # an unnecessarily large ratio; retain a readability floor.
            if new_contrast < max(3., min(4.5, contrast*.9)):
                continue
            loss = self._brand_position_loss(candidate, context)
            if loss < baseline-.2:
                ranked.append((loss, candidate))
        return min(ranked, key=lambda pair: pair[0])[1]

    def _render_penalty(self, design: dict) -> float:
        """Render shortlist and sample contrast beneath each text layer.

        A contrast proxy, not an aesthetic/semantic judge. Sampling each
        z-layer handles labels on colored buttons without changing colors.
        """
        assets = str(ROOT/"brand_kit"/"assets")
        example, _ = convert_chameleon_to_crello(design, self.renderer.font_manager, assets)
        size = max(1, min(1200, int(max(self.w, self.h))))
        rendered = self.renderer.render(example, max_size=size, format="png")
        with Image.open(BytesIO(rendered)) as im:
            im.load()
        ordered = sorted(design["elements"], key=lambda e: e.get("z_index", 0))
        penalty, count, backgrounds = 0., 0, {}

        def luminance(rgb):
            c = [v/255 for v in rgb]
            c = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in c]
            return .2126*c[0]+.7152*c[1]+.0722*c[2]

        for position, e in enumerate(ordered):
            if e["type"] != "text" or not e.get("content"):
                continue
            under = tuple(j for j in range(position) if ordered[j]["type"] != "text")
            if under not in backgrounds:
                doc = {"canvas": self.canvas, "elements": [ordered[j] for j in under]}
                ex, _ = convert_chameleon_to_crello(doc, self.renderer.font_manager, assets)
                data = self.renderer.render(ex, max_size=max(1, min(600, int(max(self.w, self.h)))), format="png")
                with Image.open(BytesIO(data)) as im:
                    backgrounds[under] = im.convert("RGB")
            bg = backgrounds[under]
            colors = [e.get("color")]+[r["color"] for r in e.get("style_runs") or [] if "color" in r]
            ratios = []
            for fx, fy in ((.2, .3), (.5, .5), (.8, .7)):
                px = min(bg.width-1, max(0, int((e["x"]+fx*e["width"])/self.w*bg.width)))
                py = min(bg.height-1, max(0, int((e["y"]+fy*e["height"])/self.h*bg.height)))
                pixel = bg.getpixel((px, py))
                base = luminance(pixel)
                for color in colors:
                    r, g, b, alpha = parse_rgba(color)
                    foreground = luminance([alpha*v+(1-alpha)*p for v, p in zip((r, g, b), pixel)])
                    ratios.append((max(base, foreground)+.05)/(min(base, foreground)+.05))
            penalty += max(0., (4.5-median(ratios))/4.5)**2
            count += 1
        return penalty/max(1, count)*1.5

    def run(self) -> dict:
        # An already feasible design at the requested size needs no redesign.
        # This also makes repeated application stable for unchanged canvases.
        if self.source["canvas"] == self.canvas:
            feasible = True
            for i, e in enumerate(self.elements):
                feasible &= min(e["width"], e["height"]) > 0 and _contains((0., 0., self.w, self.h), _bounds(e), 1e-4)
                if e["type"] == "text":
                    fs = e["font_size"]
                    tw, th, lines = self.type.measure(i, int(fs), e["width"])
                    feasible &= fs == int(fs) and tw <= e["width"] and th <= e["height"] and lines <= (e.get("max_lines") or lines)
            if feasible:
                return {"canvas": dict(self.canvas), "elements": deepcopy(self.elements)}
        baseline = self._source_layout()
        pool = [(self._score(baseline), baseline)]
        initial = tuple(1. for _ in self.units)
        seeds = []
        for serial, tree in enumerate(self._trees()):
            design = self._layout(tree, initial)
            seeds.append((self._score(design), serial, tree, initial, design))
        # Bounded beam refinement of several distinct topologies.
        ranked = sorted(seeds, key=lambda s: (s[0], s[1]))
        selected = ranked[:6]
        # Keep a footer topology alive even if its unrefined allocation is
        # initially worse; otherwise beam pruning can discard useful families.
        footers = [s for s in ranked if isinstance(s[2], Split) and s[2].axis == "y"
                   and len(s[2].children) == 2 and isinstance(s[2].children[-1], int)
                   and isinstance(s[2].children[0], Split)]
        selected.extend(s for s in footers[:2] if s[1] not in {v[1] for v in selected})
        for _, serial, tree, factors, design in selected:
            beam = [(self._score(design), factors, design)]
            seen = {factors}
            for step in (1.8, 1.45, 1.25, 1.15):
                proposals = list(beam)
                for _, current, _ in beam:
                    for i in range(len(self.units)):
                        for multiplier in (step, 1/step):
                            changed = list(current)
                            changed[i] = round(changed[i]*multiplier, 5)
                            key = tuple(changed)
                            if key in seen:
                                continue
                            seen.add(key)
                            candidate = self._layout(tree, key)
                            proposals.append((self._score(candidate), key, candidate))
                beam = sorted(proposals, key=lambda s: (s[0], s[1]))[:3]
            pool.extend((s, d) for s, _, d in beam[:2])
        pool.extend((s, d) for s, _, _, _, d in seeds)
        compositions = [(self._score(d), d) for d in self._compositions()]
        feasible = [(s, d) for s, d in compositions if self._feasible_composition(d)]
        if self._confident_composition() and feasible:
            # Preserve reliable source relationships by construction rather
            # than hoping a weighted score rescues arbitrary rearrangements.
            pool = feasible
        else:
            pool.extend(feasible)
        pool.sort(key=lambda pair: pair[0])
        best, best_score = pool[0][1], math.inf
        shortlist, seen_designs = [], set()
        for score, design in pool:
            signature = tuple(tuple(round(e.get(k, 0), 3) for k in ("x", "y", "width", "height", "font_size")) for e in design["elements"])
            if signature not in seen_designs:
                shortlist.append((score, design))
                seen_designs.add(signature)
            if len(shortlist) == 4:
                break
        for score, design in shortlist:
            # Dependency errors stay actionable rather than being swallowed.
            score += self._render_penalty(design)
            if score < best_score:
                best, best_score = design, score
        overflow = []
        for i, e in enumerate(best["elements"]):
            if e["type"] == "text":
                w, h, lines = self.type.measure(i, int(e["font_size"]), e["width"])
                if w > e["width"]+.1 or h > e["height"]+.1 or lines > (e.get("max_lines") or lines):
                    overflow.append(e["id"])
        if overflow:
            logger.warning("Text does not fit at >=1px in searched layouts: %s", ", ".join(overflow))
        return deepcopy(self._refine_brand(self._refine_effects(best)))


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Adapt geometry without mutating input or replacing any content/assets.

    Offline, CPU-only; brand-kit paths are relative to this repository.
    Very small targets may be structurally valid without being readable.
    """
    for canvas in (source["canvas"], target_canvas):
        for key in ("width", "height"):
            value = canvas[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 1 or int(value) != value):
                raise ValueError(f"Canvas {key} must be a positive integer")
    ids = set()
    for e in source["elements"]:
        if e["id"] in ids:
            raise ValueError(f"Duplicate element id: {e['id']}")
        ids.add(e["id"])
        for key in ("x", "y", "width", "height", "rotation"):
            value = e.get(key, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Nonfinite/invalid {key} on element {e['id']}")
        if min(e["width"], e["height"]) < 0:
            raise ValueError(f"Negative dimensions on element {e['id']}")
        if e["type"] == "text" and (not math.isfinite(e["font_size"]) or e["font_size"] < 1):
            raise ValueError(f"Invalid font_size on element {e['id']}")
    if not source["elements"]:
        return {"canvas": deepcopy(target_canvas), "elements": []}
    return _Search(source, target_canvas).run()

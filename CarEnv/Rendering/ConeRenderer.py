import cairo
import numpy as np
from .Rendering import stroke_fill
from ..BatchedCones import BatchedCones


def stencil_cone(ctx: cairo.Context, x, y, r):
    # Square base
    ctx.rectangle(-1.2 * r + x, -1.2 * r + y, 2.4 * r, 2.4 * r)

    # Bottom cone start
    ctx.new_sub_path()
    ctx.arc(x, y, r, 0., np.pi * 2)

    # Cap point
    ctx.new_sub_path()
    ctx.arc(x, y, r * .15, 0., np.pi * 2)


def draw_cone(ctx: cairo.Context, x, y, r, red, green, blue):
    stencil_cone(ctx, x, y, r)
    stroke_fill(ctx, (.0, .0, .0), (red, green, blue))


class ConeRenderer:
    def __init__(self):
        self._n = None
        self._r = None
        self._caches = {}

    def reset(self):
        self._n = None
        self._caches = {}

    def render(self, ctx: cairo.Context, cone_pos, cone_types, radius):
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        if len(cone_types) != self._n or radius != self._r:
            self._caches = {}
            self._r = radius
            self._n = len(cone_types)
            for which in (1, 2):
                mask = cone_types == which
                for cp in cone_pos[mask]:
                    x, y = cp
                    stencil_cone(ctx, x, y, self._r)
                self._caches[which] = ctx.copy_path()
                stroke_fill(ctx, (0., 0., 0.), BatchedCones.get_cone_color(which))
        else:
            for which, path in self._caches.items():
                ctx.append_path(path)
                stroke_fill(ctx, (0., 0., 0.), BatchedCones.get_cone_color(which))

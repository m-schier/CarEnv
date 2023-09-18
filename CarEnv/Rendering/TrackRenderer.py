import cairo
from .Rendering import stroke_fill


def _trace(ctx: cairo.Context, path):
    ctx.move_to(*path[0])

    for el in path[1:]:
        ctx.line_to(*el)

    ctx.close_path()


class TrackRenderer:
    def __init__(self):
        self._track_path = None
        self._center_path = None

    def reset(self):
        self._track_path = None
        self._center_path = None

    def render(self, ctx: cairo.Context, centerline, width):
        from shapely.geometry import LinearRing

        if self._track_path is None:
            poly = LinearRing(centerline).buffer(width / 2 + .3)  # Some extra padding

            _trace(ctx, poly.exterior.coords)
            for interior in poly.interiors:
                _trace(ctx, interior.coords)

            self._track_path = ctx.copy_path()
        else:
            ctx.append_path(self._track_path)

        stroke_fill(ctx, (0., 0., 0.), (.6, .6, .6))

        if self._center_path is None:
            _trace(ctx, centerline)
            self._center_path = ctx.copy_path()
        else:
            ctx.append_path(self._center_path)
        stroke_fill(ctx, (.8, .8, .8), None, line_width=2)

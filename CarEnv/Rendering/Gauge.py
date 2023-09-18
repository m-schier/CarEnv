from cairo import Context
import numpy as np
from .Rendering import stroke_fill


class Gauge:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.radius = 40.

    def draw(self, ctx: Context, val):
        ctx.arc(0.0, 0.0, self.radius, 0.0, 2 * np.pi)
        stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))

        start_angle = 2 * np.pi - 1
        stop_angle = 1

        majors = np.linspace(start_angle, stop_angle, 11)

        for angle in majors:
            v = np.array([np.sin(angle), np.cos(angle)])
            ctx.move_to(*(v * self.radius))
            ctx.line_to(*(v * .8 * self.radius))
            stroke_fill(ctx, (0., 0., 0.), None)

        # Needle
        val = np.clip(val, self.min_val, self.max_val)
        rel_val = (val - self.min_val) / (self.max_val - self.min_val)
        needle_angle = (stop_angle - start_angle) * rel_val + start_angle
        v = np.array([np.sin(needle_angle), np.cos(needle_angle)])
        ctx.move_to(*(v * self.radius * .8))
        ctx.line_to(*(v * self.radius * -.12))
        stroke_fill(ctx, (1., 0., 0.), None)

        # Center point needle
        ctx.arc(0.0, 0.0, self.radius * .06, 0., 2 * np.pi)
        stroke_fill(ctx, None, (1.0, 0.0, 0.0))

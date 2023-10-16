import cairo
import numpy as np
from typing import Optional
import os
from .Colors import BACKGROUND, TIRE, EGO_VEH, BRAKE_LIGHT_LIT, BRAKE_LIGHT_UNLIT, GLASS


class BitmapRenderer:
    def __init__(self, width, height, clear_color=None):
        self.width = width
        self.height = height
        self.surface: Optional[cairo.ImageSurface] = None
        self.context: Optional[cairo.Context] = None
        self.__result = None
        self.clear_color = clear_color or BACKGROUND

    @property
    def result(self):
        return self.__result

    def get_data(self):
        self.__result = np.copy(np.asarray(self.surface.get_data()))
        self.__result = self.__result.reshape((self.height, self.width, 4))[..., :3][..., ::-1]
        return self.__result

    def open(self):
        assert self.surface is None
        self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.width, self.height)
        self.context = cairo.Context(self.surface)

    def clear(self):
        self.context.identity_matrix()
        self.context.new_path()
        self.context.set_source_rgb(*self.clear_color)
        self.context.paint()
        self.context.set_source_rgb(.0, .0, .0)
        return self.context

    def close(self):
        assert self.surface is not None
        self.surface.finish()

    def __enter__(self):
        self.open()
        self.clear()
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.get_data()
        self.close()


def stroke_fill(ctx: cairo.Context, stroke_rgb, fill_rgb, line_width=1.0):
    ctx.save()
    ctx.identity_matrix()

    if fill_rgb is not None:
        ctx.set_source_rgb(*fill_rgb)
        ctx.fill_preserve()

    if stroke_rgb is not None:
        ctx.set_line_width(line_width)
        ctx.set_source_rgb(*stroke_rgb)
        ctx.stroke()
    else:
        ctx.new_path()

    ctx.restore()


def _draw_car_wheels(ctx: cairo.Context, l, y1, y2, delta):
    def draw_rect(cx, cy, rw, rh, rot, rgb):
        ctx.save()
        ctx.translate(cx, cy)
        ctx.rotate(rot)
        ctx.rectangle(-rw / 2, -rh / 2, rw, rh)
        stroke_fill(ctx, (0., 0., 0.), rgb)
        ctx.restore()

    # Rear wheels
    draw_rect(-l / 2, y1, .5, .3, 0, TIRE)
    draw_rect(-l / 2, y2, .5, .3, 0, TIRE)

    # Front wheels
    draw_rect(l / 2, y1, .5, .3, delta, TIRE)
    draw_rect(l / 2, y2, .5, .3, delta, TIRE)


def draw_old_car(ctx: cairo.Context, x, y, theta, l, delta, x1, x2, y1, y2, braking=False, color=None):
    color = color if color is not None else EGO_VEH

    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(theta)

    # Wheels more inwards
    _draw_car_wheels(ctx, l, y1 + .1, y2 - .1, delta)

    hrw = .25

    # Front bumper
    ctx.move_to(x2, y2 * .4)
    ctx.line_to(x2 * .97, y2 * .8)
    ctx.line_to(x2 * .90, y2 * .8)
    ctx.line_to(x2 * .90, y1 * .8)
    ctx.line_to(x2 * .97, y1 * .8)
    ctx.line_to(x2, y1 * .4)
    ctx.close_path()
    stroke_fill(ctx, (0., 0., 0.), (.4, .4, .4))

    # Rear bumper
    ctx.move_to(x1, y2 * .4)
    ctx.line_to(x1 * .97, y2 * .8)
    ctx.line_to(x1 * .90, y2 * .8)
    ctx.line_to(x1 * .90, y1 * .8)
    ctx.line_to(x1 * .97, y1 * .8)
    ctx.line_to(x1, y1 * .4)
    ctx.close_path()
    stroke_fill(ctx, (0., 0., 0.), (.4, .4, .4))

    # Front Radkästen
    ctx.move_to(l / 2 + hrw + .3, 0)
    ctx.line_to(l / 2 + hrw + .2, y2)
    ctx.line_to(l / 2 - hrw - .2, y2)
    ctx.line_to(l / 2 - hrw - .3, y2 - .1)
    ctx.line_to(l / 2 - hrw - .3, y1 + .1)
    ctx.line_to(l / 2 - hrw - .2, y1)
    ctx.line_to(l / 2 + hrw + .2, y1)
    ctx.close_path()
    stroke_fill(ctx, (0., 0., 0.), color)

    # Draw lower body
    ctx.move_to(x2, y2 * .4)
    ctx.line_to(x2 * .2, y2 * .9)
    ctx.line_to(-l / 2 + hrw + .3, y2 * .9)
    ctx.line_to(-l / 2 + hrw + .2, y2)
    ctx.line_to(-l / 2 - hrw - .2, y2)
    ctx.line_to(-l / 2 - hrw - .3, y2 * .9)
    ctx.line_to(x1 * .93, y2 * .8)
    ctx.line_to(x1 * .93, y1 * .8)
    ctx.line_to(-l / 2 - hrw - .3, y1 * .9)
    ctx.line_to(-l / 2 - hrw - .2, y1)
    ctx.line_to(-l / 2 + hrw + .2, y1)
    ctx.line_to(-l / 2 + hrw + .3, y1 * .9)
    ctx.line_to(x2 * .2, y1 * .9)
    ctx.line_to(x2, y1 * .4)
    ctx.close_path()
    stroke_fill(ctx, (0., 0., 0.), color)

    # Super structure
    ctx.move_to(x2 * .2, y2 * .9)
    ctx.line_to(0., y2 * .75)
    ctx.line_to(x1 * .85, y2 * .75)
    ctx.line_to(x1 * .85, y1 * .75)
    ctx.line_to(0., y1 * .75)
    ctx.line_to(x2 * .2, y1 * .9)
    ctx.close_path()
    stroke_fill(ctx, (0., 0., 0.), color)

    # Windscreen
    ctx.move_to(x2 * .15, y2 * .75)
    ctx.line_to(0., y2 * .6)
    ctx.line_to(0., y1 * .6)
    ctx.line_to(x2 * .15, y1 * .75)
    ctx.close_path()
    stroke_fill(ctx, (0., 0., 0.), GLASS)

    # Rear window
    ctx.move_to(x1 * .5, y2 * .6)
    ctx.line_to(x1 * .7, y2 * .6)
    ctx.line_to(x1 * .7, y1 * .6)
    ctx.line_to(x1 * .5, y1 * .6)
    ctx.close_path()
    stroke_fill(ctx, (0., 0., 0.), GLASS)

    # Brake lights
    ctx.rectangle(x1 * .93, y1 * .75, x2 * .08, y2 * .4)
    ctx.rectangle(x1 * .93, y2 * .75, x2 * .08, y1 * .4)
    brake_color = BRAKE_LIGHT_UNLIT if not braking else BRAKE_LIGHT_LIT
    stroke_fill(ctx, (0., 0., 0.), brake_color)

    ctx.restore()


def draw_vehicle_proxy(ctx, env, pose=None, query_env=True, color=None):
    pose = pose if pose is not None else env.vehicle_model.get_pose()
    braking = env.vehicle_model.is_braking if query_env else False
    steering_angle = env.steering_history[-1] if query_env else 0.

    draw_old_car(ctx, *pose, env.vehicle_model.wheelbase, steering_angle, *env.collision_bb, braking=braking, color=color)


def draw_vehicle_state(ctx: cairo.Context, env):
    if env.vehicle_model.v_front_ is None:
        return

    ctx.save()

    ctx.move_to(50, 0)
    ctx.line_to(-50, 0)
    stroke_fill(ctx, (0., 0., 0.), None, 3.)

    # Draw velocities
    ctx.save()
    ctx.translate(50, 0)
    ctx.move_to(0., 0.)
    ctx.line_to(*(env.vehicle_model.v_front_) * 4.)
    stroke_fill(ctx, (1., 0., 0.), None, 3.)
    ctx.restore()
    ctx.save()
    ctx.translate(-50, 0)
    ctx.move_to(0., 0.)
    ctx.line_to(*(env.vehicle_model.v_rear_) * 4.)
    stroke_fill(ctx, (1., 0., 0.), None, 3.)
    ctx.restore()

    # Draw forces
    ctx.save()
    ctx.translate(50, 0)
    ctx.move_to(0., 0.)
    ctx.line_to(*(env.vehicle_model.force_front_ * .01))
    stroke_fill(ctx, (0., 0., 1.), None, 3.)
    if env.vehicle_model.front_slip_:
        ctx.arc(0., 0., env.vehicle_model.peak_traction * .01, 0., np.pi * 2)
        stroke_fill(ctx, (0., 0., 1.), None, 3.)
    ctx.restore()
    ctx.save()
    ctx.translate(-50, 0)
    ctx.move_to(0., 0.)
    ctx.line_to(*(env.vehicle_model.force_rear_ * .01))
    stroke_fill(ctx, (0., 0., 1.), None, 3.)
    if env.vehicle_model.rear_slip_:
        ctx.arc(0., 0., env.vehicle_model.peak_traction * .01, 0., np.pi * 2)
        stroke_fill(ctx, (0., 0., 1.), None, 3.)
    ctx.restore()

    ctx.restore()

import taichi as ti
import time

ti.init(arch=ti.gpu)

pixel = ti.Vector.field(3, dtype=ti.f32, shape=(800, 600))

PI = 3.14159265359
col1 = ti.Vector([0.216, 0.471, 0.698])
col2 = ti.Vector([1.00, 0.329, 0.298])
col3 = ti.Vector([0.867, 0.910, 0.247])
m = ti.Matrix([
    [0.3, 0.9, 0.6],
    [-0.9, 0.36, -0.48],
    [-0.6, -0.48, 0.34]
])


@ti.func
def clamp(v, v_min, v_max):
    return ti.min(ti.max(v, v_min), v_max)


@ti.func
def smoothstep(edge1, edge2, v):
    assert (edge1 != edge2)
    t = (v - edge1) / float(edge2 - edge1)
    t = clamp(t, 0.0, 1.0)

    return (3 - 2 * t) * t ** 2


@ti.func
def disk(r, center, radius):
    return 1.0 - smoothstep(radius - 0.008, radius + 0.008, (r - center).norm())


@ti.func
def fract(vec):
    return vec - ti.floor(vec)


@ti.func
def mix(x, y, a):
    return x * (1 - a) + y * a


@ti.func
def hash(n):
    return fract(ti.sin(n) * 43758.5453123)


@ti.func
def noise(x):
    p = ti.floor(x)
    f = fract(x)
    f = f * f * (3.0 - 2.0 * f)
    n = p.x + p.y * 57.0
    res = mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
              mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y)
    return res


@ti.func
def fbm(p):
    f = 1.600 * noise(p)
    p = m @ p * 2.02
    f += 0.3500 * noise(p)
    p = m @ p * 2.33
    f += 0.2250 * noise(p)
    p = m @ p * 2.03
    f += 0.0825 * noise(p)
    p = m @ p * 2.01
    return f


@ti.func
def map(p, frame):
    d = 0.01 - p.y
    f = fbm(p * 1.0 - ti.Vector([.4, 0.3, -0.3]) * (frame + 46))
    d += 4.0 * f
    d = clamp(d, 0.0, 1.0)
    res = ti.Vector([d, d, d, d])
    res.w = pow(res.y, .1)
    a = mix(.7 * ti.Vector([1.0, 0.4, 0.2]), ti.Vector([0.2, 0.0, 0.2]), res.y * 1.)
    res.x = a.x + pow(abs(.95 - f), 26.0) * 1.85
    res.y = a.y + pow(abs(.95 - f), 26.0) * 1.85
    res.z = a.z + pow(abs(.95 - f), 26.0) * 1.85
    return res


@ti.func
def raymarch(ro, rd, frame):
    s = ti.Vector([0, 0, 0, 0])
    t = 0.0
    pos = ti.Vector([0.0, 0.0, 0.0])
    for i in range(100):
        if s.w > 0.8 or pos.y > 9.0 or pos.y < -2.0:
            continue
        pos = ro + t * rd
        col = map(pos, frame)
        col.w *= 0.08
        col.x *= col.w
        col.y *= col.w
        col.z *= col.w
        s = s + col * (1.0 - s.w)
        t += max(0.1, 0.04 * t)
    s.x /= (0.003 + s.w)
    s.y /= (0.003 + s.w)
    s.z /= (0.003 + s.w)

    return clamp(s, 0.0, 1.0)


@ti.kernel
def mainImage(frame: int):
    sundir = ti.Vector([1.0, 0.4, 0.0])
    for I in ti.grouped(pixel):
        i, j = I
        q = ti.Vector([i / 800, j / 600])
        p = -1.0 + 2.0 * q
        p.x *= 800 / 600
        a = ti.Vector([ti.cos(2.75 - 3.0 * 0.5), .4 - 1.3 * (0.5 - 2.4), ti.sin(2.75 - 2.0 * 0.5)])
        ro = 5.6 * ti.normalized(a)
        ta = ti.Vector([.0, 5.6, 2.4])
        ww = ti.normalized(ta - ro)
        uu = ti.normalized(ti.cross(ti.Vector([0.0, 1.0, 0.0]), ww))
        vv = ti.normalized(ti.cross(ww, uu))
        rd = ti.normalized(p.x * uu + p.y * vv + 1.5 * ww)
        res = raymarch(ro, rd, frame)
        sun = clamp(ti.dot(sundir, rd), 0.0, 2.0)
        col = mix(ti.Vector([.3, 0.0, 0.05]), ti.Vector([0.2, 0.2, 0.3]), ti.sqrt(max(rd.y, 0.001)))
        col += .4 * ti.Vector([.4, .2, 0.67]) * sun
        col = clamp(col, 0.0, 1.0)
        col += 0.43 * ti.Vector([.4, 0.4, 0.2]) * pow(sun, 21.0)
        v = 1.0 / (2. * (1. + rd.z))
        xy = ti.Vector([rd.y * v, rd.x * v])
        rd.z += frame * .002
        s = noise(ti.Vector([rd.x, rd.z]) * 134.0)
        s += noise(ti.Vector([rd.x, rd.z]) * 370.)
        s += noise(ti.Vector([rd.x, rd.z]) * 870.)
        s = pow(s, 19.0) * 0.00000001 * max(rd.y, 0.0)
        if s > 0.0:
            backStars = ti.Vector(
                [(1.0 - ti.sin(xy.x * 20.0 + (frame + 42) * 13.0 * rd.x + xy.y * 30.0)) * .5 * s, s, s])
            col += backStars
            col = mix(col, ti.Vector([res.x, res.y, res.z]), res.w * 1.3)
            dot_r = ti.dot(ti.Vector([.2125, .7154, .0721]), col * 1.03)
            col = mix(ti.Vector([0.5, 0.5, 0.5]),
                      mix(ti.Vector([dot_r, dot_r, dot_r]), col * 1.03, 1.15),
                      1.1)
            pixel[I] = col


window = ti.ui.Window('FEM128', (800, 600))
canvas = window.get_canvas()
i = 0
while window.running:
    i += 1
    mainImage(i)
    canvas.set_image(pixel)
    # canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.02)
    # canvas.circles(ball_circle, color=(0.4, 0.4, 0.4), radius=ball_radius)
    #
    # canvas.triangles(vertexPositions,
    #                  indices=triangleIndices,
    #                  per_vertex_color=vertexColors)
    # canvas.circles(vertexPositions, radius=0.003, color=(1, 0.6, 0.2))

    window.show()

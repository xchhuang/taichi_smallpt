import numpy as np
import taichi as ti
import taichi_glsl as ts
import matplotlib.pyplot as plt
from time import time

ti.init(arch=ti.cpu)    # currently only works on cpu
# real = ti.float64

# config the scene
n_spheres = 9
eps = 1e-4
inf = np.inf

downscale = 1
w = int(1024 / downscale)
h = int(768 / downscale)
samps = 1000
fov = 0.5135

eye = ti.Vector([50, 52, 295.6])  # eye
gaze = ti.Vector([0, -0.042612, -1]).normalized()  # gaze

cx = ti.Vector([w * fov / h, 0.0, 0.0])
cy = cx.cross(gaze).normalized() * fov
c = ti.Vector.field(3, dtype=ti.f32, shape=(w * h))  # color buffer


@ti.data_oriented
class Spheres:
    def __init__(self, n_spheres):
        self.rad = ti.field(dtype=ti.f64, shape=n_spheres)
        self.p = ti.field(dtype=ti.f64, shape=(n_spheres, 3))
        self.e = ti.field(dtype=ti.f32, shape=(n_spheres, 3))
        self.c = ti.field(dtype=ti.f32, shape=(n_spheres, 3))
        self.refl = ti.field(dtype=ti.i32, shape=n_spheres)  # DIFF=0, SPEC=1, REFR=2

    def init(self, rad, p, e, c, refl):
        """
        :param rad: radius
        :param p: center
        :param e: emission
        :param c: color
        :param refl: type of material, DIFF=0, SPEC=1, REFR=2
        :return:
        """
        self.rad.from_numpy(rad)
        self.p.from_numpy(p)
        self.e.from_numpy(e)
        self.c.from_numpy(c)
        self.refl.from_numpy(refl)
    
    @ti.func
    def intersect(self, ray_pos, ray_dir):
        """
        :param ray_pos: ray start position
        :param ray_dir: ray direction(normalized)
        :return: intersected object distance and id
        """
        min_t = inf
        min_index = -1
        sp_index = 0
        for i in range(n_spheres):
            r = self.rad[i]  # radius
            p = ti.Vector([self.p[i, 0], self.p[i, 1], self.p[i, 2]])  # center
            op = p - ray_pos
            dop = ray_dir.dot(op)
            D = dop * dop - op.dot(op) + r * r
            t = inf
            if D > 0:
                sqrD = ti.sqrt(D)
                tmin = dop - sqrD
                if eps < tmin < t:
                    t = tmin
                tmax = dop + sqrD
                if eps < tmax < t:
                    t = tmax

                if t < min_t:
                    min_index = sp_index
                    min_t = t
            sp_index += 1
        return ti.Vector([min_t, min_index])


@ti.func
def uniform_sample_on_hemisphere(u1, u2):
    sin_theta = ti.sqrt(ti.max(0.0, 1.0 - u1 * u1))
    phi = 2.0 * np.pi * u2
    return ti.Vector([ti.cos(phi) * sin_theta, ti.sin(phi) * sin_theta, u1])


@ti.func
def cosine_weighted_sample_on_hemisphere(u1, u2):
    cos_theta = ti.sqrt(1.0 - u1)
    sin_theta = ti.sqrt(u1)
    phi = 2.0 * np.pi * u2
    return ti.Vector([ti.cos(phi) * sin_theta, ti.sin(phi) * sin_theta, cos_theta])


spheres = Spheres(n_spheres)
spheres.init(
    rad=np.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 16.5, 16.5, 600], np.float64),
    p=np.array([
        [1e5 + 1, 40.8, 81.6],
        [-1e5 + 99, 40.8, 81.6],
        [50, 40.8, 1e5],
        [50, 40.8, -1e5 + 170],
        [50, 1e5, 81.6],
        [50, -1e5 + 81.6, 81.6],
        [27, 16.5, 47],
        [73, 16.5, 78],
        [50, 681.6 - .27, 81.6],
    ], np.float64),
    e=np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [12, 12, 12],
    ], np.float32),
    c=np.array([
        [.75, .25, .25],
        [.25, .25, .75],
        [.75, .75, .75],
        [0, 0, 0],
        [.75, .75, .75],
        [.75, .75, .75],
        [0.999, 0.999, 0.999],
        [0.999, 0.999, 0.999],
        [0, 0, 0],
    ], np.float32),
    refl=np.array([
        # 0, 0, 0, 0, 0, 0, 1, 2, 0
        0, 0, 0, 0, 0, 0, 0, 0, 0
    ], np.int32)
)


@ti.kernel
def render():
    for y, x in ti.ndrange((0, h), (0, w)):
        for sy in range(2):
            # i = (h - 1 - y) * w + x
            for sx in range(2):
                Ls = ti.Vector([0.0, 0.0, 0.0])
                for s in range(samps):
                    u1 = 2.0 * ti.random()
                    u2 = 2.0 * ti.random()
                    dx = ti.sqrt(u1) - 1.0 if u1 < 1 else 1.0 - ti.sqrt(2.0 - u1)
                    dy = ti.sqrt(u2) - 1.0 if u2 < 1 else 1.0 - ti.sqrt(2.0 - u2)
                    d = cx * (((sx + 0.5 + dx) / 2.0 + x) / w - 0.5) + cy * (((sy + 0.5 + dy) / 2.0 + y) / h - 0.5) + gaze

                    ray_pos = eye + d * 130
                    ray_dir = d.normalized()

                    L = ti.Vector([0.0, 0.0, 0.0])
                    F = ti.Vector([1.0, 1.0, 1.0])

                    depth = 0
                    while depth < 50:  # 50 is just manually set for debugging, but anyway we have Russian roulette

                        sp = spheres.intersect(ray_pos, ray_dir)
                        # print(sp[1])
                        if sp[1] == -1:
                            break
                        # if sp[1] > -1:
                        p = ray_pos + sp[0] * ray_dir  # hit point
                        # ti.atomic_add(ray_pos, sp[0] * ray_dir)
                        shape_p = ti.Vector([spheres.p[sp[1], 0], spheres.p[sp[1], 1], spheres.p[sp[1], 2]])
                        shape_e = ti.Vector([spheres.e[sp[1], 0], spheres.e[sp[1], 1], spheres.e[sp[1], 2]])
                        shape_c = ti.Vector([spheres.c[sp[1], 0], spheres.c[sp[1], 1], spheres.c[sp[1], 2]])
                        hit_type = spheres.refl[sp[1]]

                        n = (p - shape_p).normalized()
                        L += F * shape_e
                        F *= shape_c

                        if depth > 4:   # Russian roulette
                            continue_probability = shape_c.max()
                            if ti.random() >= continue_probability:
                                break
                            F /= continue_probability

                        if hit_type == 0:  # DIFF
                            ww = n if n.dot(ray_dir) < 0 else -n  # if normal and ray is consistent
                            u = ti.Vector([0.0, 1.0, 0.0]) if ti.abs(ww[0]) > 0.1 else ti.Vector([1.0, 0.0, 0.0])
                            u = u.cross(ww).normalized()
                            v = ww.cross(u)

                            sample_d = cosine_weighted_sample_on_hemisphere(ti.random(), ti.random())
                            dd = (sample_d[0] * u + sample_d[1] * v + sample_d[2] * ww).normalized()
                            ray_dir = dd
                            ray_pos = p  # + eps * ray_dir

                        depth += 1

                    Ls += L * (1.0 / samps)

                c[(h - 1 - y) * w + x] += 0.25 * ti.Vector([ts.clamp(Ls[0]), ts.clamp(Ls[1]), ts.clamp(Ls[2])])


start = time()
render()
end = time()
print('Time: {:.4f}s'.format(end - start))

Ls = c.to_numpy()

from image_io import write_ppm

write_ppm(w, h, Ls, "path.ppm")


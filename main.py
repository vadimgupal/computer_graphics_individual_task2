import math
from PIL import Image

# ==========================
# Векторная математика
# ==========================

class Vec:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec(self.x * other, self.y * other, self.z * other)
        # поэлементное умножение (для цветов)
        return Vec(self.x * other.x, self.y * other.y, self.z * other.z)

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Vec(self.x / k, self.y / k, self.z / k)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.dot(self))

    def norm(self):
        l = self.length()
        if l == 0:
            return Vec()
        return self / l

    def clamp01(self):
        return Vec(
            max(0.0, min(1.0, self.x)),
            max(0.0, min(1.0, self.y)),
            max(0.0, min(1.0, self.z)),
        )

    def to_rgb(self):
        c = self.clamp01()
        return int(c.x * 255 + 0.5), int(c.y * 255 + 0.5), int(c.z * 255 + 0.5)


def reflect(v: Vec, n: Vec) -> Vec:
    # отражение v относительно нормали n (оба векторы)
    return v - n * (2.0 * v.dot(n))


def refract(v: Vec, n: Vec, eta: float):
    # v — направление луча (нормализованное),
    # n — нормаль (нормализованная), eta = n1/n2
    cosi = max(-1.0, min(1.0, v.dot(n)))
    etai = 1.0
    etat = eta
    n_dir = n
    if cosi > 0.0:
        # луч идёт изнутри материала наружу
        etai, etat = etat, etai
        n_dir = n * -1.0
    else:
        cosi = -cosi
    eta_ratio = etai / etat
    k = 1.0 - eta_ratio * eta_ratio * (1.0 - cosi * cosi)
    if k < 0.0:
        return None  # полное внутреннее отражение
    return v * eta_ratio + n_dir * (eta_ratio * cosi - math.sqrt(k))


# ==========================
# Материалы и объекты
# ==========================

class Material:
    def __init__(self, color: Vec,
                 kd=1.0,  # diffuse
                 ks=0.0,  # зеркальность
                 kt=0.0,  # прозрачность
                 ior=1.5  # показатель преломления
                 ):
        self.color = color
        self.kd = kd
        self.ks = ks
        self.kt = kt
        self.ior = ior


class HitInfo:
    __slots__ = ("t", "point", "normal", "material")

    def __init__(self, t, point, normal, material):
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material


class Object3D:
    # базовый класс, дальше будут Sphere / Plane / Box
    def intersect(self, ro: Vec, rd: Vec):
        raise NotImplementedError


# ---------- Сфера ----------

class Sphere(Object3D):
    def __init__(self, center: Vec, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ro: Vec, rd: Vec):
        oc = ro - self.center
        b = oc.dot(rd)
        c = oc.dot(oc) - self.radius * self.radius
        disc = b * b - c
        if disc < 0.0:
            return None
        s = math.sqrt(disc)
        t1 = -b - s
        t2 = -b + s
        t = None
        if t1 > 1e-4:
            t = t1
        elif t2 > 1e-4:
            t = t2
        if t is None:
            return None
        p = ro + rd * t
        n = (p - self.center).norm()
        return HitInfo(t, p, n, self.material)


# ---------- Плоскость (стена куба) ----------

class Plane(Object3D):
    # Ось: 0=x,1=y,2=z. value — координата плоскости, normal — куда направлена внутренняя нормаль
    def __init__(self, axis: int, value: float,
                 normal: Vec, material: Material,
                 min1=-1.0, max1=1.0, min2=-1.0, max2=1.0):
        self.axis = axis
        self.value = value
        self.normal = normal
        self.material = material
        # ограничения по двум другим осям (чтобы была грань куба, а не бесконечная плоскость)
        self.min1 = min1
        self.max1 = max1
        self.min2 = min2
        self.max2 = max2

    def intersect(self, ro: Vec, rd: Vec):
        coord = [ro.x, ro.y, ro.z]
        dirc = [rd.x, rd.y, rd.z]
        denom = dirc[self.axis]
        if abs(denom) < 1e-6:
            return None
        t = (self.value - coord[self.axis]) / denom
        if t <= 1e-4:
            return None
        p = ro + rd * t

        # проверяем попадание в "квадрат" стены
        if self.axis == 0:
            u, v = p.y, p.z
        elif self.axis == 1:
            u, v = p.x, p.z
        else:
            u, v = p.x, p.y

        if not (self.min1 <= u <= self.max1 and self.min2 <= v <= self.max2):
            return None

        return HitInfo(t, p, self.normal, self.material)


# ---------- AABB-коробка (внутренние кубы) ----------

class Box(Object3D):
    def __init__(self, min_corner: Vec, max_corner: Vec, material: Material):
        self.min = min_corner
        self.max = max_corner
        self.material = material

    def intersect(self, ro: Vec, rd: Vec):
        # алгоритм "slab method"
        tmin = -1e30
        tmax = 1e30

        for i, (ro_i, rd_i, mn, mx) in enumerate((
            (ro.x, rd.x, self.min.x, self.max.x),
            (ro.y, rd.y, self.min.y, self.max.y),
            (ro.z, rd.z, self.min.z, self.max.z),
        )):
            if abs(rd_i) < 1e-6:
                if ro_i < mn or ro_i > mx:
                    return None
            else:
                t1 = (mn - ro_i) / rd_i
                t2 = (mx - ro_i) / rd_i
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmax < tmin:
                    return None

        if tmax <= 1e-4:
            return None

        t = tmin if tmin > 1e-4 else tmax
        p = ro + rd * t

        # нормаль определяем по тому, к какой грани ближе точка
        eps = 1e-4
        if abs(p.x - self.min.x) < eps:
            n = Vec(-1, 0, 0)
        elif abs(p.x - self.max.x) < eps:
            n = Vec(1, 0, 0)
        elif abs(p.y - self.min.y) < eps:
            n = Vec(0, -1, 0)
        elif abs(p.y - self.max.y) < eps:
            n = Vec(0, 1, 0)
        elif abs(p.z - self.min.z) < eps:
            n = Vec(0, 0, -1)
        else:
            n = Vec(0, 0, 1)

        return HitInfo(t, p, n, self.material)


# ==========================
# Источники света
# ==========================

class PointLight:
    def __init__(self, position: Vec, color: Vec):
        self.position = position
        self.color = color  # яркость/цвет


# ==========================
# Сцена: материалы и объекты
# ==========================

# флаги для включения/выключения эффектов
# флаги для включения/выключения эффектов
ENABLE_SECOND_LIGHT = False
ENABLE_MIRROR_OBJECTS = False
ENABLE_TRANSPARENT_OBJECTS = False
ENABLE_MIRROR_WALL = False  # пока стена обычная синяя


objects = []
lights = []

# Материалы
white = Material(Vec(0.75, 0.75, 0.75), kd=0.9, ks=0.0, kt=0.0)
red   = Material(Vec(0.75, 0.15, 0.15), kd=0.9, ks=0.0, kt=0.0)
blue  = Material(Vec(0.15, 0.15, 0.75), kd=0.9, ks=0.0, kt=0.0)
yellow = Material(Vec(0.9, 0.8, 0.2), kd=0.9, ks=0.0, kt=0.0)

mirror_mat = Material(Vec(1.0, 1.0, 1.0), kd=0.0, ks=1.0, kt=0.0)
glass_mat  = Material(Vec(0.9, 0.9, 1.0), kd=0.05, ks=0.05, kt=0.9, ior=1.5)

# Стены куба: координаты от -1 до 1
# Левая стена (красная)
objects.append(Plane(axis=0, value=-1.0, normal=Vec(1, 0, 0), material=red))
# Правая стена (синяя или зеркальная)
right_wall_material = mirror_mat if ENABLE_MIRROR_WALL else blue
objects.append(Plane(axis=0, value=1.0, normal=Vec(-1, 0, 0), material=right_wall_material))
# Пол
objects.append(Plane(axis=1, value=-1.0, normal=Vec(0, 1, 0), material=white))
# Потолок
objects.append(Plane(axis=1, value=1.0, normal=Vec(0, -1, 0), material=white))
# Задняя стена
objects.append(Plane(axis=2, value=-1.0, normal=Vec(0, 0, 1), material=white))
# Переднюю не делаем — камера стоит "в проёме"

# Внутренний куб №1 (жёлтый)
cube1 = Box(
    min_corner=Vec(-0.6, -1.0, -0.2),
    max_corner=Vec(-0.1, -0.3, 0.3),
    material=yellow,
)
objects.append(cube1)

# Внутренний куб №2 (белый)
cube2_mat = mirror_mat if ENABLE_MIRROR_OBJECTS else white
cube2 = Box(
    min_corner=Vec(0.1, -1.0, -0.6),
    max_corner=Vec(0.6, 0.2, -0.1),
    material=cube2_mat,
)
objects.append(cube2)

# Сфера (стеклянная)
if ENABLE_TRANSPARENT_OBJECTS:
    objects.append(Sphere(center=Vec(0.0, -0.3, 0.4), radius=0.25, material=glass_mat))
else:
    objects.append(Sphere(center=Vec(0.0, -0.3, 0.4), radius=0.25, material=white))

# Источники света
# Основной точечный источник сверху
lights.append(PointLight(position=Vec(0.0, 0.9, 0.0), color=Vec(5, 5, 5)))

# Дополнительный источник (например, сбоку)
if ENABLE_SECOND_LIGHT:
    lights.append(PointLight(
        position=Vec(-0.7, 0.3, 0.7),
        color=Vec(6.0, 3.0, 3.0)  # гораздо сильнее
    ))

# ==========================
# Рэйтрейсер
# ==========================

MAX_DEPTH = 5


def trace_ray(ro: Vec, rd: Vec, depth: int) -> Vec:
    """Рекурсивный трассировщик лучей"""
    if depth > MAX_DEPTH:
        return Vec(0, 0, 0)

    nearest_hit = None
    nearest_obj = None
    min_t = 1e30

    # ищем ближайшее пересечение
    for obj in objects:
        hit = obj.intersect(ro, rd)
        if hit and hit.t < min_t:
            min_t = hit.t
            nearest_hit = hit
            nearest_obj = obj

    if nearest_hit is None:
        # фон — чёрный
        return Vec(0, 0, 0)

    p = nearest_hit.point
    n = nearest_hit.normal
    m = nearest_hit.material

    # небольшое смещение точки, чтобы не пересекать саму себя
    offset_p = p + n * 1e-4

    # ----- прямое освещение -----
    color = Vec(0, 0, 0)

    for light in lights:
        to_light = light.position - p
        dist_to_light = to_light.length()
        ldir = to_light / dist_to_light

        # тень: пускаем луч к источнику и проверяем, нет ли препятствий
        shadow_ro = offset_p
        shadow_rd = ldir
        shadow_hit = False
        for obj in objects:
            h = obj.intersect(shadow_ro, shadow_rd)
            if h and 1e-4 < h.t < dist_to_light:
                shadow_hit = True
                break
        if shadow_hit:
            continue

        # Ламбертово освещение
        lambert = max(0.0, n.dot(ldir))
        # квадрат убывания интенсивности по расстоянию
        attenuation = 1.0 / (1.0 + dist_to_light * dist_to_light)
        color += m.color * light.color * (m.kd * lambert * attenuation)

    # ----- отражение -----
    refl_color = Vec(0, 0, 0)
    if m.ks > 0.0:
        rdir = reflect(rd, n).norm()
        refl_color = trace_ray(offset_p, rdir, depth + 1)

    # ----- преломление (прозрачность) -----
    refr_color = Vec(0, 0, 0)
    if m.kt > 0.0:
        rdir = refract(rd, n, m.ior)
        if rdir is not None:
            # смещаем в сторону противоположную нормали, чтобы не застрять в материале
            inside_offset = p - n * 1e-4
            refr_color = trace_ray(inside_offset, rdir.norm(), depth + 1)

    # итоговый цвет: сумма трёх вкладов
    result = color + refl_color * m.ks + refr_color * m.kt
    result += m.color * 0.03
    return result


def render(width=400, height=300, fov=60.0, filename="cornell_box.png"):
    aspect = width / height
    angle = math.tan(math.radians(fov * 0.5))

    camera_pos = Vec(0.0, 0.0, 3.0)

    pixels = []

    for j in range(height):
        y_ndc = 1.0 - 2.0 * (j + 0.5) / height
        for i in range(width):
            x_ndc = 2.0 * (i + 0.5) / width - 1.0

            x_cam = x_ndc * angle * aspect
            y_cam = y_ndc * angle
            ray_dir = Vec(x_cam, y_cam, -1.0).norm()

            col = trace_ray(camera_pos, ray_dir, 0)
            pixels.append(col.to_rgb())

    img = Image.new("RGB", (width, height))
    img.putdata(pixels)
    img.save(filename)
    print(f"Saved image to {filename}")


if __name__ == "__main__":
    render()

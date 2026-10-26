"""EarthScene: a small, reusable API for Earth-centered Blender renders.

    from barrel_render import EarthScene, data
    sats = data.load_trajectory("output/constellation.csv")
    s = EarthScene(samples=128, resolution=(1920, 1080))
    s.add_earth()                       # procedural by default; or add_earth(glb=...)
    s.add_atmosphere()
    s.add_sun(jd=sats[0]["jd"][0])
    s.add_moon(jd=sats[0]["jd"][0])
    s.add_satellites(sats)              # loops through and shows each
    s.add_scatter(points, frame="geodetic", values=vals)   # data overlays
    s.add_line(track, frame="geodetic")
    s.camera_follow(sat_id=0)           # or camera_overview() / camera_sensor(...)
    s.render_animation("output/orbit.mp4")   # or render_still("out.png")

Every object method takes an optional `glb=` path (e.g. a NASA-3D-Resources model)
that replaces the procedural default; the Earth defaults to procedural as before.
Positions are ECI metres; ECEF/geodetic overlays are fixed to (and rotate with)
the Earth, while ECI overlays (orbit trails) live in the inertial world frame.
"""

import math
import os

import bpy
import mathutils

from . import data, materials
from barrel_assets import asset_path

SCALE = 1.0e-6                       # metres -> Blender units (Mm); Earth ~6.378 BU
R_EARTH = 6378137.0 * SCALE
R_MOON = 1737400.0 * SCALE

PALETTE = [(0.95, 0.55, 0.15), (0.25, 0.7, 1.0), (0.4, 0.95, 0.5),
           (0.95, 0.35, 0.55), (0.8, 0.8, 0.3), (0.6, 0.5, 1.0)]


# --------------------------------------------------------------------------- #
# small vector helpers (ECI metres)
# --------------------------------------------------------------------------- #
def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0])
def _len(a): return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
def _norm(a):
    n = _len(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 0 else (0.0, 0.0, 1.0)
def _bu(p): return (p[0] * SCALE, p[1] * SCALE, p[2] * SCALE)


def _velocity(pos, t, k):
    n = len(pos)
    if n < 2:
        return (1.0, 0.0, 0.0)
    if k <= 0:
        a, b, dt = pos[1], pos[0], t[1] - t[0]
    elif k >= n - 1:
        a, b, dt = pos[-1], pos[-2], t[-1] - t[-2]
    else:
        a, b, dt = pos[k + 1], pos[k - 1], t[k + 1] - t[k - 1]
    dt = dt or 1.0
    return ((a[0] - b[0]) / dt, (a[1] - b[1]) / dt, (a[2] - b[2]) / dt)


# --------------------------------------------------------------------------- #
class EarthScene:
    def __init__(self, engine="CYCLES", samples=96, resolution=(1600, 900),
                 denoise=True, view_transform="AgX", onboard_exposure=-2.0,
                 flat=False, flat_height_scale=4.0):
        scene = bpy.context.scene
        scene.render.engine = engine
        if engine == "CYCLES":
            scene.cycles.device = "CPU"
            scene.cycles.samples = samples
            scene.cycles.use_denoising = denoise
        scene.render.resolution_x, scene.render.resolution_y = resolution
        scene.view_settings.view_transform = view_transform
        # Onboard (sensor/mounted) views look straight down at near-normal sun
        # incidence -- much brighter than the limb -- so darken them a couple of
        # stops to recover ground detail instead of blowing out to white.
        self.onboard_exposure = onboard_exposure
        # Flat-Earth Easter egg: render the Earth as an azimuthal-equidistant disk
        # and project the satellites' ECEF sub-points onto it (render-only; the
        # dynamics/CSV are untouched). flat_height_scale exaggerates altitude so the
        # satellites visibly circle above the disk.
        self.flat = flat
        self._flat_h = flat_height_scale

        self._clear()
        self._world()

        self.earth = None
        self.atmosphere = None
        self.clouds = None
        self._cloud_tex = None      # the cloud image-texture node (for per-frame swap)
        self._cloud_img = None      # currently loaded cloud image
        self.moon = None
        self.sun_disk = None
        self._sun_dir = (1.0, 0.0, 0.0)   # ECI; set by add_sun, used for day/night
        self.sat_data = {}      # sid -> trajectory dict
        self.sat_inst = {}      # sid -> object
        self._jd = None         # reference epoch list (from satellites)
        self._sat_template = None
        self._scatter_template = None
        self.cameras = {}        # name -> {obj, target, kind, sat_id, back, up, side}
        self.active_camera = None

    # ----- setup ---------------------------------------------------------- #
    @staticmethod
    def _clear():
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.lights):
            for block in list(coll):
                coll.remove(block)

    @staticmethod
    def _world():
        w = bpy.data.worlds.new("Space")
        bpy.context.scene.world = w
        w.use_nodes = True
        nt = w.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        bg = nt.nodes.new("ShaderNodeBackground")
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = 180.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
        ramp.color_ramp.elements[1].position = 0.05
        ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.02, 1.0)
        nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        nt.links.new(bg.outputs[0], out.inputs["Surface"])

    # ----- mesh import ---------------------------------------------------- #
    @staticmethod
    def _import_glb(path):
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.gltf(filepath=path)
        except (RuntimeError, AttributeError):
            bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
            bpy.ops.import_scene.gltf(filepath=path)
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        for o in new:
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        if not meshes:
            raise RuntimeError("no meshes in glb: " + path)
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        obj = meshes[0]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        return obj

    @staticmethod
    def _import_stl(path):
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_mesh.stl(filepath=path)
        except (RuntimeError, AttributeError):
            bpy.ops.preferences.addon_enable(module="io_mesh_stl")
            bpy.ops.import_mesh.stl(filepath=path)
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        for o in new:
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        if not meshes:
            raise RuntimeError("no meshes in stl: " + path)
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        obj = meshes[0]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        return obj

    @classmethod
    def _import_mesh(cls, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".glb", ".gltf"}:
            return cls._import_glb(path)
        if ext == ".stl":
            return cls._import_stl(path)
        raise ValueError(f"unsupported mesh asset extension {ext!r}: {path}")

    @staticmethod
    def _normalize(obj, target_max_dim):
        d = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z) or 1.0
        s = target_max_dim / d
        obj.scale = (s, s, s)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(scale=True)
        obj.location = (0.0, 0.0, 0.0)

    # ----- scene elements ------------------------------------------------- #
    def add_earth(self, texture=None, glb=None, color_map=None, bump_map=None,
                  night_map=None, normal_map=None, specular_map=None, sun_direction=None,
                  flat_sun_ecef=None):
        """Earth surface. Priority: a layered photoreal material from `color_map`
        (+ optional `normal_map`/`bump_map` terrain, `specular_map` ocean mask, and
        `night_map` city lights) -> a textured `glb` -> an equirect `texture` -> the
        procedural fallback. The night-lights day/night split uses `sun_direction`
        (ECI; defaults to the scene's sun). In flat mode, `flat_sun_ecef` (the Sun
        direction in ECEF) enables globe-matched day/night on the disk."""
        if sun_direction is not None:
            self._sun_dir = sun_direction
        if self.flat:
            disk_r = R_EARTH * math.pi          # azimuthal radius to the South Pole
            img = color_map or texture
            bpy.ops.mesh.primitive_circle_add(vertices=256, radius=disk_r, fill_type="NGON")
            obj = bpy.context.active_object
            if img and os.path.exists(img):
                nm = night_map if (night_map and os.path.exists(night_map)) else None
                if flat_sun_ecef is not None:        # globe-matched day/night
                    mat = materials.flat_earth_lit("Earth", img, nm, flat_sun_ecef, disk_r)
                else:                                # self-lit map (no day/night)
                    mat = materials.flat_earth_disk("Earth", img, disk_r)
                obj.data.materials.append(mat)
            else:
                obj.data.materials.append(materials.principled("Earth", (0.08, 0.25, 0.45, 1.0)))
            obj.name = "Earth"
            self.earth = obj
            return obj
        if color_map and os.path.exists(color_map):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=R_EARTH, segments=180, ring_count=90)
            obj = bpy.context.active_object
            bpy.ops.object.shade_smooth()
            obj.data.materials.append(materials.photoreal_earth(
                "Earth", color_map, bump_path=bump_map or None, night_path=night_map or None,
                normal_path=normal_map or None, specular_path=specular_map or None,
                sun_dir=self._sun_dir))
        elif glb:
            obj = self._import_glb(glb)
            self._normalize(obj, 2.0 * R_EARTH)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=R_EARTH, segments=96, ring_count=48)
            obj = bpy.context.active_object
            bpy.ops.object.shade_smooth()
            mat = (materials.image_texture("Earth", texture)
                   if texture and os.path.exists(texture)
                   else materials.procedural_earth("Earth"))
            obj.data.materials.append(mat)
        obj.name = "Earth"
        self.earth = obj
        return obj

    def add_clouds(self, texture, height_km=12.0, color=(1.0, 1.0, 1.0), density=1.0,
                   gamma=1.0, invert=False, segments=180, ring_count=90):
        """Add a semi-transparent cloud layer from an equirectangular cloud image
        (e.g. derived from IR data: bright = cloud). The image luminance drives
        opacity, so cloudy areas are white and clear sky shows the Earth below. The
        layer sits ~`height_km` above the surface and rotates with the Earth (call
        after add_earth). `density`/`gamma` tune coverage/contrast; `invert` if your
        IR image has clouds dark.

        For time-evolving clouds, pass the first frame's image here and then call
        set_cloud_image(path) before rendering each subsequent frame."""
        if not (texture and os.path.exists(texture)):
            return None
        r = R_EARTH + height_km * 1000.0 * SCALE
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=segments, ring_count=ring_count)
        obj = bpy.context.active_object
        obj.name = "Clouds"
        bpy.ops.object.shade_smooth()
        mat = materials.cloud_layer("Clouds", texture, color, density, gamma, invert)
        obj.data.materials.append(mat)
        self._cloud_tex = next((n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"), None)
        self._cloud_img = self._cloud_tex.image if self._cloud_tex else None
        if self.earth is not None:
            self._parent_to_earth(obj)   # share the Earth's GMST rotation (ECEF-fixed)
        self.clouds = obj
        return obj

    def set_cloud_image(self, path):
        """Swap the cloud layer's image for time-evolving clouds (one image per
        frame). Frees the previous image so a long sequence stays memory-bounded.
        No-op (keeps the current clouds) if there's no cloud layer or `path` is
        missing. Call before render_still/render_views for that frame."""
        if self._cloud_tex is None or not (path and os.path.exists(path)):
            return None
        img = bpy.data.images.load(path, check_existing=False)
        img.colorspace_settings.name = "Non-Color"
        prev = self._cloud_img
        self._cloud_tex.image = img
        self._cloud_img = img
        if prev is not None and prev is not img:
            try:
                bpy.data.images.remove(prev)
            except (RuntimeError, ReferenceError):
                pass
        return img

    def add_atmosphere(self, sun_direction=None):
        if sun_direction is not None:
            self._sun_dir = sun_direction
        bpy.ops.mesh.primitive_uv_sphere_add(radius=R_EARTH * 1.025, segments=96, ring_count=48)
        a = bpy.context.active_object
        a.name = "Atmosphere"
        bpy.ops.object.shade_smooth()
        a.data.materials.append(materials.atmosphere_scatter("Atmosphere", self._sun_dir))
        # A camera-visible glow only -- it must not flood the night side with fill
        # light (that washed out the dark hemisphere and the city lights).
        a.visible_diffuse = False
        a.visible_glossy = False
        a.visible_transmission = False
        a.visible_volume_scatter = False
        a.visible_shadow = False
        self.atmosphere = a
        return a

    def add_sun(self, jd=None, direction=None, energy=4.0, texture=None,
                disk_distance=260.0, disk_radius=10.0, disk_strength=12.0):
        if direction is None:
            direction = data.sun_direction_eci(jd) if jd is not None else (1.0, 0.0, 0.0)
        self._sun_dir = direction
        light = bpy.data.lights.new("Sun", "SUN")
        light.energy = energy
        light.angle = math.radians(0.53)
        obj = bpy.data.objects.new("Sun", light)
        bpy.context.collection.objects.link(obj)
        target = bpy.data.objects.new("SunTarget", None)
        bpy.context.collection.objects.link(target)
        target.location = (-direction[0] * 100, -direction[1] * 100, -direction[2] * 100)
        c = obj.constraints.new("TRACK_TO")
        c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
        if texture and os.path.exists(texture):
            # A visible glowing disk in the sun direction (not to scale). Lighting
            # still comes from the lamp; the disk is camera-visible only.
            bpy.ops.mesh.primitive_uv_sphere_add(radius=disk_radius, segments=64, ring_count=32)
            disk = bpy.context.active_object
            disk.name = "SunDisk"
            bpy.ops.object.shade_smooth()
            disk.location = (direction[0] * disk_distance, direction[1] * disk_distance,
                             direction[2] * disk_distance)
            disk.data.materials.append(materials.emission_texture("Sun", texture, disk_strength))
            disk.visible_diffuse = disk.visible_glossy = disk.visible_transmission = False
            disk.visible_volume_scatter = disk.visible_shadow = False
            self.sun_disk = disk
        return obj

    def _flat_subpoint(self, ecef_dir, altitude_bu):
        """(x, y, z) BU at `altitude_bu` above the sub-point of an ECEF direction."""
        x_, y_, z_ = ecef_dir
        n = math.sqrt(x_ * x_ + y_ * y_ + z_ * z_) or 1.0
        sublat = math.asin(max(-1.0, min(1.0, z_ / n)))
        sublon = math.atan2(y_, x_)
        rho = R_EARTH * (math.pi / 2.0 - sublat)
        return (rho * math.cos(sublon), rho * math.sin(sublon), altitude_bu)

    def _flat_max_sat_height(self):
        """Highest satellite above the flat disk (BU), so the Sun/Moon can clear it."""
        r_earth_m = R_EARTH / SCALE
        h = 0.0
        for s in self.sat_data.values():
            for p in s["pos"]:
                alt = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - r_earth_m
                if alt > 0.0:
                    h = max(h, alt * self._flat_h * SCALE)
        return h

    def add_flat_sun(self, sun_ecef, texture=None, altitude_frac=0.30, radius=None,
                     strength=20.0, margin=1.7):
        """Flat-Earth Sun: a glowing disk hovering at a low altitude above the
        sub-solar point (where the day/night shader is brightest), as in flat-Earth
        cosmology. Always kept above the satellites (>= margin x their max height)."""
        disk_r = R_EARTH * math.pi
        z = max(altitude_frac * disk_r, self._flat_max_sat_height() * margin)
        r = radius if radius is not None else disk_r * 0.06
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=48, ring_count=24)
        sun = bpy.context.active_object
        sun.name = "FlatSun"
        bpy.ops.object.shade_smooth()
        sun.location = self._flat_subpoint(sun_ecef, z)
        mat = (materials.emission_texture("FlatSun", texture, strength)
               if texture and os.path.exists(texture)
               else materials.emission("FlatSun", (1.0, 0.85, 0.5), strength))
        sun.data.materials.append(mat)
        self.sun_disk = sun
        return sun

    def add_flat_moon(self, moon_ecef, texture=None, altitude_frac=0.24, radius=None,
                      margin=1.35):
        """Flat-Earth Moon: a small disk circling low over its sub-point, lit by the
        scene Sun. Always kept above the satellites (>= margin x their max height)."""
        disk_r = R_EARTH * math.pi
        z = max(altitude_frac * disk_r, self._flat_max_sat_height() * margin)
        r = radius if radius is not None else disk_r * 0.045
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=48, ring_count=24)
        m = bpy.context.active_object
        m.name = "FlatMoon"
        bpy.ops.object.shade_smooth()
        m.location = self._flat_subpoint(moon_ecef, z)
        mat = (materials.image_texture("FlatMoon", texture, roughness=0.95)
               if texture and os.path.exists(texture)
               else materials.principled("FlatMoon", (0.4, 0.4, 0.4, 1.0), roughness=0.95))
        m.data.materials.append(mat)
        self.moon = m
        return m

    def add_moon(self, jd=None, position_eci=None, glb=None, texture=None, radius_scale=1.0):
        pos = position_eci
        if pos is None and jd is not None:
            pos = data.moon_position_eci(jd)
        if pos is None:
            return None
        if glb:
            obj = self._import_glb(glb)
            self._normalize(obj, 2.0 * R_MOON * radius_scale)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=R_MOON * radius_scale,
                                                 segments=64, ring_count=32)
            obj = bpy.context.active_object
            bpy.ops.object.shade_smooth()
            mat = (materials.image_texture("Moon", texture, roughness=0.95)
                   if texture and os.path.exists(texture)
                   else materials.principled("Moon", (0.4, 0.4, 0.4, 1.0),
                                             metallic=0.0, roughness=0.95))
            obj.data.materials.append(mat)
        obj.name = "Moon"
        obj.location = _bu(pos)
        self.moon = obj
        return obj

    def add_stars(self, texture=None, strength=1.0):
        """Set the world background to an equirectangular star map (inertial sky),
        or a dark sky if no texture is given. The stars are shown to the camera
        only (gated by a Light Path node) so they don't illuminate the scene --
        the night side stays dark and the city lights pop."""
        w = bpy.context.scene.world
        w.use_nodes = True
        nt = w.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        star = nt.nodes.new("ShaderNodeBackground")
        star.inputs["Strength"].default_value = strength
        if texture and os.path.exists(texture):
            env = nt.nodes.new("ShaderNodeTexEnvironment")
            env.image = bpy.data.images.load(texture, check_existing=True)
            nt.links.new(env.outputs["Color"], star.inputs["Color"])
        else:
            star.inputs["Color"].default_value = (0.02, 0.02, 0.03, 1.0)
        dark = nt.nodes.new("ShaderNodeBackground")
        dark.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
        nt.links.new(dark.outputs[0], mix.inputs[1])    # lighting rays -> no starlight
        nt.links.new(star.outputs[0], mix.inputs[2])    # camera rays -> see the stars
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        return w

    def _procedural_sat(self, scale):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        body = bpy.context.active_object
        body.scale = (0.5, 0.5, 0.6)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.25, 0, 0))
        pl = bpy.context.active_object
        pl.scale = (1.5, 0.03, 0.55)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.25, 0, 0))
        pr = bpy.context.active_object
        pr.scale = (1.5, 0.03, 0.55)
        for o in (body, pl, pr):
            bpy.context.view_layer.objects.active = o
            o.select_set(True)
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.select_all(action="DESELECT")
        for o in (body, pl, pr):
            o.select_set(True)
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.join()
        body.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True)
        body.data.materials.append(materials.principled("Sat", (0.75, 0.65, 0.25, 1.0),
                                                        metallic=1.0, roughness=0.35))
        return body

    def add_satellites(self, sats, scale=0.30, glb=None, asset=None, trails=True,
                       trail_width=0.012, point_nadir=False):
        """Add every satellite in `sats` (from data.load_trajectory).

        asset: named spacecraft asset from ``barrel_assets`` (e.g. ``landsat9``,
        ``cygnss``, ``cubesat_mid``) or a direct ``.stl/.glb/.gltf`` path. ``glb`` is
        retained for older callers and is treated as a direct mesh path.

        point_nadir: override each satellite's attitude with a nadir-pointing
        (LVLH) one, so the body frame is meaningful and body-mounted cameras
        image the ground. Use for orbit-only sims whose attitude is identity;
        leave False to honour the attitude in the trajectory.
        """
        mesh = asset_path(asset) if asset else glb
        if mesh:
            template = self._import_mesh(mesh)
            self._normalize(template, scale)
        else:
            template = self._procedural_sat(scale)
        template.name = "SatTemplate"
        template.hide_render = True
        self._sat_template = template

        for idx, (sid, s) in enumerate(sorted(sats.items())):
            if point_nadir:
                s["quat"] = data.nadir_attitude(s["pos"], s["t"])
            self.sat_data[sid] = s
            if self._jd is None:
                self._jd = s["jd"]
            inst = template.copy()           # linked duplicate (shares mesh)
            inst.hide_render = False
            inst.visible_shadow = False       # satellites must not cast shadows on the Earth
            inst.rotation_mode = "QUATERNION"
            bpy.context.collection.objects.link(inst)
            self.sat_inst[sid] = inst
            if trails:
                if self.flat:
                    pts = [data.flat_earth_point(p, jd, self._flat_h)
                           for p, jd in zip(s["pos"], s["jd"])]
                else:
                    pts = s["pos"]
                self.add_line(pts, frame="eci", color=PALETTE[idx % len(PALETTE)],
                              width=trail_width)
        return self.sat_inst

    def _to_bu(self, points, frame):
        """Return (points in Blender units, earth_fixed?)."""
        if frame == "geodetic":
            out = []
            for p in points:
                lat, lon = math.radians(p[0]), math.radians(p[1])
                alt = p[2] if len(p) > 2 else 0.0
                out.append(_bu(data.geodetic_to_ecef(lat, lon, alt)))
            return out, True
        if frame == "ecef":
            return [_bu(p) for p in points], True
        return [_bu(p) for p in points], False   # eci

    def _parent_to_earth(self, obj):
        if self.earth is not None:
            obj.parent = self.earth
            obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)

    def add_line(self, points, frame="geodetic", color=(1.0, 1.0, 1.0), width=0.01,
                 emissive=True, strength=3.0):
        bu, earth_fixed = self._to_bu(points, frame)
        cu = bpy.data.curves.new("line", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(len(bu) - 1)
        for i, p in enumerate(bu):
            sp.points[i].co = (p[0], p[1], p[2], 1.0)
        cu.bevel_depth = width
        cu.bevel_resolution = 1
        obj = bpy.data.objects.new("line", cu)
        bpy.context.collection.objects.link(obj)
        obj.visible_shadow = False
        mat = (materials.emission("line", color, strength) if emissive
               else materials.principled("line", (*color, 1.0), 0.0, 0.4))
        obj.data.materials.append(mat)
        if earth_fixed:
            self._parent_to_earth(obj)
        return obj

    def add_scatter(self, points, frame="geodetic", color=(0.95, 0.2, 0.2), size=0.04,
                    values=None, value_range=None, strength=1.8):
        bu, earth_fixed = self._to_bu(points, frame)
        if self._scatter_template is None:
            bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=1)
            t = bpy.context.active_object
            t.name = "ScatterTemplate"
            t.hide_render = True
            t.data.materials.append(materials.emission("scatter_base", (1, 1, 1), strength))
            self._scatter_template = t
        template = self._scatter_template

        cache = {}

        def mat_for(rgb):
            key = (round(rgb[0], 2), round(rgb[1], 2), round(rgb[2], 2))
            if key not in cache:
                cache[key] = materials.emission(f"scatter_{len(cache)}", rgb, strength)
            return cache[key]

        lo = hi = 0.0
        if values is not None:
            lo, hi = value_range if value_range else (min(values), max(values))
        rng = (hi - lo) or 1.0

        for i, p in enumerate(bu):
            inst = template.copy()
            inst.hide_render = False
            inst.visible_shadow = False       # emissive markers (lightning/estimates) cast no shadow
            inst.scale = (size, size, size)
            inst.location = p
            bpy.context.collection.objects.link(inst)
            rgb = data.colormap((values[i] - lo) / rng) if values is not None else color
            inst.material_slots[0].link = "OBJECT"
            inst.material_slots[0].material = mat_for(rgb)
            if earth_fixed:
                self._parent_to_earth(inst)
        return len(bu)

    def add_lightning(self, flashes, frame="geodetic", color=(1.0, 0.95, 0.4),
                      size=0.03, values=None, value_range=None, strength=4.0):
        """Lightning flashes as glowing markers (bright emissive scatter). `flashes`
        is a list of (lat, lon[, alt]) geodetic-deg points (or ECEF/ECI); optional
        `values` (e.g. flash energy) drive a viridis colour. Thin wrapper over
        add_scatter with lightning-bright defaults."""
        return self.add_scatter(flashes, frame=frame, color=color, size=size,
                                values=values, value_range=value_range, strength=strength)

    def add_point_cloud(self, points, frame="eci", sat_id=None, color=(0.8, 0.85, 1.0),
                        size=0.02, values=None, value_range=None, strength=1.5,
                        point_scale=0.05):
        """Scatter a 3D point cloud -- a stereo reconstruction of another satellite, or
        a template-pose estimate. `frame` is 'eci'/'ecef'/'geodetic' (delegates to
        add_scatter) or 'body': points in satellite `sat_id`'s body frame (metres),
        parented to that satellite so they ride its position + attitude. Body points are
        drawn at `point_scale` BU/metre (the satellite glyphs are not to scale)."""
        if frame != "body":
            return self.add_scatter(points, frame=frame, color=color, size=size,
                                    values=values, value_range=value_range, strength=strength)
        if sat_id is None or sat_id not in self.sat_inst:
            raise KeyError("add_point_cloud(frame='body') needs a valid sat_id (run add_satellites first)")
        host = self.sat_inst[sat_id]
        if self._scatter_template is None:
            bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=1)
            t = bpy.context.active_object
            t.name = "ScatterTemplate"
            t.hide_render = True
            t.data.materials.append(materials.emission("scatter_base", (1, 1, 1), strength))
            self._scatter_template = t
        template = self._scatter_template
        cache = {}

        def mat_for(rgb):
            key = (round(rgb[0], 2), round(rgb[1], 2), round(rgb[2], 2))
            if key not in cache:
                cache[key] = materials.emission(f"pc_{len(cache)}", rgb, strength)
            return cache[key]

        lo = hi = 0.0
        if values is not None:
            lo, hi = value_range if value_range else (min(values), max(values))
        rng = (hi - lo) or 1.0
        for i, p in enumerate(points):
            inst = template.copy()
            inst.hide_render = False
            inst.scale = (size, size, size)
            inst.location = (p[0] * point_scale, p[1] * point_scale, p[2] * point_scale)
            bpy.context.collection.objects.link(inst)
            inst.parent = host
            inst.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            rgb = data.colormap((values[i] - lo) / rng) if values is not None else color
            inst.material_slots[0].link = "OBJECT"
            inst.material_slots[0].material = mat_for(rgb)
        return len(points)

    # ----- cameras (multiple per scene) ----------------------------------- #
    def add_camera(self, name, kind="overview", sat_id=0, fov_deg=20.0,
                   back=1.8, up=2.4, side=0.6, lens=40.0, off_nadir_deg=0.0,
                   make_active=True):
        """Add a named camera. Several can coexist in one scene; render_views()
        renders the scene from each.

          kind="overview" : fixed wide shot framing the whole scene.
          kind="follow"   : orbital chase camera trailing satellite `sat_id`.
          kind="sensor"   : at satellite `sat_id`, field of view `fov_deg`, tilted
                            `off_nadir_deg` forward of nadir (0 = straight down;
                            ~55 deg looks toward the horizon so the curved Earth and
                            limb fill the frame).
        """
        cam = bpy.data.cameras.new(name)
        if kind == "sensor":
            cam.angle = math.radians(fov_deg)
            cam.clip_start = 0.001
        else:
            cam.lens = lens
            cam.clip_start = 0.01
        cam.clip_end = 1.0e7
        obj = bpy.data.objects.new(name, cam)
        bpy.context.collection.objects.link(obj)
        target = bpy.data.objects.new(name + "_target", None)
        bpy.context.collection.objects.link(target)
        c = obj.constraints.new("TRACK_TO")
        c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"

        self.cameras[name] = dict(obj=obj, target=target, kind=kind, sat_id=sat_id,
                                  back=back, up=up, side=side, off_nadir=off_nadir_deg)
        if kind == "overview" and self.flat:
            d = R_EARTH * math.pi                # disk radius. Low, oblique angle
            obj.location = (0.0, -d * 2.7, d * 1.45)  # (~28 deg elevation) so it reads
            target.location = (0.0, 0.0, d * 0.15)    # as a flat plane tilted in 3D
        elif kind == "overview":
            max_r = R_EARTH * 1.2
            if self.sat_data:
                max_r = max(max(_len(p) for p in s["pos"])
                            for s in self.sat_data.values()) * SCALE
            obj.location = (max_r * 2.2, -max_r * 2.6, max_r * 1.5)
            target.location = (0.0, 0.0, 0.0)
        elif kind == "sensor":
            target.location = (0.0, 0.0, 0.0)   # nadir = toward Earth centre
        if make_active or self.active_camera is None:
            self.active_camera = name
        return obj

    # Convenience wrappers (single-camera workflows).
    def camera_overview(self, name="overview", lens=40.0, make_active=True):
        return self.add_camera(name, "overview", lens=lens, make_active=make_active)

    def camera_follow(self, sat_id=0, name=None, back=1.8, up=2.4, side=0.6,
                      lens=30.0, make_active=True):
        return self.add_camera(name or f"follow_{sat_id}", "follow", sat_id=sat_id,
                               back=back, up=up, side=side, lens=lens, make_active=make_active)

    def camera_sensor(self, sat_id=0, fov_deg=20.0, off_nadir_deg=0.0, name=None,
                      make_active=True):
        return self.add_camera(name or f"sensor_{sat_id}", "sensor", sat_id=sat_id,
                               fov_deg=fov_deg, off_nadir_deg=off_nadir_deg,
                               make_active=make_active)

    def add_sensor_cameras(self, fov_deg=20.0, off_nadir_deg=0.0, prefix="sat"):
        """One EO-sensor camera per satellite. Pair with render_views() to render
        what every satellite observes. `off_nadir_deg` tilts the boresight forward
        of nadir (use ~50-60 to take in the curved Earth and limb). Returns names."""
        names = []
        for sid in sorted(self.sat_data):
            self.add_camera(f"{prefix}{sid}", "sensor", sat_id=sid, fov_deg=fov_deg,
                            off_nadir_deg=off_nadir_deg, make_active=False)
            names.append(f"{prefix}{sid}")
        return names

    def add_mounted_camera(self, name, sat_id=0, offset=(0.0, 0.0, 0.0),
                           rotation=(0.0, 0.0, 0.0), fov_deg=None, lens=35.0,
                           make_active=True):
        """A camera rigidly bolted to a satellite body. It is parented to the
        satellite, so it inherits the satellite's position AND attitude -- the
        attitude dynamics move it (unlike camera_sensor, which is always nadir).

          offset   : mount position in body axes, metres.
          rotation : mount orientation in body axes (XYZ Euler radians); the
                     camera looks down its local -Z, so (0,0,0) aims along -Z_body
                     (= nadir when the satellite is nadir-pointing).
          fov_deg  : sensor field of view; if None, use `lens` (mm).

        Mount two of these (or use add_stereo_pair) for stereo vision.
        """
        if sat_id not in self.sat_inst:
            raise KeyError("add_satellites() must run before mounting a camera on sat "
                           + str(sat_id))
        cam = bpy.data.cameras.new(name)
        if fov_deg is not None:
            cam.angle = math.radians(fov_deg)
            cam.clip_start = 0.001
        else:
            cam.lens = lens
            cam.clip_start = 0.01
        cam.clip_end = 1.0e7
        obj = bpy.data.objects.new(name, cam)
        bpy.context.collection.objects.link(obj)
        obj.parent = self.sat_inst[sat_id]                 # ride the body + its attitude
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        obj.location = _bu(offset)                         # body-frame metres -> BU
        obj.rotation_euler = rotation
        self.cameras[name] = dict(obj=obj, target=None, kind="mounted", sat_id=sat_id)
        if make_active or self.active_camera is None:
            self.active_camera = name
        return obj

    def add_stereo_pair(self, sat_id=0, baseline=2.0, axis="x", fov_deg=None,
                        lens=35.0, converge_deg=0.0, pitch_deg=0.0, prefix=None,
                        make_active=False):
        """A stereo rig: two body-mounted cameras `baseline` metres apart along a
        body axis ('x','y','z'), both riding the satellite attitude. `converge_deg`
        toes them inward; `pitch_deg` tilts both forward of nadir (toward the
        along-track horizon, like off-nadir sensors). Returns (left_name,
        right_name). Note that at LEO a rigid baseline gives tiny disparity --
        exaggerate `baseline`/`converge_deg`, or sample two times for along-track
        stereo."""
        prefix = prefix or ("stereo_" + str(sat_id))
        ai = {"x": 0, "y": 1, "z": 2}[axis]
        toe = {"x": 1, "y": 0, "z": 0}[axis]               # toe about a perpendicular axis
        names = []
        for sign, side in ((-1.0, "L"), (1.0, "R")):
            offset = [0.0, 0.0, 0.0]
            offset[ai] = sign * baseline / 2.0
            rotation = [0.0, 0.0, 0.0]
            rotation[1] += -math.radians(pitch_deg)        # pitch forward about body +Y
            rotation[toe] += -sign * math.radians(converge_deg)
            nm = prefix + "_" + side
            self.add_mounted_camera(nm, sat_id=sat_id, offset=tuple(offset),
                                    rotation=tuple(rotation), fov_deg=fov_deg,
                                    lens=lens, make_active=False)
            names.append(nm)
        if make_active:
            self.active_camera = names[0]
        return tuple(names)

    def _place_camera(self, name, k):
        rec = self.cameras[name]
        if rec["kind"] == "follow":
            s = self.sat_data[rec["sat_id"]]
            p = s["pos"][k]
            rhat, vhat = _norm(p), _norm(_velocity(s["pos"], s["t"], k))
            hhat = _norm(_cross(p, _velocity(s["pos"], s["t"], k)))
            pb, b, u, sd = _bu(p), rec["back"], rec["up"], rec["side"]
            rec["obj"].location = (pb[0] - b * vhat[0] + u * rhat[0] + sd * hhat[0],
                                   pb[1] - b * vhat[1] + u * rhat[1] + sd * hhat[1],
                                   pb[2] - b * vhat[2] + u * rhat[2] + sd * hhat[2])
            rec["target"].location = pb
        elif rec["kind"] == "sensor":
            s = self.sat_data[rec["sat_id"]]
            p = s["pos"][k]
            cam = _bu(p)
            rec["obj"].location = cam
            theta = math.radians(rec.get("off_nadir", 0.0))
            if theta <= 0.0:
                rec["target"].location = (0.0, 0.0, 0.0)   # straight-down nadir
            else:
                phat = _norm(p)
                nhat = (-phat[0], -phat[1], -phat[2])      # nadir
                v = _velocity(s["pos"], s["t"], k)
                vd = v[0] * phat[0] + v[1] * phat[1] + v[2] * phat[2]
                vhat = _norm((v[0] - vd * phat[0], v[1] - vd * phat[1], v[2] - vd * phat[2]))
                ct, st = math.cos(theta), math.sin(theta)
                b = (nhat[0] * ct + vhat[0] * st,          # boresight tilted forward
                     nhat[1] * ct + vhat[1] * st,
                     nhat[2] * ct + vhat[2] * st)
                rec["target"].location = (cam[0] + b[0] * R_EARTH,
                                          cam[1] + b[1] * R_EARTH,
                                          cam[2] + b[2] * R_EARTH)

    # ----- state placement ----------------------------------------------- #
    def _place(self, k):
        for sid, inst in self.sat_inst.items():
            s = self.sat_data[sid]
            if self.flat:
                inst.location = _bu(data.flat_earth_point(s["pos"][k], s["jd"][k], self._flat_h))
            else:
                inst.location = _bu(s["pos"][k])
            qx, qy, qz, qw = s["quat"][k]
            inst.rotation_quaternion = (qw, qx, qy, qz)
        # The flat disk is the (fixed) ECEF map; only the spherical Earth spins by GMST.
        if self.earth is not None and self._jd is not None and not self.flat:
            self.earth.rotation_euler = (0.0, 0.0, data.gmst(self._jd[k]))
        for name in self.cameras:
            self._place_camera(name, k)

    def _frame_index(self, frame):
        n = len(self._jd) if self._jd else 1
        return (n - 1) if frame < 0 else min(frame, n - 1)

    def _onboard_hidden(self, name):
        """Hide a camera's own host bus from its onboard view (it shouldn't image
        itself). The atmosphere is kept -- it is transparent looking down and adds
        the limb glow on off-nadir/limb views."""
        rec = self.cameras[name]
        if rec["kind"] not in ("sensor", "mounted"):
            return []
        host = self.sat_inst.get(rec["sat_id"])
        return [host] if host is not None else []

    @staticmethod
    def _set_hidden(objs, hidden):
        for o in objs:
            o.hide_render = hidden

    def _begin_onboard(self, name):
        """For sensor/mounted cameras: hide the host bus + atmosphere and darken
        the exposure for clean, well-exposed ground imagery. Returns the hidden
        list to pass to _end_onboard()."""
        hidden = self._onboard_hidden(name)
        self._set_hidden(hidden, True)
        if self.cameras[name]["kind"] in ("sensor", "mounted"):
            bpy.context.scene.view_settings.exposure = self.onboard_exposure
        return hidden

    def _end_onboard(self, hidden):
        self._set_hidden(hidden, False)
        bpy.context.scene.view_settings.exposure = 0.0

    # ----- render --------------------------------------------------------- #
    def render_still(self, out, frame=-1, camera=None):
        name = camera or self.active_camera
        self._place(self._frame_index(frame))
        bpy.context.scene.camera = self.cameras[name]["obj"]
        hidden = self._begin_onboard(name)
        bpy.context.scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        self._end_onboard(hidden)
        print("BARREL render -> " + out)

    def render_views(self, out_pattern, frame=-1, names=None):
        """Render the scene from every camera (or `names`) -> one image each.
        `out_pattern` may contain '{name}', else the camera name is appended.
        Use with add_sensor_cameras() to capture what each satellite observes."""
        self._place(self._frame_index(frame))
        outs = []
        for name in (names or list(self.cameras)):
            bpy.context.scene.camera = self.cameras[name]["obj"]
            hidden = self._begin_onboard(name)
            if "{name}" in out_pattern:
                path = out_pattern.format(name=name)
            else:
                base, ext = os.path.splitext(out_pattern)
                path = f"{base}_{name}{ext or '.png'}"
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            self._end_onboard(hidden)
            outs.append(path)
            print("BARREL render -> " + path)
        return outs

    def render_stereo(self, observer_id, target_id, out_prefix, frame=-1,
                      baseline=2.0, fov_deg=20.0, isolate=True):
        """Stereo CMOS sensor: render a stereo image pair of satellite `target_id` as
        seen from a rig on satellite `observer_id`. Two cameras offset `baseline` metres
        along the observer body x-axis, both aimed at the target, write
        {out_prefix}_L.png / _R.png. Returns a dict with the image paths, pinhole
        intrinsics (fx, fy, cx, cy), `baseline`, and each camera's 4x4 world matrix
        (Blender units) -- enough to recover a point cloud with a stereo matcher +
        data.disparity_to_depth + data.backproject_depth. `isolate` hides everything but
        the target for a clean silhouette."""
        if observer_id not in self.sat_inst or target_id not in self.sat_inst:
            raise KeyError("render_stereo needs add_satellites() to have created both sats")
        scene = bpy.context.scene
        k = self._frame_index(frame)
        self._place(k)
        obs, tgt = self.sat_inst[observer_id], self.sat_inst[target_id]

        cams = []
        for sign, side in ((-1.0, "L"), (1.0, "R")):
            cam = bpy.data.cameras.new(f"stereo_{observer_id}_{side}")
            cam.angle = math.radians(fov_deg)
            cam.clip_start, cam.clip_end = 0.0001, 1.0e7
            obj = bpy.data.objects.new(cam.name, cam)
            bpy.context.collection.objects.link(obj)
            obj.parent = obs
            obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            obj.location = (sign * baseline / 2.0 * SCALE, 0.0, 0.0)   # body x, metres -> BU
            c = obj.constraints.new("TRACK_TO")
            c.target, c.track_axis, c.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
            cams.append((obj, side))

        hidden = []
        if isolate:
            for o in list(bpy.data.objects):
                if o.type == "MESH" and o is not tgt and not o.hide_render:
                    o.hide_render = True
                    hidden.append(o)

        bpy.context.view_layer.update()
        res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
        out = {"intrinsics": data.camera_intrinsics(res_x, res_y, fov_deg),
               "baseline": baseline, "frame": k}
        for obj, side in cams:
            scene.camera = obj
            path = f"{out_prefix}_{side}.png"
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            out[f"image_{side}"] = path
            out[f"pose_{side}"] = [list(row) for row in obj.matrix_world]
            print("BARREL stereo -> " + path)

        self._set_hidden(hidden, False)
        for obj, _ in cams:
            cam_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.cameras.remove(cam_data)
        return out

    def render_animation(self, out, fps=24, max_frames=0, camera=None):
        scene = bpy.context.scene
        name = camera or self.active_camera
        rec = self.cameras[name]
        scene.camera = rec["obj"]
        hidden = self._begin_onboard(name)
        n = len(self._jd) if self._jd else 1
        if max_frames and 1 < max_frames < n:
            frames = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
        else:
            frames = list(range(n))

        if out.lower().endswith(".mp4"):
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format = "MPEG4"
            scene.render.ffmpeg.codec = "H264"
            scene.render.ffmpeg.constant_rate_factor = "HIGH"
            scene.render.fps = fps
        scene.render.filepath = out
        scene.frame_start = 1
        scene.frame_end = len(frames)

        moving = rec["kind"] in ("follow", "sensor")
        for j, k in enumerate(frames):
            scene.frame_set(j + 1)
            self._place(k)
            for inst in self.sat_inst.values():
                inst.keyframe_insert("location")
                inst.keyframe_insert("rotation_quaternion")
            if self.earth is not None:
                self.earth.keyframe_insert("rotation_euler")
            if moving:
                rec["obj"].keyframe_insert("location")
                if rec["kind"] == "follow":
                    rec["target"].keyframe_insert("location")
        bpy.ops.render.render(animation=True)
        self._end_onboard(hidden)
        print("BARREL render -> " + out)

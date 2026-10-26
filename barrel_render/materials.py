"""Material helpers for the BARREL renderer (Blender / bpy)."""

import bpy


def cloud_layer(name, image_path, color=(1.0, 1.0, 1.0), density=1.0, gamma=1.0,
                invert=False):
    """Semi-transparent cloud material from an equirectangular coverage image
    (e.g. derived from IR data; bright = cloud). The image luminance drives the
    Principled BSDF Alpha, so cloudy areas are opaque white and clear sky is
    transparent (the Earth shows through). `density` scales opacity, `gamma`
    adjusts contrast, `invert` if your image has clouds dark."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.1   # clouds aren't glossy
    mat.blend_method = "BLEND"                               # EEVEE; Cycles uses Alpha directly

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path, check_existing=True)
    tex.image.colorspace_settings.name = "Non-Color"        # coverage is data, not colour
    out = tex.outputs["Color"]
    if invert:
        inv = nt.nodes.new("ShaderNodeInvert")
        nt.links.new(out, inv.inputs["Color"])
        out = inv.outputs["Color"]
    if gamma != 1.0:
        g = nt.nodes.new("ShaderNodeGamma")
        g.inputs["Gamma"].default_value = gamma
        nt.links.new(out, g.inputs["Color"])
        out = g.outputs["Color"]
    mul = nt.nodes.new("ShaderNodeMath")                    # density -> alpha (clamped)
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = density
    mul.use_clamp = True
    nt.links.new(out, mul.inputs[0])
    nt.links.new(mul.outputs[0], bsdf.inputs["Alpha"])
    return mat


def flat_earth_disk(name, image_path, disk_radius, strength=1.2):
    """Self-lit material for the flat-Earth disk Easter egg: remaps an
    equirectangular map onto an azimuthal-equidistant disk (centre = North Pole,
    edge = South Pole) using the disk's local XY coordinates."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Strength"].default_value = strength

    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Object"], sep.inputs["Vector"])
    # rho = length(x, y) -> V = 1 - rho/disk_radius  (N pole centre -> top of map)
    vxy = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], vxy.inputs["X"])
    nt.links.new(sep.outputs["Y"], vxy.inputs["Y"])
    length = nt.nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    nt.links.new(vxy.outputs["Vector"], length.inputs[0])
    vmap = nt.nodes.new("ShaderNodeMapRange")
    vmap.inputs["From Min"].default_value = 0.0
    vmap.inputs["From Max"].default_value = disk_radius
    vmap.inputs["To Min"].default_value = 1.0
    vmap.inputs["To Max"].default_value = 0.0
    vmap.clamp = True
    nt.links.new(length.outputs["Value"], vmap.inputs["Value"])
    # theta = atan2(y, x) -> U in [0,1]
    atan = nt.nodes.new("ShaderNodeMath")
    atan.operation = "ARCTAN2"
    nt.links.new(sep.outputs["Y"], atan.inputs[0])
    nt.links.new(sep.outputs["X"], atan.inputs[1])
    umap = nt.nodes.new("ShaderNodeMapRange")
    umap.inputs["From Min"].default_value = -3.14159265
    umap.inputs["From Max"].default_value = 3.14159265
    umap.inputs["To Min"].default_value = 0.0
    umap.inputs["To Max"].default_value = 1.0
    umap.clamp = True
    nt.links.new(atan.outputs["Value"], umap.inputs["Value"])
    uv = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(umap.outputs["Result"], uv.inputs["X"])
    nt.links.new(vmap.outputs["Result"], uv.inputs["Y"])

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path, check_existing=True)
    nt.links.new(uv.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    nt.links.new(emi.outputs[0], out.inputs["Surface"])
    return mat


def flat_earth_lit(name, color_path, night_path, sun_ecef, disk_radius,
                   day_strength=1.9, night_strength=6.0):
    """Flat-Earth disk with globe-matched day/night. Each disk point's geographic
    surface normal is recovered from the azimuthal mapping and dotted with the Sun
    direction in ECEF: the day side shows the lit colour map (Lambertian falloff to
    the terminator) and the night side shows city lights -- the same look as the
    spherical photoreal Earth, projected onto the disk."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Object"], sep.inputs["Vector"])
    vxy = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], vxy.inputs["X"])
    nt.links.new(sep.outputs["Y"], vxy.inputs["Y"])
    rho = nt.nodes.new("ShaderNodeVectorMath"); rho.operation = "LENGTH"
    nt.links.new(vxy.outputs["Vector"], rho.inputs[0])
    rho_safe = nt.nodes.new("ShaderNodeMath"); rho_safe.operation = "MAXIMUM"
    rho_safe.inputs[1].default_value = 1e-6
    nt.links.new(rho.outputs["Value"], rho_safe.inputs[0])

    # equirect UV: U from atan2(y,x), V from radius (N pole centre -> top)
    atan = nt.nodes.new("ShaderNodeMath"); atan.operation = "ARCTAN2"
    nt.links.new(sep.outputs["Y"], atan.inputs[0])
    nt.links.new(sep.outputs["X"], atan.inputs[1])
    umap = nt.nodes.new("ShaderNodeMapRange")
    umap.inputs["From Min"].default_value = -3.14159265
    umap.inputs["From Max"].default_value = 3.14159265
    nt.links.new(atan.outputs["Value"], umap.inputs["Value"])
    vmap = nt.nodes.new("ShaderNodeMapRange")
    vmap.inputs["From Min"].default_value = 0.0
    vmap.inputs["From Max"].default_value = disk_radius
    vmap.inputs["To Min"].default_value = 1.0
    vmap.inputs["To Max"].default_value = 0.0
    vmap.clamp = True
    nt.links.new(rho.outputs["Value"], vmap.inputs["Value"])
    uv = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(umap.outputs["Result"], uv.inputs["X"])
    nt.links.new(vmap.outputs["Result"], uv.inputs["Y"])
    color = nt.nodes.new("ShaderNodeTexImage")
    color.image = bpy.data.images.load(color_path, check_existing=True)
    nt.links.new(uv.outputs["Vector"], color.inputs["Vector"])

    # geographic ECEF normal: lat from radius; (cos lon, sin lon) = (x, y)/rho
    lat = nt.nodes.new("ShaderNodeMapRange")
    lat.inputs["From Min"].default_value = 0.0
    lat.inputs["From Max"].default_value = disk_radius
    lat.inputs["To Min"].default_value = 1.57079633
    lat.inputs["To Max"].default_value = -1.57079633
    lat.clamp = True
    nt.links.new(rho.outputs["Value"], lat.inputs["Value"])
    cl = nt.nodes.new("ShaderNodeMath"); cl.operation = "COSINE"
    nt.links.new(lat.outputs["Result"], cl.inputs[0])
    sl = nt.nodes.new("ShaderNodeMath"); sl.operation = "SINE"
    nt.links.new(lat.outputs["Result"], sl.inputs[0])
    ux = nt.nodes.new("ShaderNodeMath"); ux.operation = "DIVIDE"
    nt.links.new(sep.outputs["X"], ux.inputs[0]); nt.links.new(rho_safe.outputs[0], ux.inputs[1])
    uy = nt.nodes.new("ShaderNodeMath"); uy.operation = "DIVIDE"
    nt.links.new(sep.outputs["Y"], uy.inputs[0]); nt.links.new(rho_safe.outputs[0], uy.inputs[1])
    nx = nt.nodes.new("ShaderNodeMath"); nx.operation = "MULTIPLY"
    nt.links.new(cl.outputs[0], nx.inputs[0]); nt.links.new(ux.outputs[0], nx.inputs[1])
    ny = nt.nodes.new("ShaderNodeMath"); ny.operation = "MULTIPLY"
    nt.links.new(cl.outputs[0], ny.inputs[0]); nt.links.new(uy.outputs[0], ny.inputs[1])
    nrm = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(nx.outputs[0], nrm.inputs["X"])
    nt.links.new(ny.outputs[0], nrm.inputs["Y"])
    nt.links.new(sl.outputs[0], nrm.inputs["Z"])
    sunv = nt.nodes.new("ShaderNodeCombineXYZ")
    sunv.inputs[0].default_value = sun_ecef[0]
    sunv.inputs[1].default_value = sun_ecef[1]
    sunv.inputs[2].default_value = sun_ecef[2]
    dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    nt.links.new(nrm.outputs["Vector"], dot.inputs[0])
    nt.links.new(sunv.outputs["Vector"], dot.inputs[1])

    # day = colour * max(dot,0) * day_strength ; night lights where dot < 0
    dayfac = nt.nodes.new("ShaderNodeMath"); dayfac.operation = "MAXIMUM"
    dayfac.inputs[1].default_value = 0.0
    nt.links.new(dot.outputs["Value"], dayfac.inputs[0])
    dstr = nt.nodes.new("ShaderNodeMath"); dstr.operation = "MULTIPLY"
    dstr.inputs[1].default_value = day_strength
    nt.links.new(dayfac.outputs[0], dstr.inputs[0])
    emi_day = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(color.outputs["Color"], emi_day.inputs["Color"])
    nt.links.new(dstr.outputs[0], emi_day.inputs["Strength"])

    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(emi_day.outputs[0], add.inputs[0])
    if night_path:
        night = nt.nodes.new("ShaderNodeTexImage")
        night.image = bpy.data.images.load(night_path, check_existing=True)
        nt.links.new(uv.outputs["Vector"], night.inputs["Vector"])
        nfac = nt.nodes.new("ShaderNodeMapRange")
        nfac.inputs["From Min"].default_value = 0.25
        nfac.inputs["From Max"].default_value = -0.30
        nfac.clamp = True
        nt.links.new(dot.outputs["Value"], nfac.inputs["Value"])
        nstr = nt.nodes.new("ShaderNodeMath"); nstr.operation = "MULTIPLY"
        nstr.inputs[1].default_value = night_strength
        nt.links.new(nfac.outputs["Result"], nstr.inputs[0])
        emi_night = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(night.outputs["Color"], emi_night.inputs["Color"])
        nt.links.new(nstr.outputs[0], emi_night.inputs["Strength"])
        nt.links.new(emi_night.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    return mat


def principled(name, base_rgba, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = base_rgba
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = roughness
    return mat


def emission(name, rgb, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Color"].default_value = (*rgb, 1.0)
    emi.inputs["Strength"].default_value = strength
    nt.links.new(emi.outputs[0], out.inputs["Surface"])
    return mat


def procedural_earth(name):
    """Blue-marble-ish ocean/land material (used until a texture/glb is supplied)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Metallic"].default_value = 0.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.3
    noise.inputs["Detail"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.40
    cr.elements[0].color = (0.02, 0.10, 0.32, 1.0)
    cr.elements[1].position = 0.52
    cr.elements[1].color = (0.12, 0.30, 0.55, 1.0)
    cr.elements.new(0.54).color = (0.10, 0.34, 0.12, 1.0)
    cr.elements.new(0.72).color = (0.36, 0.30, 0.16, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def image_texture(name, image_path, roughness=0.55):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def emission_texture(name, image_path, strength=1.0):
    """Emissive material textured by an image (for a glowing sun disk, etc.)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Strength"].default_value = strength
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path, check_existing=True)
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    nt.links.new(emi.outputs[0], out.inputs["Surface"])
    return mat


def _image(nt, path, non_color=False, interp="Cubic"):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.interpolation = interp
    return node


def _sun_dot(nt, sun_dir):
    """A node whose Value output = dot(world surface normal, sun direction):
    >0 on the sunlit side, <0 on the night side, ~0 at the terminator."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sunv = nt.nodes.new("ShaderNodeCombineXYZ")
    sunv.inputs[0].default_value = sun_dir[0]
    sunv.inputs[1].default_value = sun_dir[1]
    sunv.inputs[2].default_value = sun_dir[2]
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    nt.links.new(sunv.outputs["Vector"], dot.inputs[1])
    return dot


def photoreal_earth(name, color_path, bump_path=None, night_path=None,
                    normal_path=None, specular_path=None,
                    sun_dir=(1.0, 0.0, 0.0), bump_strength=0.12, normal_strength=1.0,
                    night_strength=6.0, land_roughness=0.65, water_roughness=0.80):
    """Layered, texture-driven Earth: day colour map; terrain from a normal map
    (preferred) or a bump map; ocean roughness from a specular/water mask
    (preferred) or derived from the colour map; and night city lights that glow
    only on the dark side (sun-angle blended).

    Lower `water_roughness` = sharper/brighter sun-glint; raise it toward
    `land_roughness` for matte oceans (no shiny-ball reflection)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Metallic"].default_value = 0.0

    color = _image(nt, color_path)                      # sRGB day map
    nt.links.new(color.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness / ocean glint: land matte, water per `water_roughness`.
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = land_roughness    # mask 0 -> land
    rough.inputs["To Max"].default_value = water_roughness    # mask 1 -> water
    rough.clamp = True
    if specular_path:                                   # water mask (bright = water)
        spec = _image(nt, specular_path, non_color=True)
        rough.inputs["From Min"].default_value = 0.0
        rough.inputs["From Max"].default_value = 1.0
        nt.links.new(spec.outputs["Color"], rough.inputs["Value"])
    else:                                               # derive water = blue-dominant
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(color.outputs["Color"], sep.inputs["Color"])
        maxrg = nt.nodes.new("ShaderNodeMath")
        maxrg.operation = "MAXIMUM"
        nt.links.new(sep.outputs["Red"], maxrg.inputs[0])
        nt.links.new(sep.outputs["Green"], maxrg.inputs[1])
        diff = nt.nodes.new("ShaderNodeMath")
        diff.operation = "SUBTRACT"
        nt.links.new(sep.outputs["Blue"], diff.inputs[0])
        nt.links.new(maxrg.outputs[0], diff.inputs[1])
        rough.inputs["From Min"].default_value = 0.0
        rough.inputs["From Max"].default_value = 0.06
        nt.links.new(diff.outputs[0], rough.inputs["Value"])
    nt.links.new(rough.outputs[0], bsdf.inputs["Roughness"])

    # Terrain relief: a tangent-space normal map if given, else a bump map.
    if normal_path:
        ntex = _image(nt, normal_path, non_color=True)
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = normal_strength
        nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    elif bump_path:
        bump_tex = _image(nt, bump_path, non_color=True)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_strength
        nt.links.new(bump_tex.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    if night_path:
        night = _image(nt, night_path)
        nightfac = nt.nodes.new("ShaderNodeMapRange")   # dark side -> 1, lit side -> 0
        nightfac.inputs["From Min"].default_value = 0.25    # wide, soft terminator:
        nightfac.inputs["From Max"].default_value = -0.30   # city lights fade in gradually
        nightfac.inputs["To Min"].default_value = 0.0
        nightfac.inputs["To Max"].default_value = 1.0
        nightfac.clamp = True
        nt.links.new(_sun_dot(nt, sun_dir).outputs["Value"], nightfac.inputs["Value"])
        estr = nt.nodes.new("ShaderNodeMath")
        estr.operation = "MULTIPLY"
        estr.inputs[1].default_value = night_strength
        nt.links.new(nightfac.outputs[0], estr.inputs[0])
        nt.links.new(night.outputs["Color"], bsdf.inputs["Emission Color"])
        nt.links.new(estr.outputs[0], bsdf.inputs["Emission Strength"])
    return mat


def atmosphere_scatter(name, sun_dir=(1.0, 0.0, 0.0), strength=2.6):
    """Fresnel-limb atmosphere tinted by the sun angle: blue on the day limb,
    orange at the terminator, dim on the night limb -> a scattering-like glow."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    emi = nt.nodes.new("ShaderNodeEmission")
    fres = nt.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.2

    dot = _sun_dot(nt, sun_dir)
    tint = nt.nodes.new("ShaderNodeMapRange")
    tint.inputs["From Min"].default_value = -0.45     # wide, soft colour transition
    tint.inputs["From Max"].default_value = 0.65
    tint.clamp = True
    nt.links.new(dot.outputs["Value"], tint.inputs["Value"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.02, 0.04, 0.10, 1.0)      # night limb
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.20, 0.45, 1.0, 1.0)       # day limb (blue)
    cr.elements.new(0.4).color = (1.0, 0.35, 0.12, 1.0)  # terminator (orange)
    nt.links.new(tint.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], emi.inputs["Color"])

    lit = nt.nodes.new("ShaderNodeMapRange")            # dim the night-side limb (soft)
    lit.inputs["From Min"].default_value = -0.4
    lit.inputs["From Max"].default_value = 0.4
    lit.inputs["To Min"].default_value = 0.08
    lit.inputs["To Max"].default_value = 1.0
    lit.clamp = True
    nt.links.new(dot.outputs["Value"], lit.inputs["Value"])
    s1 = nt.nodes.new("ShaderNodeMath")
    s1.operation = "MULTIPLY"
    nt.links.new(fres.outputs[0], s1.inputs[0])
    nt.links.new(lit.outputs[0], s1.inputs[1])
    s2 = nt.nodes.new("ShaderNodeMath")
    s2.operation = "MULTIPLY"
    s2.inputs[1].default_value = strength
    nt.links.new(s1.outputs[0], s2.inputs[0])
    nt.links.new(s2.outputs[0], emi.inputs["Strength"])

    nt.links.new(fres.outputs[0], mix.inputs[0])
    nt.links.new(transp.outputs[0], mix.inputs[1])
    nt.links.new(emi.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat

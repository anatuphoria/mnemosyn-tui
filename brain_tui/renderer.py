from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .scene import SceneState, Region

LIGHT_DIR = np.array([0.4, 0.8, 0.6], dtype=np.float32)
LIGHT_DIR /= float(np.linalg.norm(LIGHT_DIR))
AMBIENT = 0.35

LIGHT_CAM = np.array([0.35, 0.55, 0.75], dtype=np.float32)
LIGHT_CAM /= float(np.linalg.norm(LIGHT_CAM))

_NCACHE: dict[str, np.ndarray] = {}


def vertex_normals(rid: str, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals; assumes outward winding."""
    key = f"{rid}:{len(verts)}"
    hit = _NCACHE.get(key)
    if hit is not None and len(hit) == len(verts):
        return hit
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(area, 1e-9)
    n = np.zeros_like(verts)
    w = area.astype(np.float32)
    np.add.at(n, faces[:, 0], fn * w)
    np.add.at(n, faces[:, 1], fn * w)
    np.add.at(n, faces[:, 2], fn * w)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.maximum(l, 1e-9)).astype(np.float32)
    _NCACHE[key] = n
    return n


def shade_lambert(n_cam: np.ndarray, selected: bool = False) -> np.ndarray:
    d = n_cam @ LIGHT_CAM
    s = AMBIENT + 0.75 * np.clip(d, 0.0, 1.0) + 0.15 * np.clip(-d, 0.0, 1.0)
    if selected:
        s = np.minimum(1.3, s + 0.3)
    return s.astype(np.float32)


def _project(points_cam: np.ndarray, fov: float, aspect: float):
    z = -points_cam[:, 2]
    z = np.clip(z, 1e-4, 1e6)
    f = 1.0 / math.tan(fov / 2.0)
    x_ndc = (points_cam[:, 0] * f / aspect) / z
    y_ndc = (points_cam[:, 1] * f) / z
    return x_ndc, y_ndc, z


def render_scene(scene: SceneState, w_px: int, h_px: int):
    w_px = max(8, int(w_px))
    h_px = max(8, int(h_px))
    color = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    color[:, :] = (10, 10, 16)
    depth = np.full((h_px, w_px), np.inf, dtype=np.float32)
    idbuf = np.full((h_px, w_px), -1, dtype=np.int32)

    V = scene.camera.view_matrix()
    aspect = w_px / max(1, h_px)
    fov = scene.camera.fov

    regions = scene.visible_regions()
    id_to_region = [r.id for r in regions]
    by_id = {r.id: i for i, r in enumerate(regions)}

    clip_axis = {"x": 0, "y": 1, "z": 2}[scene.clip.axis]

    for ridx, r in enumerate(regions):
        offset = scene.region_world_center(r) - r.mesh.centroid * 0 + (scene.region_world_center(r) - r.base_center)
        world_center = scene.region_world_center(r)
        verts_obj = r.mesh.verts
        verts_world = verts_obj - r.mesh.centroid + world_center

        faces = r.mesh.faces
        MAX_TRIS = 400
        if len(faces) > MAX_TRIS:
            faces = faces[np.linspace(0, len(faces) - 1, MAX_TRIS).astype(np.int64)]

        if scene.clip.enabled:
            keep_tri = verts_world[faces][:, :, clip_axis].mean(axis=1) <= scene.clip.value
        else:
            keep_tri = None

        ones = np.ones((verts_world.shape[0], 1), dtype=np.float32)
        verts_cam = (V @ np.concatenate([verts_world, ones], axis=1).T).T[:, :3]

        x_ndc, y_ndc, z = _project(verts_cam, fov, aspect)
        sx = ((x_ndc * 0.5 + 0.5) * w_px).astype(np.float32)
        sy = ((-y_ndc * 0.5 + 0.5) * h_px).astype(np.float32)

        base = np.array(r.color, dtype=np.float32)

        for fi in range(faces.shape[0]):
            if keep_tri is not None and not keep_tri[fi]:
                continue
            i0, i1, i2 = (int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2]))
            x0, y0, z0 = float(sx[i0]), float(sy[i0]), float(z[i0])
            x1, y1, z1 = float(sx[i1]), float(sy[i1]), float(z[i1])
            x2, y2, z2 = float(sx[i2]), float(sy[i2]), float(z[i2])
            if z0 <= 0.05 and z1 <= 0.05 and z2 <= 0.05:
                continue
            if max(x0, x1, x2) < 0 or min(x0, x1, x2) >= w_px:
                continue
            if max(y0, y1, y2) < 0 or min(y0, y1, y2) >= h_px:
                continue
            v0, v1, v2 = verts_cam[i0], verts_cam[i1], verts_cam[i2]
            n = np.cross(v1 - v0, v2 - v0)
            nl = float(np.linalg.norm(n))
            if nl < 1e-9:
                continue
            n /= nl
            tri_center = (v0 + v1 + v2) / 3.0
            view_dir = tri_center / max(1e-6, float(np.linalg.norm(tri_center)))
            if float(np.dot(n, view_dir)) > -0.05:
                continue
            d = float(n @ LIGHT_CAM)
            shade = float(np.clip(
                AMBIENT + 0.75 * max(0.0, d) + 0.15 * max(0.0, -d), 0.0, 1.0))
            if r.id == scene.selected_id:
                shade = min(1.3, shade + 0.3)
            shaded = np.clip(base * shade, 0, 255).astype(np.uint8)

            minx = max(0, int(min(x0, x1, x2)))
            maxx = min(w_px - 1, int(max(x0, x1, x2)) + 1)
            miny = max(0, int(min(y0, y1, y2)))
            maxy = min(h_px - 1, int(max(y0, y1, y2)) + 1)
            if maxx < minx or maxy < miny:
                continue
            if (maxx - minx) * (maxy - miny) > 4000:
                pass
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-9:
                continue
            for py in range(miny, maxy + 1):
                w1 = ((y1 - y2) * (minx - x2) + (x2 - x1) * (py - y2)) / denom
                xs = np.arange(minx, maxx + 1, dtype=np.float32)
                w1r = ((y1 - y2) * (xs - x2) + (x2 - x1) * (py - y2)) / denom
                w2r = ((y2 - y0) * (xs - x2) + (x0 - x2) * (py - y2)) / denom
                w0r = 1.0 - w1r - w2r
                mask = (w0r >= -0.02) & (w1r >= -0.02) & (w2r >= -0.02)
                if not bool(mask.any()):
                    continue
                zrow = w0r * z0 + w1r * z1 + w2r * z2
                idx = np.where(mask)[0] + minx
                old = depth[py, idx]
                upd = zrow[mask] < old
                if not bool(upd.any()):
                    continue
                uidx = idx[upd]
                depth[py, uidx] = zrow[mask][upd]
                color[py, uidx] = shaded
                idbuf[py, uidx] = ridx

    img = Image.fromarray(color, "RGB")
    return img, idbuf, id_to_region


_REGION_STATIC: dict[str, dict] = {}
_FULL_STATIC: dict[tuple, dict] = {}
_FULL_BUDGET = 90000
_DRAG_BUDGET = 45000


def _region_static(r: Region) -> dict:
    st = _REGION_STATIC.get(r.id)
    if (st is not None and st["nverts"] == len(r.mesh.verts)
            and st["color"] == tuple(r.color)):
        return st
    F = r.mesh.faces
    Vv = r.mesh.verts
    V0, V1, V2 = Vv[F[:, 0]], Vv[F[:, 1]], Vv[F[:, 2]]
    N0 = vertex_normals(r.id, Vv, F)
    N1, N2, N3 = N0[F[:, 0]], N0[F[:, 1]], N0[F[:, 2]]
    pts = np.concatenate([
        Vv, (V0 + V1 + V2) / 3.0, (V0 + V1) / 2.0, (V1 + V2) / 2.0,
    ]).astype(np.float32)
    nrm = np.concatenate([
        N0, (N1 + N2 + N3) / 3.0, (N1 + N2) / 2.0, (N2 + N3) / 2.0,
    ]).astype(np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    if len(pts) > 1500:
        sel = np.linspace(0, len(pts) - 1, 1500).astype(np.int64)
        pts, nrm = pts[sel], nrm[sel]
    centroid = Vv.mean(axis=0).astype(np.float32)
    radius = float(np.linalg.norm(Vv - centroid, axis=1).max()) + 0.05
    col = np.tile(np.array(r.color, dtype=np.uint8), (len(pts), 1))
    st = {"pts": pts, "nrm": nrm, "col": col, "centroid": centroid,
          "radius": radius, "nverts": len(Vv), "color": tuple(r.color)}
    _REGION_STATIC[r.id] = st
    return st


try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=True, nogil=True)
    def _splat_kernel(xi, yi, dd, cc, ii, depth, color, idbuf, rad, h, w):
        n = xi.shape[0]
        for k in range(n):
            x0 = xi[k]
            y0 = yi[k]
            dk = dd[k]
            cr = cc[k, 0]
            cg = cc[k, 1]
            cb = cc[k, 2]
            ik = ii[k]
            for dy in range(-rad, rad + 1):
                yy = y0 + dy
                if yy < 0 or yy >= h:
                    continue
                for dx in range(-rad, rad + 1):
                    xx = x0 + dx
                    if xx < 0 or xx >= w:
                        continue
                    if dk < depth[yy, xx]:
                        depth[yy, xx] = dk
                        color[yy, xx, 0] = cr
                        color[yy, xx, 1] = cg
                        color[yy, xx, 2] = cb
                        idbuf[yy, xx] = ik

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _splat_numpy(xi, yi, dd, cc, ii, depth, color, idbuf, rad, h, w):
    for dx in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            xx = np.clip(xi + dx, 0, w - 1)
            yy = np.clip(yi + dy, 0, h - 1)
            upd = dd < depth[yy, xx]
            depth[yy[upd], xx[upd]] = dd[upd]
            color[yy[upd], xx[upd]] = cc[upd]
            idbuf[yy[upd], xx[upd]] = ii[upd]


def render_scene_fast(scene: SceneState, w_px: int, h_px: int, splat: int = 2,
                      max_pts: int = _FULL_BUDGET):
    global _HAVE_NUMBA
    w_px = max(8, int(w_px))
    h_px = max(8, int(h_px))
    color = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    color[:, :] = (10, 10, 16)
    depth = np.full((h_px, w_px), np.inf, dtype=np.float32)
    idbuf = np.full((h_px, w_px), -1, dtype=np.int32)

    V = scene.camera.view_matrix()
    R = V[:3, :3]
    t = V[:3, 3]
    aspect = w_px / max(1, h_px)
    fov = scene.camera.fov
    f = 1.0 / math.tan(fov / 2.0)
    regions = scene.visible_regions()
    id_to_region = [r.id for r in regions]
    n = len(regions)
    if n == 0:
        return Image.fromarray(color, "RGB"), idbuf, id_to_region
    clip_axis = {"x": 0, "y": 1, "z": 2}[scene.clip.axis]

    offs = scene.world_centers(regions)

    key = tuple(r.id for r in regions)
    full = _FULL_STATIC.get(key)
    if full is None:
        P_list, N_list, C_list, G_list, Cen_list, Rad_list = [], [], [], [], [], []
        for ridx, r in enumerate(regions):
            st = _region_static(r)
            P_list.append(st["pts"])
            N_list.append(st["nrm"])
            C_list.append(st["col"])
            G_list.append(np.full(len(st["pts"]), ridx, dtype=np.int32))
            Cen_list.append(st["centroid"])
            Rad_list.append(st["radius"])
        full = {"P": np.concatenate(P_list), "N": np.concatenate(N_list),
                "C": np.concatenate(C_list), "G": np.concatenate(G_list),
                "Cen": np.stack(Cen_list),
                "Rad": np.array(Rad_list, dtype=np.float32)}
        _FULL_STATIC[key] = full
    Pall, Nall, Call, Gall, Cen, Rad = (full["P"], full["N"], full["C"],
                                        full["G"], full["Cen"], full["Rad"])

    ccam = offs @ R.T + t
    zc = -ccam[:, 2]
    zsafe = np.clip(zc, 1e-4, 1e6)
    xN = (ccam[:, 0] * f / aspect) / zsafe
    yN = (ccam[:, 1] * f) / zsafe
    m = Rad * f / np.clip(zc, 0.2, 1e6)
    keep_reg = (zc > 0.15) & (np.abs(xN) < 1.0 + m) & (np.abs(yN) < 1.0 + m)
    if not bool(keep_reg.any()):
        return Image.fromarray(color, "RGB"), idbuf, id_to_region
    if bool(keep_reg.all()) and len(Gall) <= max_pts and not scene.clip.enabled:
        P, Nn, Cb, Gg = Pall, Nall, Call, Gall
    else:
        idx = np.nonzero(keep_reg[Gall])[0]
        if len(idx) > max_pts:
            idx = idx[np.linspace(0, len(idx) - 1, max_pts).astype(np.int64)]
        P, Nn, Cb, Gg = Pall[idx], Nall[idx], Call[idx], Gall[idx]

    W = P - Cen[Gg] + offs[Gg]
    if scene.clip.enabled:
        cm = W[:, clip_axis] <= scene.clip.value
        if not bool(cm.any()):
            return Image.fromarray(color, "RGB"), idbuf, id_to_region
        W, Nn, Cb, Gg = W[cm], Nn[cm], Cb[cm], Gg[cm]
    Cam = W @ R.T + t
    Ncam = Nn @ R.T
    z = -Cam[:, 2]
    fm = z > 0.1
    Cam, Ncam, z = Cam[fm], Ncam[fm], z[fm]
    Cb, Gg = Cb[fm], Gg[fm]
    if len(z) == 0:
        return Image.fromarray(color, "RGB"), idbuf, id_to_region
    x_ndc = (Cam[:, 0] * f / aspect) / np.clip(z, 1e-4, 1e6)
    y_ndc = (Cam[:, 1] * f) / np.clip(z, 1e-4, 1e6)
    sx = (x_ndc * 0.5 + 0.5) * w_px
    sy = (-y_ndc * 0.5 + 0.5) * h_px
    on = (sx >= 0) & (sx < w_px) & (sy >= 0) & (sy < h_px)
    sx, sy, z = sx[on], sy[on], z[on]
    Ncam, Cb, Gg = Ncam[on], Cb[on], Gg[on]
    if len(z) == 0:
        return Image.fromarray(color, "RGB"), idbuf, id_to_region

    shade = shade_lambert(Ncam)
    try:
        sel_idx = id_to_region.index(scene.selected_id) if scene.selected_id else -1
    except ValueError:
        sel_idx = -1
    if sel_idx >= 0:
        shade = np.where(Gg == sel_idx, np.minimum(1.3, shade + 0.3), shade)
    zn = (z - z.min()) / max(1e-6, float(z.max() - z.min()))
    shade = np.clip(shade * (1.02 - 0.22 * zn), 0.0, 1.3)
    C = np.clip(Cb.astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)

    order = np.argsort(-z)
    xi = sx[order].astype(np.int64)
    yi = sy[order].astype(np.int64)
    Ds = z[order].astype(np.float32)
    Cs = C[order]
    Is = Gg[order].astype(np.int32)
    rad = splat // 2
    if _HAVE_NUMBA:
        try:
            _splat_kernel(xi, yi, Ds, Cs, Is, depth, color, idbuf,
                          np.int64(rad), np.int64(h_px), np.int64(w_px))
        except Exception:
            _HAVE_NUMBA = False
            _splat_numpy(xi, yi, Ds, Cs, Is, depth, color, idbuf, rad, h_px, w_px)
    else:
        _splat_numpy(xi, yi, Ds, Cs, Is, depth, color, idbuf, rad, h_px, w_px)
    return Image.fromarray(color, "RGB"), idbuf, id_to_region


def render_scene_dispatch(scene: SceneState, w_px: int, h_px: int, hifi: bool = False,
                          max_pts: int = _FULL_BUDGET):
    if hifi:
        img, idbuf, ids = render_scene_fast(scene, w_px * 2, h_px * 2, splat=3,
                                            max_pts=max_pts * 2)
        img = img.resize((w_px, h_px), Image.LANCZOS)
        return img, idbuf, ids
    return render_scene_fast(scene, w_px, h_px, max_pts=max_pts)


def pick_region(scene: SceneState, idbuf: np.ndarray, id_to_region: list[str], px: int, py: int) -> str | None:
    h, w = idbuf.shape
    if not (0 <= px < w and 0 <= py < h):
        return None
    ridx = int(idbuf[py, px])
    if ridx < 0 or ridx >= len(id_to_region):
        return None
    return id_to_region[ridx]


def project_centroids(scene: SceneState, w_px: int, h_px: int) -> dict[str, tuple[float, float, float]]:
    regions = scene.visible_regions()
    if not regions:
        return {}
    V = scene.camera.view_matrix()
    cam = scene.world_centers(regions) @ V[:3, :3].T + V[:3, 3]
    x_ndc, y_ndc, z = _project(cam, scene.camera.fov, w_px / max(1, h_px))
    sx = (x_ndc * 0.5 + 0.5) * w_px
    sy = (-y_ndc * 0.5 + 0.5) * h_px
    return {r.id: (float(sx[i]), float(sy[i]), float(z[i])) for i, r in enumerate(regions)}

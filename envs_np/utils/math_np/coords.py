# 坐标系转换模块
from __future__ import annotations
from ._head import *
from .rotation import *
import pymap3d
from pymap3d import Ellipsoid

_PI = PI
_PI_HALF = _PI * 0.5
_NPI = -_PI


def _ned2aer(xyz: ndarray) -> ndarray:
    _R = norm(xyz, 2, -1, True)
    x = xyz[..., 0:1]
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]  # (...,1)
    _is0 = _R < 1e-3  # 过零处理
    x = where(_is0, 1, x)
    y = where(_is0, 0, y)
    z = where(_is0, 0, z)
    rxy2 = pow(y, 2) + pow(x, 2)
    rxy = sqrt(rxy2)
    slantRange = sqrt(rxy2 + pow(z, 2))

    elev = atan2(-z, rxy)  # -> [-pi/2,pi/2]
    azi = atan2(x, y)  # -> [0,2pi)
    # azi = azi % _2PI
    return cat([azi, elev, slantRange], axis=-1)


def ned2aer(xyz: ndarray) -> ndarray:
    r"""求解 NED xyz 直角坐标对应的 方位角 azimuth, 俯仰角 elevation, 距离 r\
    即 (r,0,0) 依次 绕Z内旋 azimuth, 绕Y内旋 elevation 得到 (x,y,z), 右手定则

    Args:
        xyz (_NDArr): NED 直角坐标 shape: (...,3)

    Returns:
        aer (_NDArr): shape: (...,3)
            azimuth \in [-pi,pi)
            elevation \in [-pi/2,pi/2]
            slant range \in [0,inf)
    """
    return _ned2aer(xyz)


def _aer2ned(aer: ndarray) -> ndarray:
    az = aer[..., 0:1]
    el = aer[..., 1:2]
    r = aer[..., 2:3]  # (...,1)
    rxy = r * cos(el)
    z = -r * sin(el)
    x = rxy * cos(az)
    y = rxy * sin(az)
    return cat([x, y, z], axis=-1)


def aer2ned(aer: ndarray) -> ndarray:
    """
    ned2aer 的逆映射
    Args:
        aer (_NDArr): 方位角 azimuth, 俯仰角 elevation, 距离 r\
            即 (r,0,0) 依次 绕Z内旋 azimuth, 绕Y内旋 elevation 得到 (x,y,z), 右手定则 shape: (...,3)

    Returns:
        ned (_NDArr): NED 直角坐标 shape: (...,3)
    """
    return _aer2ned(aer)


def _uvw2alpha_beta(uvw: ndarray, axis=-1):
    assert uvw.shape[axis] == 3, f"uvw must be 3D in dim[{axis}]"
    uvw = normalize(uvw)
    u, v, w = unbind_keepdim(uvw, axis)  # (...,1)
    alpha = atan2(w, u)  # (0,0)->0
    beta = asin(v)
    return alpha, beta


def uvw2alpha_beta(uvw: ndarray):
    r"""
    NED 体轴速度分量 (U,V,W)->(\alpha,\beta)\
    坐标系 旋转关系\
    $$
    \Phi_v R_z(-\beta) R_y(\alpha) = \Phi_b
    $$

    (U,V,W) 单位向量的分解

    $$
    i_{v/b} = ( \cos\beta \cos\alpha, 
                \sin\beta, 
                \cos\beta \sin\alpha)
    $$

    Args:
        uvw (_NDArr): 体轴速度分量 shape: (...,3)

    Returns:
        alpha (_NDArr): 迎角 \in [-pi/2,pi/2] shape: (...,1)
        beta (_NDArr): 侧滑角 \in (-pi,pi] shape: (...,1)
    """
    # assert (norm(uvw)>0).all()
    return _uvw2alpha_beta(uvw)


def quat_enu_ned() -> ndarray:
    return quat_mul(
        Qz(asarray([PI * 0.5])),
        Qx(asarray([PI])),
    )  # (4,)


def ned2enu(xyz, axis=-1) -> ndarray:
    xyz = asarray(xyz)
    n, e, d = unbind_keepdim(xyz, axis)  # (...,1)
    xyz = cat([e, n, -d], axis=axis)
    return xyz


def enu2ned(xyz, axis=-1) -> ndarray:
    xyz = asarray(xyz)
    e, n, u = unbind_keepdim(xyz, axis)  # (...,1)
    xyz = cat([n, e, -u], axis=axis)
    return xyz


def nue2ned(xyz, axis=-1) -> ndarray:
    xyz = asarray(xyz)
    n, u, e = unbind_keepdim(xyz, axis)  # (...,1)
    xyz = cat([n, e, -u], axis=axis)
    return xyz


def ned2nue(xyz, axis=-1) -> ndarray:
    xyz = asarray(xyz)
    n, e, d = unbind_keepdim(xyz, axis)  # (...,1)
    xyz = cat([n, -d, e], axis=axis)
    return xyz


def nue2enu(xyz, axis=-1) -> ndarray:
    xyz = asarray(xyz)
    n, u, e = unbind_keepdim(xyz, axis)  # (...,1)
    xyz = cat([e, n, u], axis=axis)
    return xyz


def enu2nue(xyz, axis=-1) -> ndarray:
    xyz = asarray(xyz)
    e, n, u = unbind_keepdim(xyz, axis)  # (...,1)
    xyz = cat([n, u, e], axis=axis)
    return xyz


def rpy_NEDLight2Len(rpy_rad: ndarray):
    """将光线系的 NED 姿态 转为相机 amuzi 姿态， 从目标坐标系到两个坐标系的旋转顺序均为 ZYX"""
    r, p, y = rpy_reg(rpy_rad)
    r_c = modrad(-r, _NPI)
    p_c = _PI_HALF - p  # 转为天顶角, in [0, pi]
    y_c = modrad(-y, _NPI)
    return r_c, p_c, y_c


def rpy_NEDLight2Len_inv(roll_rad: float, pitch_rad: float, yaw_rad: float):
    r_l = modrad(-roll_rad, _NPI)
    p_l = _PI_HALF - pitch_rad  # 转为俯仰角, in [-\pi/2, \pi/2]
    y_l = modrad(-yaw_rad, _NPI)
    return r_l, p_l, y_l


def T_NEDLight_Pic():
    r"""坐标旋转矩阵 T_{LP}, $\Phi_L$=NED 光线系, $\Phi_P$= 右-下-前 图像坐标系;
    $\xi_L = T_{LP} \xi_P$
    """
    return asarray(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
    )


ELL = Ellipsoid.from_name("wgs84")


def geodetic2ecef(
    lat: ndarray,
    lon: ndarray,
    alt: ndarray,
    ell: Ellipsoid = ELL,
    deg: bool = True,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    point transformation from Geodetic of specified ellipsoid (default WGS-84) to ECEF

    Parameters
    ----------

    lat:
           target geodetic latitude, shape=(...)
    lon
           target geodetic longitude, shape=(...)
    alt
         target altitude above geodetic ellipsoid (meters), shape=(...)
    ell : Ellipsoid, optional
          reference ellipsoid
    deg : bool, optional
          degrees input/output  (False: radians in/out)


    Returns
    -------

    ECEF (Earth centered, Earth fixed)  x,y,z

    x
        target x ECEF coordinate (meters), shape=(...)
    y
        target y ECEF coordinate (meters), shape=(...)
    z
        target z ECEF coordinate (meters), shape=(...)
    """
    # pymap3d.geodetic2ecef

    if deg:
        lat = deg2rad(lat)
        lon = deg2rad(lon)

    cosB = cos(lat)
    sinB = sin(lat)

    a = ell.semimajor_axis
    b = ell.semiminor_axis
    ba2 = (b / a) ** 2

    # radius of curvature of the prime vertical section
    N = a / sqrt(cosB**2 + (sinB**2) * ba2)
    # Compute cartesian (geocentric) coordinates given (curvilinear) geodetic coordinates.
    rxy = (N + alt) * cosB
    x = rxy * cos(lon)
    y = rxy * sin(lon)
    z = (N * ba2 + alt) * sinB
    return x, y, z


def ecef2geodetic(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    ell: Ellipsoid = ELL,
    deg: bool = True,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    convert ECEF (meters) to geodetic coordinates

    Parameters
    ----------
    x
        target x ECEF coordinate (meters), shape=(...,1)
    y
        target y ECEF coordinate (meters), shape=(...,1)
    z
        target z ECEF coordinate (meters), shape=(...,1)
    ell : Ellipsoid, optional
          reference ellipsoid
    deg : bool, optional
          degrees input/output  (False: radians in/out)

    Returns
    -------
    lat
           target geodetic latitude
    lon
           target geodetic longitude
    alt
         target altitude above geodetic ellipsoid (meters)

    based on:
    You, Rey-Jer. (2000). Transformation of Cartesian to Geodetic Coordinates without Iterations.
    Journal of Surveying Engineering. doi: 10.1061/(ASCE)0733-9453
    """
    pymap3d.ecef2geodetic
    x2 = x**2
    y2 = y**2
    z2 = z**2
    rxy2 = x2 + y2
    r2 = rxy2 + z2
    r = sqrt(r2)
    _0 = zeros_like(r)

    E = math.sqrt(ell.semimajor_axis**2 - ell.semiminor_axis**2)
    E2 = E**2

    # eqn. 4a
    r2e2 = r2 - E2
    u2 = 0.5 * (r2e2 + hypot(r2e2, (2 * E) * z))
    u = sqrt(u2)

    rxy = hypot(x, y)

    huE = sqrt(u2 + E2)

    # eqn. 4b
    Beta = empty_like(r)  # Beta
    ibad = isclose(u, _0) | isclose(rxy, _0)
    #
    zis0 = isclose(z, _0)
    msk1 = ~ibad
    msk2 = ibad & zis0
    _ = ibad & ~zis0
    msk3 = _ & (z > 0)
    msk4 = _ & (z < 0)

    Beta[msk2] = 0
    Beta[msk3] = PI * 0.5
    Beta[msk4] = -PI * 0.5
    #
    huE_ = huE[msk1]
    rxy_ = rxy[msk1]
    u_ = u[msk1]
    z_ = z[msk1]
    B_ = atan2(huE_ * z_, rxy_ * u_)
    # eqn. 13
    huE_a = huE_ * ell.semimajor_axis
    u_b = u_ * ell.semiminor_axis
    Beta[msk1] = B_ + ((u_b - huE_a + E2) * tan(B_) / (huE_a / (cos(B_) ** 2) - E2))

    # eqn. 4c
    # %% final output
    lat = atan2(sin(Beta) * (ell.semimajor_axis / ell.semiminor_axis), cos(Beta))
    lim_pi2 = PI * 0.5 - bkbn.finfo(Beta.dtype).eps
    mskB1 = Beta >= lim_pi2
    mskB2 = Beta <= -lim_pi2
    lat[mskB1] = PI * 0.5
    lat[mskB2] = -PI * 0.5

    # eqn. 7
    lon = atan2(y, x)
    cosB = cos(Beta)
    cosB[mskB1 | mskB2] = 0

    alt = hypot(z - ell.semiminor_axis * sin(Beta), rxy - ell.semimajor_axis * cosB)
    # inside ellipsoid?
    inside = rxy2 / (ell.semimajor_axis**2) + z2 / (ell.semiminor_axis**2) < 1
    alt = where(inside, -alt, alt)

    if deg:
        lat = rad2deg(lat)
        lon = rad2deg(lon)
    return lat, lon, alt

# 250611 taview 可视化扩展
from __future__ import annotations
from datetime import datetime
from enum import Enum
import logging
import os
from pathlib import Path
import socket
import traceback
from typing import Any


class Tag_Class(Enum):
    """
    大类
    最后一次同步 250603
    """

    Air = "Air"
    Ground = "Ground"
    Sea = "Sea"
    Weapon = "Weapon"
    Sensor = "Sensor"
    Navaid = "Navaid"
    Misc = "Misc"


class Tag_Attributes(Enum):
    """
    类型
    最后一次同步 250603
    """

    Static = "Static"
    Heavy = "Heavy"
    Medium = "Medium"
    Light = "Light"
    Minor = "Minor"


class Tag_BasicTypes(Enum):
    """
    基础类型
    最后一次同步 250603
    """

    FixedWing = "FixedWing"
    """固定翼"""
    Rotorcraft = "Rotorcraft"
    """旋翼"""
    Armor = "Armor"
    AntiAircraft = "AntiAircraft"
    Vehicle = "Vehicle"
    Watercraft = "Watercraft"
    Human = "Human"
    Biologic = "Biologic"
    Missile = "Missile"
    Rocket = "Rocket"
    Bomb = "Bomb"
    Torpedo = "Torpedo"
    Projectile = "Projectile"
    Beam = "Beam"
    Decoy = "Decoy"
    """诱饵"""
    Building = "Building"
    Bullseye = "Bullseye"
    Waypoint = "Waypoint"
    """路径点"""


class Tag_SpecificTypes(Enum):
    """
    特殊类型
    最后一次同步 250603
    """

    Tank = "Tank"
    Warship = "Warship"
    AircraftCarrier = "AircraftCarrier"
    Submarine = "Submarine"
    Infantry = "Infantry"
    Parachutist = "Parachutist"
    Shell = "Shell"
    Bullet = "Bullet"
    Grenade = "Grenade"
    Flare = "Flare"
    Chaff = "Chaff"
    SmokeGrenade = "SmokeGrenade"
    Aerodrome = "Aerodrome"
    Container = "Container"
    Shrapnel = "Shrapnel"
    Explosion = "Explosion"


class ACMI_Types(Enum):
    """
    可参考的组合类型(可直接打印)
    最后一次同步 250603
    """

    Plane = "Air+FixedWing"
    Helicopter = "Air+Rotorcraft"
    AntiAircraft = "Ground+AntiAircraft"
    Armor = "Ground+Heavy+Armor+Vehicle"
    Tank = "Ground+Heavy+Armor+Vehicle+Tank"
    GroundVehicle = "Ground+Vehicle"
    Watercraft = "Sea+Watercraft"
    Warship = "Sea+Watercraft+Warship"
    AircraftCarrier = "Sea+Watercraft+AircraftCarrier"
    Submarine = "Sea+Watercraft+Submarine"
    Sonobuoy = "Sea+Sensor"
    Human = "Ground+Light+Human"
    Infantry = "Ground+Light+Human+Infantry"
    Parachutist = "Ground+Light+Human+Air+Parachutist"
    Missile = "Weapon+Missile"
    Rocket = "Weapon+Rocket"
    Bomb = "Weapon+Bomb"
    Projectile = "Weapon+Projectile"
    Beam = "Weapon+Beam"
    Shell = "Projectile+Shell"
    Bullet = "Projectile+Bullet"
    BallisticShell = "Projectile+Shell+Heavy"
    Grenade = "Projectile+Grenade"
    Decoy = "Misc+Decoy"
    Flare = "Misc+Decoy+Flare"
    Chaff = "Misc+Decoy+Chaff"
    SmokeGrenade = "Misc+Decoy+SmokeGrenade"
    Building = "Ground+Static+Building"
    Aerodrome = "Ground+Static+Aerodrome"
    Bullseye = "Navaid+Static+Bullseye"
    Waypoint = "Navaid+Static+Waypoint"
    Container = "Misc+Container"
    Shrapnel = "Misc+Shrapnel"
    MinorObject = "Misc+Minor"
    Explosion = "Misc+Explosion"
    FlareDecoy = "Misc+Decoy+Flare"


_FILE_ENCODING = "utf-8-sig"  # 本地文件编码格式
_RT_ENCODING = "utf-8"  # 实时遥测数据编码格式
__ACMI_HEAD = "FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime="


def acmi_head(reftime: datetime):
    """ACMI文件头(与实时遥测共用)"""
    t = reftime.strftime("%Y-%m-%dT%H:%M:%SZ")
    head_acmi = __ACMI_HEAD + t + "\n"
    return head_acmi


def format_timestamp(sec: float) -> str:
    """格式化ACMI时间戳"""
    return f"#{sec:.2f}"


def format_id(id: int | str) -> str:
    r"""
    格式化为16进制大写 Tacview Object ID\
    注意, 0x/0 开头会导致 Tacview 解析失败
    """
    if isinstance(id, str):
        try:
            id = int(id, 16)
        except ValueError as e:
            raise ValueError(f"invalid id: {id}") from e
    id_hex = "{:X}".format(id)
    return id_hex


def format_destroy(id: str):
    """坠毁事件"""
    msg = f"0,Event=Destroyed|{id}|"
    return msg


def format_remove(id: str):
    """删除对象"""
    msg = f"-{id}"
    return msg


def format_bookmark(msg: str):
    return f"0,Event=Bookmark|{msg}"


def format_unit(
    id: str | int,
    lat: float,  # deg
    lon: float,  # deg
    alt: float,  # m
    roll: float | Any = None,  # deg
    pitch: float | Any = None,  # deg
    yaw: float | Any = None,  # deg
    Name: str | Any = None,
    Color: str | Any = None,
    Type: str | Any = None,
    CallSign: str | Any = None,
    TAS: float | str | Any = None,  # 真空速
    Speed: float | str | Any = None,  # 航速
    Parent: str | Any = None,  # 父对象 ID
    Next: str | Any = None,  # 下一个导航点 ID
    **etc: str | Any,  # 其他元数据
) -> str:
    """
    格式化对象信息
    Args:
        reltime (float): 相对时间, sec
        id (str): ID, 16进制大写字符串(无0x前缀)
        lon (float): 经度, deg
        lat (float): 纬度, deg
        alt (float): 高度, m
        roll (Optional[float], optional): 滚转角, deg. Defaults to None.
        pitch (Optional[float], optional): 俯仰角, deg. Defaults to None.
        yaw (Optional[float], optional): 偏航角, deg. Defaults to None.
        Name (Optional[str], optional): 模型名(Tacview 本地数据库可检索的名称). Defaults to None.
        Color (Optional[str], optional): 阵营颜色. Defaults to None.
        Type (Optional[str], optional): 物体类型. Defaults to None.
        CallSign (Optional[str], optional): 呼号. Defaults to None.
        TAS (Optional[float], optional): 真空速, knots. Defaults to None.
        Speed (Optional[float], optional): 航速, knots. Defaults to None.
        Parent (Optional[str], optional): 父对象 ID. Defaults to None.
        Next (Optional[str], optional): 下一个导航点 ID. Defaults to None.
        **etc (str): 其他元数据, 格式为 key=value, 多个元数据以逗号分隔,
            参见 "Text Properties"@ https://www.tacview.net/documentation/acmi/en/ .
    """
    lbh_str = "{:.07f}|{:.07f}|{:.02f}".format(float(lon), float(lat), float(alt))
    pose_ = [lbh_str]
    if not (roll is None or pitch is None or yaw is None):
        rpy_str = "{:.01f}|{:.01f}|{:.01f}".format(
            float(roll), float(pitch), float(yaw)
        )
        pose_.append(rpy_str)
    pose_str = "T=" + ("|".join(pose_))

    id_hex = format_id(id)
    state_str = [id_hex, pose_str]
    _meta = [
        ("Name", Name),
        ("Color", Color),
        ("Type", Type),
        ("CallSign", CallSign),
        ("TAS", TAS),
        ("Speed", Speed),
        ("Parent", Parent),
        ("Next", Next),
    ]
    _meta.extend(etc.items())
    for k, v in _meta:
        if v is None or (isinstance(v, str) and len(v) == 0):
            continue
        state_str.append(f"{k}={v}")
    state_str = ",".join(state_str)
    return state_str


class TacviewRecorder:
    """负责将一局交战数据实时转发到 tacview 遥测端 or 本地文件"""

    _obj_count = 0  # (被进程共享)

    def __init__(self):
        cls = self.__class__
        cls._obj_count += 1
        self._id = f"{os.getpid()}_{self._obj_count}"
        self._logr = logr = logging.getLogger(f"{cls.__name__}.{self._id}")
        logr.setLevel(logging.DEBUG)

        self._srvr_sock: " socket.socket|None" = None
        self._clnt_sock: "socket.socket|None" = None
        self.__is_connected = False
        self.__is_reset = False
        self._fn: "Path|None" = None

        self.format_id = format_id
        self.format_unit = format_unit
        self.format_time = format_timestamp
        self.format_destroy = format_destroy
        self.format_bookmark = format_bookmark
        self.format_remove = format_remove

    def reset_local(
        self,
        fn: "os.PathLike[str]|str",
        reference_time: "datetime|None" = None,
        encoding=_FILE_ENCODING,
    ):
        logr = self._logr
        if reference_time is None:
            reference_time = datetime(2022, 1, 1, 0, 0, 0)
        fn = Path(fn)
        if fn.exists():
            logr.warning(f"overwriting existing acmi file: {fn}")
        else:
            fn.parent.mkdir(parents=True, exist_ok=True)
        with open(fn, "w", encoding=encoding) as f:
            f.write(acmi_head(reference_time))
            logr.debug(f"acmi>>{fn}")

        self._fn = fn
        self._buf = []
        self.__is_reset = True

    def reset_remote(
        self,
        addr: tuple[str, int],
        timeout: float = 5.0,
        reference_time: "datetime|None" = None,
        encoding=_RT_ENCODING,
    ):
        logr = self._logr
        suc = False
        if reference_time is None:
            reference_time = datetime(2022, 1, 1, 0, 0, 0)

        head_acmi = acmi_head(reference_time)

        ip, port = addr

        self.close()  # 在 bind 前需要关闭已有连接

        clnt_sock = None
        srvr_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srvr_sock.settimeout(timeout)  # timeout for connection
        srvr_sock.bind(addr)
        srvr_sock.listen(1)  # 最多排队的 clients 连接数
        logr.debug(f"Server listening on {ip}:{port}, timeout={timeout}s")

        # [1]等待连接
        logr.info(
            f"请在 {float(timeout):.0f}s 内打开 Tacview*高级版*->记录->实时遥测, 填写以下内容:\n"
            f"数据记录器地址: {ip}\n"
            f"数据记录器端口: {port}\n",
        )
        # funcname = "{}.{}".format(self.__class__.__name__, self.reset_remote.__name__)
        # print(
        #     "若发生 RealTimeTelemetry 连接失败, 请依次执行以下步骤:",
        #     "1. Tacview->记录->实时遥测->断开连接",
        #     f"2. 等待本次 {funcname} 结束",
        #     "3. Tacview->记录->实时遥测->连接",
        #     f"4. 重新调用 {funcname}",
        #     sep="\n",
        # )
        try:  # TCP
            clnt_sock, clnt_addr = srvr_sock.accept()
            logr.debug(f"created Tacview client on {clnt_addr}")
        except socket.timeout as e:
            logr.warning(f"no client: {e}")

        if clnt_sock:
            # [2]握手(只需按协议发送消息即可)
            head_com = "XtraLib.Stream.0\n" "Tacview.RealTimeTelemetry.0\n"  # 协议头
            head_clnt = head_com + f"Client username\npassword_hash\0"
            msg1 = head_clnt.encode(encoding)
            try:
                clnt_sock.send(msg1)
                logr.debug(f"sent handshake data:\n{len(msg1)},{msg1}")
                data = clnt_sock.recv(2048)
                msg2 = data.decode(encoding)
                logr.debug(
                    f"received handshake data:\n{len(msg2)},{msg2}"
                )  # 后两行依次是在 Tacview 界面中填写的 用户名,密码哈希值
                # (可选)解析数据并进行用户名、密码校验
                logr.debug("handshake done")
            except socket.timeout as e:
                logr.warning("handshake timeout")
                clnt_sock.close()
                clnt_sock = None

        if clnt_sock:
            try:
                clnt_sock.send(head_acmi.encode(encoding))
                logr.debug(f"sent acmi header")
            except Exception as e:
                logr.warning(f"send acmi header error: {e}")
                clnt_sock.close()
                clnt_sock = None
        #
        if clnt_sock:
            self._srvr_sock = srvr_sock
            self._clnt_sock = clnt_sock
            self.__is_connected = True
            suc = True
        return suc

    def is_connected(self):
        return self.__is_connected

    def add(self, line: str):
        self._buf.append(line)

    def merge(self, sep="\n", clear=True):
        r"""合并缓存消息"""
        buf = self._buf
        if len(buf):
            msg = sep.join(self._buf)
            if clear:
                self.clear_buf()
        else:
            msg = ""
        return msg

    def clear_buf(self):
        self._buf.clear()

    def _check_reset(self):
        assert self.__is_reset, "not reset"

    def write_local(self, msg: str, endl="\n", encoding=_FILE_ENCODING):
        r"""直接写入消息到文件"""
        self._check_reset()
        assert self._fn is not None
        with open(self._fn, "a", encoding=encoding) as f:
            f.write(msg + endl)

    def write_remote(self, msg: str, endl="\n", encoding=_RT_ENCODING):
        r"""直接写入消息到客户端"""
        logr = self._logr
        suc = False
        self._check_reset()
        assert self.__is_connected, "not connected"
        try:
            data = (msg + endl).encode(encoding)
            assert self._clnt_sock
            self._clnt_sock.send(data)
            suc = True
        except socket.timeout as e:
            logr.debug(f"send timeout")
        except ConnectionError as e:
            logr.warning(f"connection error: {e}")
            self._close_client()
        except Exception as e:
            logr.warning(f"send error: {e}\n{traceback.format_exc()}")
        return suc

    def _close_client(self):
        """关闭与当前客户端的连接"""
        self.__is_connected = False
        if self._clnt_sock:
            self._clnt_sock.close()
            self._clnt_sock = None

    def close(self):
        """关闭所有socket"""
        self._close_client()
        for attn, attv in self.__dict__.items():
            if isinstance(attv, socket.socket):
                attv.close()
                setattr(self, attn, None)

    def __del__(self):
        self.close()

    @staticmethod
    def demo():
        from pymap3d import ned2geodetic
        import numpy as np
        import time

        dir_out = Path(__file__).parent / "tmp"
        rng = np.random.default_rng()

        from logging import basicConfig

        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s|%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d-%H:%M:%S",
        )  # 通过日志显示信息
        render = TacviewRecorder()
        id0 = 0xA0000
        nagents = 10
        colors = np.random.choice(["Red", "Blue"], nagents, replace=True)
        lat0 = 35.9
        lon0 = 120.0
        alt0 = 10000.0

        def rpy2mat(rpy):
            psi, theta, phi = np.split(rpy, 3, axis=-1)  # (...,1)
            newshape = psi.shape[:-1] + (3, 3)
            _1 = np.ones_like(psi)
            _0 = np.zeros_like(psi)
            c1 = np.cos(phi)
            s1 = np.sin(phi)
            c2 = np.cos(theta)
            s2 = np.sin(theta)
            c3 = np.cos(psi)
            s3 = np.sin(psi)
            rx = np.concatenate([_1, _0, _0, _0, c1, -s1, _0, s1, c1], axis=-1).reshape(
                newshape
            )  # (...,3,3)
            ry = np.concatenate([c2, _0, s2, _0, _1, _0, -s2, _0, c2], axis=-1).reshape(
                newshape
            )  # (...,3,3)
            rz = np.concatenate([c3, -s3, _0, s3, c3, _0, _0, _0, _1], axis=-1).reshape(
                newshape
            )  # (...,3,3)
            return rz @ ry @ rx  # (...,3,3)

        def modin(x, a, b):
            a = np.asarray(a)
            m = b - a
            return np.where(m == 0, a, (x - a) % m + a)

        dt = 1.0 / 20
        t_tol = 10.0
        try:
            for itr in range(3):
                if itr == 0:
                    ip = "127.0.0.1"
                else:
                    ip = socket.gethostbyname(socket.gethostname())
                render.reset_remote(
                    addr=(ip, 21567),
                    timeout=10.0,
                )
                render.reset_local(dir_out / f"test{itr}.acmi")

                t0 = time.time()
                p_e = rng.random((nagents, 3)) * 1e3
                rpy = np.deg2rad(rng.random((nagents, 3)) * 30)
                V = rng.uniform(240, 340, (nagents, 1))
                _1 = np.ones_like(V)
                _0 = np.zeros_like(V)
                e1 = np.concatenate([_1, _0, _0], axis=-1)
                rpy_low = np.reshape((-np.pi, -np.pi / 2, -np.pi), (1, 3))
                rpy_high = np.reshape((np.pi, np.pi / 2, np.pi), (1, 3))
                k = 0
                while True:
                    t = time.time() - t0
                    if t > t_tol:
                        break
                    V += np.random.uniform(-1, 1) * 1e-2 * V
                    rpy += np.random.uniform(-1, 1, (nagents, 3)) * 1e-2
                    rpy = modin(rpy, rpy_low, rpy_high)
                    Rew = rpy2mat(rpy)
                    v_e = (Rew @ ((V * e1).reshape(-1, 3, 1))).reshape(-1, 3)
                    p_e += v_e * dt
                    blh = np.asarray(
                        [
                            ned2geodetic(xyz[0], xyz[1], xyz[2], lat0, lon0, alt0)
                            for xyz in p_e
                        ]
                    )
                    render.add(render.format_time(t))
                    nagt_ = np.random.randint(1, nagents + 1)
                    idxs = np.random.choice(nagents, nagt_, replace=False)
                    rpy_deg = np.rad2deg(rpy)
                    for i in idxs:
                        unit_id = render.format_id(id0 + i)
                        render.add(
                            render.format_unit(
                                unit_id,
                                *blh[i],
                                *rpy_deg[i],
                                Color=str(colors[i]),
                                Name="J-20",
                            )
                        )
                    msg = render.merge()
                    render.write_local(msg)
                    render.write_remote(msg)
                    k += 1
                    tk = t0 + k * dt
                    time.sleep(max(0, tk - time.time()))

        except KeyboardInterrupt:
            pass
        # tacview.close()
        return


if __name__ == "__main__":
    TacviewRecorder.demo()

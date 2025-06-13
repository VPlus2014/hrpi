from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import numpy as np
import pymap3d
import hashlib
from pydantic import BaseModel, Field


class ObjectAttr(BaseModel):
    pass


class AircraftAttr(ObjectAttr):
    Name: str = "F-16C-52"
    Color: Literal["Red", "Blue"] | str
    TAS: float | str | None = None
    AOA: float | str | None = None
    CallSign: str | None = None


class MissileAttr(ObjectAttr):
    Name: str = "AIM_9"
    Color: Literal["Red", "Blue"] | str
    TAS: float | str
    Radius: float | str | None = None
    CallSign: str | None = None


class DecoyAttr(ObjectAttr):
    Type: str = "Misc+Decoy+Flare"
    CallSign: str | None = None


class WaypointAttr(ObjectAttr):
    Type: str = "Navaid+Static+Waypoint"
    Next: str = Field(
        default="", init=False
    )  # 后继点UID(在初始化时不直接接受 Next=... 形式输入)
    CallSign: str | None = None

    def __init__(self, Next: str, **data):
        super().__init__(**data)
        self.Next = get_obj_id(Next)  # 使用 get_obj_id 生成 Next 属性


class TacviewEvent(BaseModel):
    EventName: Literal["Message", "Bookmark", "Destroyed"]
    FirstObjectName: str | None = None
    SecondObjectName: str | None = None
    EventText: str | None = None


def get_obj_id(obj_name: str):
    try:
        uid = int(obj_name, 16)
        hex_value = hex(uid)[2:].upper()
    except ValueError:
        hash_md5 = hashlib.md5(obj_name.encode())
        hash_value = hash_md5.hexdigest()
        hex_value = hash_value[:7]
    return hex_value.upper()


class ObjectState:

    def __init__(
        self,
        sim_time_s: float,
        name: str,  # [ID]唯一的名称->16进制 Object ID
        attr: ObjectAttr | AircraftAttr | MissileAttr,
        pos_ned: np.ndarray,
        lat0: float = 30,
        lon0: float = 120,
        h0: float = 10000,
        rpy_rad: np.ndarray | None = None,
    ):
        self.sim_time_s = sim_time_s
        self.id = get_obj_id(name)  # UID
        # 记录目标属性
        self.attr = attr
        pos_blh = pymap3d.ned2geodetic(*(pos_ned.tolist()), lat0=lat0, lon0=lon0, h0=h0)
        self.pos_lbh = [pos_blh[1], pos_blh[0], pos_blh[2]]
        # 记录目标姿态
        if rpy_rad is not None:
            self.rpy_deg = np.rad2deg(rpy_rad).tolist()
        else:
            self.rpy_deg = None
        self.event_list: list[TacviewEvent] = []

    @property
    def pos_lla(self):
        return self.pos_lbh

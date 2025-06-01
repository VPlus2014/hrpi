from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
import pymap3d
import torch
import hashlib
from pydantic import BaseModel, Field


class ObjectAttr(BaseModel):
    pass


class AircraftAttr(ObjectAttr):
    Name: str = "F-16C-52"
    Color: Literal["Red", "Blue"] | str
    TAS: float | None = None
    AOA: float | None = None


class MissileAttr(ObjectAttr):
    Name: str = "AIM_9"
    Color: Literal["Red", "Blue"] | str
    TAS: float
    Radius: float | None = None


class DecoyAttr(ObjectAttr):
    Type: str = "Misc+Decoy+Flare"


class WaypointAttr(ObjectAttr):
    Type: str = "Navaid+Static+Waypoint"
    Next: str = Field(default=None, init=False)  # Next 属性在初始化时不直接接受输入

    def __init__(self, name: str, **data):
        super().__init__(**data)
        self.Next = get_obj_id(name)  # 使用 get_obj_id 生成 Next 属性


class TacviewEvent(BaseModel):
    EventName: Literal["Message", "Bookmark", "Destroyed"]
    FirstObjectName: str | None = None
    SecondObjectName: str | None = None
    EventText: str | None = None


def get_obj_id(obj_name: str):
    hash_md5 = hashlib.md5(obj_name.encode())
    hash_value = hash_md5.hexdigest()
    hex_value = hash_value[:7]
    return hex_value.upper()


class ObjectState:
    def __init__(
        self,
        sim_time_s: float,
        name: str,
        attr: ObjectAttr | AircraftAttr | MissileAttr,
        pos_ned: torch.Tensor,
        rpy_rad: torch.Tensor | None = None,
    ):
        self.sim_time_s = sim_time_s
        self.id = get_obj_id(name)
        # 记录目标属性
        self.attr = attr
        # 记录目标位置
        pos_lla = pymap3d.ned2geodetic(*pos_ned.tolist(), lat0=30, lon0=120, h0=0)
        self.pos_lla = [pos_lla[1], pos_lla[0], pos_lla[2]]
        # 记录目标姿态
        if rpy_rad is not None:
            self.rpy_deg = torch.rad2deg(rpy_rad).tolist()
        else:
            self.rpy_deg = None
        self.event_list: list[TacviewEvent] = []

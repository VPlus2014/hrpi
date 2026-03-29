from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import logging
import numpy as np
from typing import Any, List, Dict, Tuple
from ..simulators.plane import BasePlane, PlaneP6DOF
from ..simulators.missile import BaseMissile, MissileP6DOF
from ..simulators.decoy import BaseDecoy, DecoyDOF0Ball


class PMDPEGCore:
    CAP_MISSILE = 16  # 1飞机对应最大挂载导弹数
    CAP_DECOY = 16**2  # 1飞机对应最大挂载诱饵数

    ADDRMUL_MISSILE = CAP_MISSILE
    ADDRMUL_DECOY = CAP_MISSILE * CAP_DECOY

    def __init__(
        self,
        num_aircraft: int,
        missiles_per_aircraft_max: int,
        decoys_per_aircraft_max: int = 0,
        config: dict = {},
        fri_color="Blue",
        enm_color="Red",
        logger: logging.Logger | None = None,
    ):
        """
        多机多弹交互对局核心逻辑

        Args:
            missiles_per_aircraft: 单机最大挂载导弹数
            decoys_per_aircraft_max: 单机最大挂载伪装弹数
            missiles_per_aircraft: 单机最大挂载导弹数
            logger: 日志记录器
        """
        self.logr = logger or logging.getLogger(__name__)

        def __plns_maker():
            plns: dict[int, BasePlane] = {}
            return plns

        def __msls_maker():
            msls: dict[int, BaseMissile] = {}
            return msls

        def __decs_maker():
            decs: dict[int, BaseDecoy] = {}
            return decs

        def __acs_maker():
            plns: dict[int, BasePlane] = {}
            msls: dict[int, List[BaseMissile]] = defaultdict(list)
            decoys: dict[int, List[BaseDecoy]] = defaultdict(list)
            return plns, msls, decoys

        self.fri_color = fri_color
        self.planes = __plns_maker()
        self.fri_missiles = __msls_maker()
        self.fri_decoys = __decs_maker()
        self.enm_missiles = __msls_maker()

        self.enm_color = enm_color
        self.enm_planes, self.enm_missiles, self.enm_decoys = __acs_maker()

        self.history: List[Dict] = []  # 历史状态记录
        self.max_history_size = 1000

        cfg = deepcopy(config)

        self._init_agents(cfg)

    def clean_planes(self):
        for g in [
            self.planes,
            self.fri_missiles,
            self.enm_missiles,
            self.fri_decoys,
        ]:
            g.clear()

    def _init_agents(self, cfg: dict):
        """初始化飞机和导弹"""
        pinfos: list[dict[str, Any]] = cfg.get("planes", [])
        for pinfo in pinfos:
            uid = pinfo["acmi_id"]
            name = pinfo.get("acmi_name", "J-20")
            color = pinfo.get("acmi_color", "")
            call_sign = pinfo.get("call_sign", "")

            pln = PlaneP6DOF(
                acmi_id=uid, acmi_name=name, acmi_color=color, call_sign=call_sign
            )
            p_id = pln.acmi_id[0, 0]

            # 创建导弹对象
            n_msl = pinfo.get("missile_limit", 0)
            if n_msl > 0:
                assert n_msl <= self.CAP_MISSILE, (
                    f"expected missile_limit <= {self.CAP_MISSILE}, got",
                    n_msl,
                )
                msls = MissileP6DOF(
                    group_shape=(n_msl,), acmi_color=color, acmi_parent=p_id
                )
                for i in range(n_msl):
                    m_id = p_id * 100 + i
                    msls.acmi_id[i, 0] = m_id
                self.fri_missiles[p_id] = msls

            # 创建诱饵对象
            n_deco = pinfo.get("decoy_limit", 0)
            if n_deco > 0:
                assert n_deco <= self.CAP_DECOY, (
                    f"expected decoy_limit <= {self.CAP_DECOY}, got",
                    n_deco,
                )
                decoys = DecoyDOF0Ball(
                    group_shape=(n_deco,), acmi_color=color, acmi_parent=p_id
                )
                for i in range(n_deco):
                    d_id = p_id * 10000 + i
                    decoys.acmi_id[i, 0] = d_id
                self.fri_decoys[p_id] = decoys

        minfos: list[dict[str, Any]] = cfg.get("missiles", [])
        for minfo in minfos:
            uid = minfo["acmi_id"]
            name = minfo.get("acmi_name", "PL-12")
            color = minfo.get("acmi_color", "Gray")

            msl = MissileP6DOF(acmi_id=uid, acmi_name=name, acmi_color=color)

        # 遍历飞机数量
        for i in range(num_aircraft):
            # 创建飞机对象
            pln = PlaneP6DOF()
            # 将飞机对象添加到飞机列表中
            self.planes.append(pln)
            # 创建导弹列表，并添加到导弹字典中
            self.enm_missiles[i] = [MissileP6DOF() for _ in range(missiles_per)]

    def step(self) -> Tuple[Dict, Dict, Dict]:
        """
        执行一步仿真

        Returns:
            tuple: (aircraft_states, missile_states, hit_events)
        """
        # 1. 运动状态更新
        for plns in [self.planes]:
            for p_id, pln in plns.items():
                pln.run(None)
                for msl in self.enm_missiles[p_id]:
                    msl.run(None)
                for decoy in self.enm_decoys[p_id]:
                    decoy.run(None)

        # 2.

        pln_states = {}
        for i, pln in enumerate(self.planes):
            pln.run(None)
            pln_states[i] = self._get_aircraft_state(pln)

        # 2. 更新所有导弹状态
        missile_states = {}
        hit_events = {}

        for pln_id, missiles in self.enm_missiles.items():
            for j, missile in enumerate(missiles):
                if missile.is_launched():
                    # 更新导弹目标信息
                    target_pln = self.planes[missile.target_id]
                    missile.observe(
                        pos_e=target_pln.position_e(),
                        vel_e=target_pln.velocity_e(),
                        mask=np.array([True]),
                    )

                    # 更新导弹状态
                    missile.update()
                    missile.try_hit()
                    missile.try_miss()

                    # 记录命中事件
                    if missile.is_hit():
                        hit_events[(pln_id, j)] = {
                            "target_id": missile.target_id,
                            "position": missile.position_e(),
                        }
                        missile.deactivate()  # 命中后导弹失效

                    missile_states[(pln_id, j)] = self._get_missile_state(missile)

        # 3. 处理飞机碰撞检测
        self._check_collisions()

        # 4. 清理已消亡导弹
        self._cleanup_missiles()

        # 记录关键状态
        self._log_system_status(hit_events)

        # 保存历史状态
        self._save_history(pln_states, missile_states, hit_events)

        return pln_states, missile_states, hit_events

    def _save_history(self, pln_states: Dict, missile_states: Dict, hit_events: Dict):
        """保存当前状态到历史记录"""
        snapshot = {
            "timestamp": len(self.history),
            "aircraft": pln_states,
            "missiles": missile_states,
            "hits": hit_events,
        }
        self.history.append(snapshot)

        # 限制历史记录大小
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

    def get_history(self, steps: int = 10) -> List[Dict]:
        """获取最近的历史状态记录"""
        return self.history[-steps:] if self.history else []

    def _log_system_status(self, hit_events: Dict):
        """记录系统关键状态"""
        active_missiles = sum(
            len([m for m in missiles if m.is_launched()])
            for missiles in self.enm_missiles.values()
        )
        alive_aircraft = sum(1 for pln in self.planes if pln.health_point > 0)

        print(
            f"[Status] Aircraft: {alive_aircraft}/{len(self.planes)} | "
            f"Active Missiles: {active_missiles} | "
            f"Hits: {len(hit_events)}"
        )

    def _check_collisions(self):
        """检测飞机间碰撞"""
        for i, ac1 in enumerate(self.planes):
            for j, ac2 in enumerate(self.planes[i + 1 :], i + 1):
                dist = np.linalg.norm(ac1.position_e() - ac2.position_e())
                if dist < 50:  # 碰撞阈值(米)
                    ac1.health_point -= 50
                    ac2.health_point -= 50

    def _get_aircraft_state(self, aircraft: BasePlane) -> Dict:
        """获取飞机状态字典（包含视野相关信息）"""
        visible_objects = {
            "missiles": self._get_visible_missiles(aircraft),
            "aircraft": self._get_visible_aircraft(aircraft),
        }
        return {
            "position": aircraft.position_e(),
            "velocity": aircraft.velocity_e(),
            "health": aircraft.health_point,
            "visible_objects": visible_objects,
        }

    def _get_visible_missiles(self, observer: BasePlane) -> List[Dict]:
        """获取观察者可见的导弹列表"""
        visible = []
        for pln_id, missiles in self.enm_missiles.items():
            for missile in missiles:
                if (
                    missile.is_launched()
                    and observer.is_in_fov(missile.position_e())[0]
                ):
                    visible.append(
                        {
                            "position": missile.position_e(),
                            "velocity": missile.velocity_e(),
                            "owner_id": pln_id,
                        }
                    )
        return visible

    def _get_visible_aircraft(self, observer: BasePlane) -> List[Dict]:
        """获取观察者可见的飞机列表"""
        visible = []
        for pln_id, aircraft in enumerate(self.planes):
            if aircraft != observer and observer.is_in_fov(aircraft.position_e())[0]:
                visible.append(
                    {
                        "position": aircraft.position_e(),
                        "velocity": aircraft.velocity_e(),
                        "id": pln_id,
                    }
                )
        return visible

    def _get_missile_state(self, missile: BaseMissile) -> Dict:
        """获取导弹状态字典（包含燃料和生命周期信息）"""
        return {
            "position": missile.position_e(),
            "velocity": missile.velocity_e(),
            "target_id": missile.target_id,
            "is_active": missile.is_launched(),
            "fuel_remaining": max(0, missile.max_flight_time - missile.sim_time_s),
            "status": self._get_missile_status(missile),
            "in_fov": self._check_missile_visibility(missile),
        }

    def _check_missile_visibility(self, missile: BaseMissile) -> Dict:
        """检查导弹对各飞行器的可见性"""
        visibility = {}
        for pln_id, aircraft in enumerate(self.planes):
            visible = aircraft.is_in_fov(missile.position_e())
            visibility[pln_id] = (
                bool(visible[0]) if visible.size == 1 else visible.tolist()
            )
        return visibility

    def _get_missile_status(self, missile: BaseMissile) -> str:
        """获取导弹状态描述"""
        if missile.is_hit():
            return "hit"
        elif missile.is_missed():
            return "missed"
        elif missile.sim_time_s > missile.max_flight_time:
            return "fuel_exhausted"
        elif missile.is_launched():
            return "active"
        return "inactive"

    def get_system_status(self) -> Dict:
        """获取系统全局状态快照"""
        return {
            "aircraft": [self._get_aircraft_state(pln) for pln in self.planes],
            "missiles": {
                pln_id: [self._get_missile_state(m) for m in missiles]
                for pln_id, missiles in self.enm_missiles.items()
            },
        }

    def launch_missile(
        self, aircraft_id: int, missile_idx: int, target_id: int
    ) -> bool:
        """
        发射指定导弹
        Returns:
            bool: 是否发射成功
        """
        # 参数校验
        if aircraft_id not in self.enm_missiles or missile_idx >= len(
            self.enm_missiles[aircraft_id]
        ):
            return False

        missile = self.enm_missiles[aircraft_id][missile_idx]
        if missile.is_launched() or target_id >= len(self.planes):
            return False

        try:
            # 设置导弹初始状态
            missile.reset()
            missile.target_id = target_id
            missile.set_target_info(
                pos_e=self.planes[target_id].position_e(),
                vel_e=self.planes[target_id].velocity_e(),
            )

            # 执行发射
            missile.launch()
            return True
        except Exception as e:
            self.logr.error(f"Missile launch failed: {str(e)}")
            return False

    def _cleanup_missiles(self):
        """清理已消亡的导弹（命中、脱靶、燃料耗尽）"""
        for pln_id in list(self.enm_missiles.keys()):
            active_missiles = []
            for missile in self.enm_missiles[pln_id]:
                if not missile.is_launched():
                    active_missiles.append(missile)
                    continue

                # 检查导弹状态
                if missile.is_hit():
                    self._handle_hit_effect(missile)
                elif missile.is_missed():
                    self._handle_miss_effect(missile)
                elif missile.sim_time_s > missile.max_flight_time:
                    missile.set_result(missile.RESULT_MISSED)
                    self._handle_miss_effect(missile)
                else:
                    active_missiles.append(missile)

            self.enm_missiles[pln_id] = active_missiles

    def _handle_miss_effect(self, missile: BaseMissile):
        """处理脱靶效果"""
        missile.deactivate()
        if self.logr:
            self.logr.info(
                f"Missile {id(missile)} missed target after {missile.sim_time_s:.1f}s"
            )

    def _handle_hit_effect(self, missile: BaseMissile):
        """处理命中效果"""
        target = self.planes[missile.target_id]
        target.health_point -= missile.demage
        if target.health_point <= 0:
            target.set_termstate(target.TERMSTATE_SHOTDOWN)

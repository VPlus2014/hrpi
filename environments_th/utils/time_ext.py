import math
import time
from datetime import datetime
import contextlib


def datetime2str(now: datetime | None = None, fmt: str = "%Y%m%d_%H%M%S"):
    if now is None:
        now = datetime.now()
    return now.strftime(fmt)


def sec2hmr(sec: float) -> str:
    m, s = divmod(math.ceil(sec), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    ss = ""
    if d > 0:
        ss += f"{int(d)}d "
    ss += f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return ss


class Timer_Context(contextlib.ContextDecorator):
    """上下文计时器"""

    # Usage: @TimerL() decorator or 'with TimerL():' context manager
    def __init__(self, t=0.0):
        """上下文计时器"""
        self.t = t
        self.dt = 0.0
        self.__nstack = 0

    def __enter__(self):
        self.push()
        return self

    def __exit__(self, type, value, traceback):
        self.pop()

    def time(self):
        return time.time()

    def add_dt(self, dt: float):
        self.dt += dt  # delta-time
        self.t += dt  # accumulate dt

    def push(self):
        if self.__nstack == 0:
            self.start = self.time()
            self.dt = 0.0
        self.__nstack += 1

    def pop(self):
        assert self.__nstack > 0, "Timer stack underflow"
        self.__nstack -= 1
        if self.__nstack == 0:
            dt = self.time() - self.start  # delta-time
            self.add_dt(dt)


class Timer_Pulse:
    """脉冲信号计时器"""

    __k: int
    """所处周期数"""
    __t0: float
    """初始时刻"""
    __n: int
    """累计步进次数"""
    __fps: float
    """事件发生频率"""

    def __init__(self, period_sec: float = 1.0):
        """脉冲信号计时器"""
        self.reset(period_sec)

    def reset(self, period_sec: float = None):
        """发生脉冲信号的时刻为 t_k = t0 + k * period_sec for k = 0, 1, 2, 3,..."""
        if period_sec is None:
            period_sec = self._period
        self._period = period_sec = float(period_sec)
        assert period_sec > 0, ("period must be positive, but get", period_sec)
        self.__t0 = time.time()
        self.__k = -1  #
        #
        self.__t0_first = None
        self.__n = 0
        self.__fps = 0.0

    def period(self):
        return self._period

    def k(self):
        """最近一次步进时刻所属周期"""
        return self.__k

    def step(self, nevents=1):
        """检测从最近一次脉冲信号后，直到现在新出现了多少次脉冲信号(>=0)"""
        t0 = self.__t0
        t1 = time.time()
        k1 = int((t1 - t0) / self._period)
        dk = k1 - self.__k
        self.__k = k1
        self.__t = t1
        self.__n += nevents
        if self.__t0_first is None:
            self.__t0_first = t1
        else:
            self.__fps = self.__n / max(1e-9, t1 - self.__t0_first)
        return dk

    def t_next(self):
        """最近一次 step 后的距离下一个脉冲信号时刻"""
        return self.__t0 + (self.__k + 1) * self._period

    def t_to_next(self):
        """从现在起到下一个脉冲信号的剩余秒数(>=0)"""
        self.step()
        return max(0.0, self.t_next() - time.time())

    def fps(self):
        """事件发生频率"""
        return self.__fps

    def n_events(self):
        """事件计数"""
        return self.__n

# 250602 日志功能扩展
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from datetime import datetime
import logging
import time

import logging
from pathlib import Path
import sys

if TYPE_CHECKING:
    from typing import Union

try:
    import colorlog
except ImportError:
    colorlog = None


def setup_logging():
    """配置日志默认格式和级别"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                "log/{}.log".format(time.ctime().replace(":", "_"))
            ),  # 日志写入文件
            logging.StreamHandler(),  # 日志输出到控制台
        ],
    )


@dataclass
class LogConfig:
    """可传入进程的日志配置"""

    logger: str | None = None
    """名称, 默认为None, 即root logger"""
    level: int = logging.INFO
    format: str | None = None
    """行格式, 默认为 [%(levelname)s|%(asctime)s] %(message)s """
    date_format: str | None = None
    """时间格式, 默认为 %m-%d %H:%M:%S """
    file_path: str | None = None
    """输出文件路径, 默认为不输出"""
    file_append: bool = True
    """是否不覆盖写入, 默认为不覆盖"""


def reset_logger(
    logger: Union[logging.Logger, str, None] = None,
    level: int = logging.INFO,
    format: str | None = None,  # "[%(levelname)s|%(asctime)s] %(message)s"
    date_format: str | None = None,  # "%Y-%m-%d %H:%M:%S"
    file_path: str | None = None,
    file_append: bool = True,
    file_encoding: str = "utf-8",
    new_stdout: bool = False,
    new_stderr: bool = False,
):
    format = format or "[%(levelname)s|%(asctime)s] %(message)s"
    date_format = date_format or "%m-%d %H:%M:%S"

    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    logger.setLevel(level)
    # remove all handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        # create file handler
        file_handler = logging.FileHandler(
            file_path, mode="a" if file_append else "w", encoding=file_encoding
        )
        file_handler.setFormatter(logging.Formatter(format, date_format))
        logger.addHandler(file_handler)

    _meta = [
        (new_stdout, sys.stdout, level),
        (new_stderr, sys.stderr, logging.ERROR),
    ]
    for use, strm, lv in _meta:
        if not use:
            continue
        h = logging.StreamHandler(strm)
        if colorlog:
            fmtr = colorlog.ColoredFormatter("%(log_color)s" + format, date_format)
        else:
            fmtr = logging.Formatter(format, date_format)

        h.setFormatter(fmtr)
        h.setLevel(lv)
        logger.addHandler(h)
    return logger


def reset_root_logger(stdout_level: int = logging.INFO):
    reset_logger(
        logging.root,
        level=stdout_level,
        new_stdout=True,
        new_stderr=True,
    )


_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[2]
reset_root_logger(logging.INFO)
LOG_TIME = datetime.now()  # (同一进程共享)当前模块的加载时间
LOG_DIR = _ROOT / "tmp" / LOG_TIME.strftime("%Y%m%d_%H%M%S_%f") / "logs"
"""默认日志目录"""

logging.debug("滥用字典的人要吞一万根针")

if __name__ == "__main__":
    reset_root_logger(logging.DEBUG)
    logging.root.debug("hello world")
    logging.root.info("hello world")
    logging.root.warning("hello world")
    logging.root.error("hello world")
    logging.root.fatal("hello world")

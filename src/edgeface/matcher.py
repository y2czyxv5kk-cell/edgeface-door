"""比对决策：特征库余弦比对 + 双阈值判定 + 开门冷却。

判定规则（50~500 人规模建议值）：
  top1 相似度 >= sim_thresh（0.50）且 top1-top2 >= margin（0.10）
  两个条件同时满足才算识别成功，否则按陌生人处理。
"""
from dataclasses import dataclass
import json
import os
import time
from typing import Dict

import numpy as np


@dataclass
class MatchResult:
    """一次比对的结果。

    top1_idx:   特征库中最相似者的下标（库空时为 -1）
    top1_name:  最相似者的姓名
    top1_sim:   top1 余弦相似度
    top2_sim:   top2 余弦相似度（库只有 1 人时为 -1）
    recognized: 是否满足双阈值判定
    """
    top1_idx: int
    top1_name: str
    top1_sim: float
    top2_sim: float
    recognized: bool


class FaceMatcher:
    """特征库比对 + 双阈值决策。

    特征库：N×512 矩阵 + 下标到姓名映射。
    500 人规模一次比对就是一次矩阵乘，毫秒级，无需向量检索库。
    """

    def __init__(self, db_embeddings: np.ndarray, id_map: Dict[int, str],
                 sim_thresh: float = 0.50, margin: float = 0.10):
        """Args:
            db_embeddings: (N, 512) 特征库，每行一人（已 L2 归一化）
            id_map: 下标 → 姓名
            sim_thresh: 相似度下限
            margin: top1 与 top2 的最小差距
        """
        self.db = db_embeddings
        self.id_map = id_map
        self.sim_thresh = sim_thresh
        self.margin = margin

    @classmethod
    def from_db_dir(cls, db_dir: str, sim_thresh: float = 0.50,
                    margin: float = 0.10) -> "FaceMatcher":
        """从特征库目录加载（embeddings.npy + id_map.json，由 register 生成）。"""
        emb_path = os.path.join(db_dir, "embeddings.npy")
        map_path = os.path.join(db_dir, "id_map.json")
        if not os.path.isfile(emb_path) or not os.path.isfile(map_path):
            raise FileNotFoundError(
                f"特征库不完整：{db_dir}。请先运行 python tools/register.py"
            )
        db = np.load(emb_path)
        with open(map_path, encoding="utf-8") as f:
            id_map = {int(k): v for k, v in json.load(f).items()}
        return cls(db, id_map, sim_thresh=sim_thresh, margin=margin)

    def match(self, embedding: np.ndarray) -> MatchResult:
        """拿一个特征向量与全库比对。

        Args:
            embedding: (512,) L2 归一化特征
        Returns:
            MatchResult。特征均已归一化，点积即余弦相似度。
        """
        if len(self.db) == 0:
            return MatchResult(-1, "陌生人", -1.0, -1.0, False)

        # (N,512) @ (512,) = (N,) 全库相似度
        sims = self.db @ embedding

        order = np.argsort(-sims)
        top1_idx = int(order[0])
        top1_sim = float(sims[top1_idx])
        top2_sim = float(sims[order[1]]) if len(order) > 1 else -1.0

        recognized = (top1_sim >= self.sim_thresh
                      and (top1_sim - top2_sim) >= self.margin)

        return MatchResult(
            top1_idx=top1_idx,
            top1_name=self.id_map.get(top1_idx, f"id_{top1_idx}"),
            top1_sim=top1_sim, top2_sim=top2_sim,
            recognized=recognized,
        )


class DoorCooldown:
    """开门冷却：同一人两次开门的最小间隔。

    视频流每秒多帧，识别命中是持续状态，不做冷却会连续触发开门脉冲。
    """

    def __init__(self, cooldown_seconds: float = 5.0, clock=time.time):
        """Args:
            cooldown_seconds: 同一人两次开门的最小间隔（秒）
            clock: 时间源函数，测试时可注入假时钟
        """
        self.cooldown = cooldown_seconds
        self._clock = clock
        self._last_trigger: Dict[str, float] = {}

    def allow(self, name: str, now: float = None) -> bool:
        """判断该人现在是否允许触发开门。

        Returns:
            True = 允许（内部已记录本次触发时间）；False = 冷却期内
        """
        if now is None:
            now = self._clock()
        last = self._last_trigger.get(name)
        if last is not None and now - last < self.cooldown:
            return False
        self._last_trigger[name] = now
        return True

    def reset(self) -> None:
        """清空冷却记录。"""
        self._last_trigger.clear()

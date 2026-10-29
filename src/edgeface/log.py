"""识别日志：把识别事件写入 JSONL 文件，供大模型工具查询。

门禁主循环写，edgebrain 工具读。JSONL 每行一条记录：
{"ts": 时间戳, "name": 人名, "sim": 相似度, "recognized": 布尔}
"""
import json
import os
import time


def _default_log_path() -> str:
    """默认日志路径：<项目根>/assets/logs/recognition.jsonl"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "assets", "logs", "recognition.jsonl")


class RecognitionLog:
    """追加式识别日志。单写多读场景，JSONL 行级追加天然无锁。"""

    def __init__(self, path: str = None):
        self.path = path or _default_log_path()

    def append(self, name: str, sim: float, recognized: bool = True,
               ts: float = None) -> None:
        """追加一条记录。"""
        if ts is None:
            ts = time.time()
        record = {"ts": ts, "name": name,
                  "sim": round(float(sim), 3), "recognized": recognized}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recent(self, n: int = 10) -> list:
        """读最近 n 条记录（文件顺序，最新在末尾）。文件不存在返回 []。"""
        if not os.path.isfile(self.path):
            return []
        lines = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return lines[-n:]

    def today_summary(self) -> str:
        """今天的识别统计（按人聚合 + 陌生人计数），供工具返回给大模型。"""
        import datetime
        records = self.recent(2000)
        today = datetime.date.today()
        today_records = [
            r for r in records
            if datetime.date.fromtimestamp(r["ts"]) == today
        ]
        if not today_records:
            return "今天还没有任何识别记录。"

        counts = {}
        for r in today_records:
            name = r["name"]
            counts[name] = counts.get(name, 0) + 1
        parts = [f"{name} {n} 次" for name, n in counts.items()]
        return f"今天共 {len(today_records)} 人次识别：" + "，".join(parts) + "。"

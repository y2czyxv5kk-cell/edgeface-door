# -*- coding: utf-8 -*-
"""视频识别链路验证：不依赖摄像头，跑完整 检测→识别→比对→开门 流程。

用法：python diagnostics/video_recog_test.py <视频路径> [输出路径]
输出：带标注框的视频 + 控制台统计（识别到谁、开门几次、每帧耗时）。
"""
import os
import sys
import time
import warnings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import cv2  # noqa: E402

from edgeface import create_engine  # noqa: E402
from edgeface.pipeline import FacePipeline  # noqa: E402
from edgeface.matcher import FaceMatcher, DoorCooldown  # noqa: E402
from edgeface.door import FakeDoor  # noqa: E402
from edgeface.draw import draw_face_box  # noqa: E402
from edgeface.log import RecognitionLog  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_DB = os.path.join(_PROJECT_ROOT, "assets", "face_db")


def main():
    src = sys.argv[1]
    out = (sys.argv[2] if len(sys.argv) > 2
           else os.path.splitext(src)[0] + "_annotated.mp4")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"FAIL: 打不开视频 {src}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {w}x{h} @ {fps:.1f}fps, {n_total} 帧")

    matcher = FaceMatcher.from_db_dir(_DB)
    print(f"特征库: {len(matcher.db)} 人")
    cooldown = DoorCooldown(cooldown_seconds=5.0)
    engine = create_engine(input_size=640)
    pipe = FacePipeline(engine)
    door = FakeDoor()
    log = RecognitionLog()

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"FAIL: 无法创建输出视频 {out}")
        sys.exit(1)

    faces_seen = {}          # 名字 → 帧数
    door_events = []         # (帧号, 名字, 相似度)
    t_total, n_proc = 0.0, 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1

            t0 = time.time()
            results = pipe.process(frame, max_num=5)
            t_total += time.time() - t0
            n_proc += 1

            for r in results:
                m = matcher.match(r.embedding)
                name = m.top1_name if m.recognized else "陌生人"
                faces_seen[name] = faces_seen.get(name, 0) + 1
                color = (0, 255, 0) if m.recognized else (0, 0, 255)
                label = (f"{m.top1_name} {m.top1_sim:.2f}" if m.recognized
                         else f"陌生人 {m.top1_sim:.2f}")
                frame = draw_face_box(frame, r.bbox, label, color_bgr=color)

                if m.recognized and cooldown.allow(m.top1_name):
                    log.append(m.top1_name, m.top1_sim, recognized=True)
                    door.trigger()
                    door_events.append((idx, m.top1_name, round(m.top1_sim, 3)))
                    print(f"[door] 第 {idx} 帧 开门：{m.top1_name} "
                          f"(sim={m.top1_sim:.3f})")

            writer.write(frame)
            if idx % 150 == 0:
                print(f"... {idx} 帧")
    finally:
        cap.release()
        writer.release()
        door.close()
        pipe.release()

    print("\n===== 验证结果 =====")
    print(f"处理帧数: {n_proc} / {n_total}")
    if n_proc:
        print(f"平均推理耗时: {t_total*1000/n_proc:.0f}ms/帧")
    print(f"检测到的人脸（按帧数）: {faces_seen or '无'}")
    print(f"开门事件 {len(door_events)} 次:")
    for idx_, name, sim in door_events:
        print(f"  帧 {idx_}: {name} sim={sim}")
    print(f"输出视频: {out}")
    ok = bool(faces_seen) and any(e[1] != "陌生人" for e in door_events)
    print("结论: 识别链路工作正常 ✅" if ok else "结论: 未识别到已注册人员 ⚠️")


if __name__ == "__main__":
    main()

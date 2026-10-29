"""实时门禁（CLI）：摄像头 → 推理 → 比对 → 开门。

画面实时显示：绿框=命中（名字+相似度），红框=陌生人。
无 gpiod 环境自动降级 FakeDoor。按 q 退出，退出时打印平均耗时。

用法：
    python tools/realtime.py
    python tools/realtime.py --src /dev/video9     # 板端指定设备
    python tools/realtime.py --src test.mp4 --no-gpio   # 调试：视频文件
"""
import argparse
import os
import platform
import sys
import time
import warnings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import cv2  # noqa: E402

from edgeface import create_engine  # noqa: E402
from edgeface.pipeline import FacePipeline  # noqa: E402
from edgeface.matcher import FaceMatcher, DoorCooldown  # noqa: E402
from edgeface.door import DoorController, FakeDoor  # noqa: E402
from edgeface.draw import draw_face_box  # noqa: E402
from edgeface.log import RecognitionLog  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "assets", "face_db")
_DEFAULT_CAM = 0


def _open_camera(src):
    """打开摄像头/视频源。Linux 用 V4L2；Windows 该常量也存在但后端不可用，走默认。"""
    if isinstance(src, str):
        return cv2.VideoCapture(src)
    if platform.system() == "Linux":
        return cv2.VideoCapture(src, cv2.CAP_V4L2)
    return cv2.VideoCapture(src)


def main():
    parser = argparse.ArgumentParser(description="实时人脸门禁（CLI）")
    parser.add_argument("--src", default=_DEFAULT_CAM, help="摄像头设备号或视频文件路径")
    parser.add_argument("--db", default=_DEFAULT_DB, help="特征库目录")
    parser.add_argument("--sim-thresh", type=float, default=0.50, help="相似度阈值")
    parser.add_argument("--margin", type=float, default=0.10, help="top1-top2 差距")
    parser.add_argument("--cooldown", type=float, default=5.0, help="开门冷却秒数")
    parser.add_argument("--det-size", type=int, default=640, help="检测尺寸")
    parser.add_argument("--gpio-pin", type=int, default=17, help="GPIO 引脚号")
    parser.add_argument("--no-gpio", action="store_true", help="强制 FakeDoor")
    parser.add_argument("--no-mjpg", action="store_true", help="不切换 MJPG 编码")
    args = parser.parse_args()

    matcher = FaceMatcher.from_db_dir(
        args.db, sim_thresh=args.sim_thresh, margin=args.margin)
    cooldown = DoorCooldown(cooldown_seconds=args.cooldown)
    print(f"特征库：{len(matcher.db)} 人")

    engine = create_engine(input_size=args.det_size)
    pipe = FacePipeline(engine)

    # 开门控制器：--no-gpio 或 gpiod 不可用时降级 FakeDoor
    # 安全：视频文件源是回放不是真人，禁止驱动真实门锁
    is_file_src = isinstance(args.src, str) and not args.src.isdigit()
    if args.no_gpio or is_file_src:
        door = FakeDoor()
        if is_file_src and not args.no_gpio:
            print("[warn] 视频文件源不驱动真实 GPIO（防回放），使用 FakeDoor 模拟")
    else:
        try:
            door = DoorController(pin=args.gpio_pin)
            print(f"GPIO 开门就绪：pin={args.gpio_pin}")
        except Exception as e:
            print(f"[warn] GPIO 不可用（{e}），降级为 FakeDoor 模拟")
            door = FakeDoor()

    src = args.src
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = _open_camera(src)
    if not cap.isOpened():
        print(f"无法打开视频源：{src}")
        sys.exit(1)
    # USB 摄像头切 MJPG 提速（YUYV 未压缩帧大）；仅对设备号生效
    if not args.no_mjpg and isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print("\n实时识别中… 绿框=命中开门，红框=陌生人。按 q 退出\n")

    log = RecognitionLog()
    last_stranger_log = 0.0
    t_total, n_frames = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("视频源结束/掉线")
            break

        t0 = time.time()
        results = pipe.process(frame, max_num=5)  # 每帧最多 5 张脸
        t_total += time.time() - t0
        n_frames += 1
        now = time.time()

        for r in results:
            m = matcher.match(r.embedding)
            color = (0, 255, 0) if m.recognized else (0, 0, 255)
            label = (f"{m.top1_name} {m.top1_sim:.2f}" if m.recognized
                     else f"陌生人 {m.top1_sim:.2f}")
            frame = draw_face_box(frame, r.bbox, label, color_bgr=color)

            if m.recognized and cooldown.allow(m.top1_name):
                log.append(m.top1_name, m.top1_sim, recognized=True)
                door.trigger()
                print(f"[door] 开门：{m.top1_name} (sim={m.top1_sim:.3f})")
            elif not m.recognized and now - last_stranger_log > 30:
                # 陌生人日志节流：30 秒最多记一条
                last_stranger_log = now
                log.append("陌生人", m.top1_sim, recognized=False)

        cv2.imshow("EdgeFace Gate", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if n_frames:
        print(f"\n平均每帧推理耗时：{t_total*1000/n_frames:.0f}ms "
              f"({n_frames} 帧)，约 {n_frames/t_total:.1f} FPS（不含开门阻塞）")
    cap.release()
    cv2.destroyAllWindows()
    door.close()
    pipe.release()


if __name__ == "__main__":
    main()

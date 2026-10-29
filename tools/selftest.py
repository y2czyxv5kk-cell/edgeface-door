"""环境自检：验证依赖、推理后端、特征库、真实推理、UI 加载。

用法：
    python tools/selftest.py
    python tools/selftest.py --with-ui   # 额外做 PyQt5 无头加载测试
"""
import argparse
import os
import sys
import warnings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PASS, _FAIL = "[OK]", "[FAIL]"


def _check(name: str, fn) -> bool:
    """跑一项检查并打印结果。fn 返回 (ok, detail)。"""
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, str(e)
    print(f"  {_PASS if ok else _FAIL} {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description="环境自检")
    parser.add_argument("--with-ui", action="store_true", help="UI 无头加载测试")
    args = parser.parse_args()

    print("=" * 50)
    print("EdgeFace 环境自检")
    print("=" * 50)
    all_ok = True

    print("\n[1] 核心依赖")
    all_ok &= _check("numpy", lambda: (True, __import__("numpy").__version__))
    all_ok &= _check("opencv", lambda: (True, __import__("cv2").__version__))
    all_ok &= _check("onnxruntime", lambda: (True, __import__("onnxruntime").__version__))
    all_ok &= _check("Pillow", lambda: (True, __import__("PIL").__version__))

    print("\n[2] 推理后端")
    from edgeface import BACKEND
    detail = {"rknn": "RKNN（板端 NPU）", "onnx": "ONNX（PC/WSL CPU）"}.get(BACKEND)
    if detail is None:
        detail = "无可用后端！PC 装 insightface，板端装 rknnlite-api"
    all_ok &= _check("后端探测", lambda: (BACKEND is not None, detail))

    print("\n[3] 特征库")
    db_dir = os.path.join(_PROJECT_ROOT, "assets", "face_db")

    def _db_check():
        emb = os.path.join(db_dir, "embeddings.npy")
        idm = os.path.join(db_dir, "id_map.json")
        if not (os.path.isfile(emb) and os.path.isfile(idm)):
            return False, f"缺特征库文件：{db_dir}（运行 tools/register.py 生成）"
        from edgeface.matcher import FaceMatcher
        m = FaceMatcher.from_db_dir(db_dir)
        return True, f"{len(m.db)} 人已注册"
    all_ok &= _check("特征库", _db_check)

    print("\n[4] 推理冒烟测试（内置测试图，检测→对齐→识别）")

    def _infer_check():
        import cv2
        from skimage import data
        from edgeface import create_engine
        from edgeface.pipeline import FacePipeline
        engine = create_engine()
        pipe = FacePipeline(engine)
        img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
        results = pipe.process(img)
        pipe.release()
        if not results:
            return False, "未检测到人脸"
        r = results[0]
        return True, (f"{len(results)} 张脸, score={r.det_score:.2f}, "
                      f"embedding={r.embedding.shape}, "
                      f"norm={float((r.embedding**2).sum()**0.5):.3f}")
    all_ok &= _check("流水线推理", _infer_check)

    if args.with_ui:
        print("\n[5] PyQt5 UI 加载（无头模式）")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        def _ui_check():
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance() or QApplication([])
            from ui.main_window import MainWindow
            from ui.login_window import LoginWindow
            from ui.face_worker import FaceWorker
            from ui.agent_worker import AgentWorker
            w = MainWindow("selftest")
            LoginWindow()
            return True, f"主窗口 {w.stack.count()} 页 + 登录窗 + 两个 worker 加载正常"
        all_ok &= _check("UI 模块", _ui_check)

    print("\n" + "=" * 50)
    if all_ok:
        print("全部通过 ✔  可以运行：python ui/main.py")
    else:
        print("存在失败项 ✘  请按上面的提示修复后重跑")
    print("=" * 50)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""开门执行：GPIO 控制继电器 → 电磁锁（含调试用 FakeDoor）。

电磁锁通电吸门、断电开门（fail-safe）。接线：
  RK3588 GPIO → 继电器模块（光耦隔离）→ 锁回路 NC 常闭端 → 电磁锁
平时继电器不动作（锁通电）；GPIO 拉高 → 继电器动作 → NC 断开 →
锁断电 → 开门。接线图详见 docs/。

安全注意：GPIO 为 3.3V/几 mA，严禁直接接锁（锁需 12V/1A+），
必须经继电器；现场必须保留物理开门方式。
"""
import time

try:
    import gpiod
except ImportError:
    gpiod = None  # 延迟到 DoorController 初始化时才报错

DEFAULT_CHIP = "gpiochip0"   # 用 `gpioinfo` 命令确认本板 chip 名
DEFAULT_PIN = 17             # 按板子物理引脚选，确认设备树未占用
DEFAULT_OPEN_PULSE = 0.8     # 开门脉冲宽度（秒）


class DoorController:
    """真实开门控制：GPIO 输出开门脉冲。

    door = DoorController(pin=17)
    door.trigger()    # 拉高 0.8s 后自动复位（阻塞）
    door.close()      # 退出前释放 GPIO

    兼容 libgpiod v1（Debian 11/Ubuntu 22.04，gpiod 1.6 API）与
    v2（Ubuntu 24.04，LineSettings/request_lines API）。
    """

    def __init__(self, chip: str = DEFAULT_CHIP, pin: int = DEFAULT_PIN,
                 active_high: bool = True, open_pulse: float = DEFAULT_OPEN_PULSE):
        """Args:
            chip: gpiochip 名称（v2 会补全为 /dev/gpiochipN）
            pin: GPIO 引脚号
            active_high: True=高电平触发；False=低电平触发
            open_pulse: 脉冲宽度（秒），结束自动复位回关门电平
        """
        if gpiod is None:
            raise ImportError(
                "找不到 gpiod 模块。安装：pip install gpiod 或 "
                "apt install python3-libgpiod"
            )
        self.active_high = active_high
        self.open_pulse = open_pulse
        self._pin = pin

        self._v2 = hasattr(gpiod, "LineSettings")
        # v2 的 Chip 需要设备路径（如 /dev/gpiochip0）；v1 接受裸名称
        path = chip
        if self._v2 and not chip.startswith("/"):
            path = f"/dev/{chip}"
        self._chip = gpiod.Chip(path)
        if self._v2:
            # v2：request_lines + LineSettings；set_value(line, value)
            settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT)
            self._req = self._chip.request_lines(
                consumer="face-door", config={pin: settings})
        else:
            # v1：get_line + request(type=OUT)；line.set_value(value)
            self._line = self._chip.get_line(pin)
            self._line.request(consumer="face-door",
                               type=gpiod.LINE_REQ_DIR_OUT)
        # 启动即置为关门电平
        self._set_value(self._off_value())

    def _on_value(self):
        """开门电平值。"""
        return 1 if self.active_high else 0

    def _off_value(self):
        """关门电平值。"""
        return 0 if self.active_high else 1

    def _set_value(self, value: int) -> None:
        if self._v2:
            self._req.set_value(self._pin, value)
        else:
            self._line.set_value(value)

    def trigger(self, duration: float = None):
        """执行一次开门脉冲：拉高 duration 秒后自动复位。

        本方法阻塞 duration 秒，实时循环中应放在识别线程执行。
        """
        if duration is None:
            duration = self.open_pulse
        self._set_value(self._on_value())
        time.sleep(duration)
        self._set_value(self._off_value())

    def close(self):
        """复位并释放 GPIO。进程退出前必须调用。"""
        try:
            self._set_value(self._off_value())
            if self._v2:
                self._req.release()
            else:
                self._line.release()
            self._chip.close()
        except Exception:
            pass


class FakeDoor:
    """调试用假门：不触发 GPIO，只打印日志。

    接口与 DoorController 一致；无 gpiod 环境自动降级使用。
    """

    def __init__(self, *args, **kwargs):
        print("[FakeDoor] 初始化（模拟模式，不真开门）")

    def trigger(self, duration: float = None):
        if duration is None:
            duration = DEFAULT_OPEN_PULSE
        print(f"[FakeDoor] 开门脉冲 ({duration:.1f}s) —— 板端此处将触发 GPIO 继电器")

    def close(self):
        pass

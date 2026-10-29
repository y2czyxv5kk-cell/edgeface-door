"""SCRFD 后处理：网络输出的 9 个张量 → 解码 → NMS。

SCRFD 输出 3 个尺度（stride 8/16/32）× 3 类张量：
  cls (N,1) 置信度、bbox (N,4) 距离预测、kps (N,10) 关键点偏移。
本模块按张量最后一维长度自动分类（不依赖输出顺序），解码后用 NMS 去重。
纯 numpy 实现，PC 与板端共用。
"""
from functools import lru_cache
import numpy as np

# 三个特征图下采样倍率：8 负责小脸，16 中脸，32 大脸
_STRIDES = (8, 16, 32)

# 每个 grid cell 有 2 个 anchor（物理位置相同）
_NUM_ANCHORS = 2


@lru_cache(maxsize=16)
def _generate_centers(input_size: int, stride: int) -> np.ndarray:
    """生成某 stride 下所有 anchor 中心坐标 (N, 2)。

    anchor 位置 = grid 顶点 (i*stride, j*stride)，与 InsightFace SCRFD
    参考实现一致（不加 stride/2 偏移）。与输入内容无关，lru_cache 只算一次。
    """
    h = w = input_size // stride
    ys = np.arange(h, dtype=np.float32) * stride  # (h,)
    xs = np.arange(w, dtype=np.float32) * stride  # (w,)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')  # (h, w)
    cell_centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (h*w, 2)
    return np.repeat(cell_centers, _NUM_ANCHORS, axis=0).astype(np.float32)


def _classify_outputs(outputs, input_size: int) -> dict:
    """把输出张量按最后一维长度分组。

    Returns:
        {"cls": {stride: (N,1)}, "bbox": {stride: (N,4)}, "kps": {stride: (N,10)}}
    """
    # N → stride 反查表：N = (input_size // stride)^2 * 2
    n_to_stride = {}
    for stride in _STRIDES:
        grid = input_size // stride
        n_to_stride[grid * grid * _NUM_ANCHORS] = stride

    grouped = {"cls": {}, "bbox": {}, "kps": {}}
    for out in outputs:
        arr = np.asarray(out)
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])
        n, k = arr.shape

        if k == 1:
            type_name = "cls"
        elif k == 4:
            type_name = "bbox"
        elif k == 10:
            type_name = "kps"
        else:
            continue

        stride = n_to_stride.get(n)
        if stride is None:
            # 兜底：由 N 反推 stride
            grid_sq = n // _NUM_ANCHORS
            grid = int(round(grid_sq ** 0.5))
            if grid * grid * _NUM_ANCHORS == n and input_size % grid == 0:
                stride = input_size // grid
        if stride not in _STRIDES:
            continue

        grouped[type_name].setdefault(stride, arr)

    return grouped


def _decode_stride(centers, bbox_pred, kps_pred, score_pred, stride, thresh):
    """单 stride 解码：距离预测 → 实际坐标，过滤低置信度候选。

    Returns:
        (boxes, kps, scores)，或 None（无候选通过阈值）
    """
    scores = score_pred.reshape(-1)
    keep = scores > thresh
    if not np.any(keep):
        return None
    centers = centers[keep]
    bbox_pred = bbox_pred.reshape(-1, 4)[keep]
    kps_pred = kps_pred.reshape(-1, 10)[keep]
    scores = scores[keep]

    # bbox 解码：预测值是距离/stride，乘回 stride 得像素距离
    x1 = centers[:, 0] - bbox_pred[:, 0] * stride
    y1 = centers[:, 1] - bbox_pred[:, 1] * stride
    x2 = centers[:, 0] + bbox_pred[:, 2] * stride
    y2 = centers[:, 1] + bbox_pred[:, 3] * stride
    boxes = np.stack([x1, y1, x2, y2], axis=1)  # (M, 4) xyxy

    # 关键点解码：偏移 × stride + anchor 中心。布局 [x0,y0,...,x4,y4]
    kps = np.empty((len(boxes), 5, 2), dtype=np.float32)
    kps[:, :, 0] = centers[:, 0:1] + kps_pred[:, 0::2] * stride
    kps[:, :, 1] = centers[:, 1:2] + kps_pred[:, 1::2] * stride

    return boxes, kps, scores


def _nms(boxes, scores, iou_thresh):
    """按置信度降序贪心保留框，IoU 超过阈值的丢弃。"""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.clip(union, 1e-12, None)
        order = order[1:][iou <= iou_thresh]
    return keep


def detect(outputs, input_size: int, det_thresh: float = 0.5,
           nms_iou: float = 0.4, max_num: int = 0, debug: bool = False,
           img_center_xy: tuple = None):
    """SCRFD 完整后处理入口。

    Args:
        outputs: 模型输出张量列表（顺序任意）
        input_size: 模型输入边长
        det_thresh: 置信度阈值
        nms_iou: NMS IoU 阈值
        max_num: NMS 后最多保留框数，0 不限
        debug: 打印每个张量的分类结果
        img_center_xy: 图像内容中心（模型输入坐标系），用于 max_num 截断时
            的居中偏好打分；None 则取输入平面中心

    Returns:
        [{"bbox": (4,) xyxy, "kps": (5,2), "score": float}, ...]
        坐标为模型输入坐标系，调用方需缩放回原图。
    """
    grouped = _classify_outputs(outputs, input_size)

    if debug:
        print("[postprocess debug] 输出张量分类：")
        for t in ("cls", "bbox", "kps"):
            for stride, arr in grouped[t].items():
                print(f"  {t} stride={stride}: shape={arr.shape}, "
                      f"min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")

    # 完整性校验：3 类 × 3 stride 必须齐全
    for t in ("cls", "bbox", "kps"):
        if len(grouped[t]) != len(_STRIDES):
            raise ValueError(
                f"输出分类不完整：{t} 只有 stride {sorted(grouped[t].keys())}，"
                f"期望 {list(_STRIDES)}。用 debug=True 检查。"
            )

    all_boxes, all_kps, all_scores = [], [], []
    for stride in _STRIDES:
        centers = _generate_centers(input_size, stride)
        decoded = _decode_stride(
            centers,
            grouped["bbox"][stride],
            grouped["kps"][stride],
            grouped["cls"][stride],
            stride, det_thresh,
        )
        if decoded is None:
            continue
        all_boxes.append(decoded[0])
        all_kps.append(decoded[1])
        all_scores.append(decoded[2])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    kps = np.concatenate(all_kps, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    keep = _nms(boxes, scores, nms_iou)
    boxes, kps, scores = boxes[keep], kps[keep], scores[keep]

    # max_num 截断：与 InsightFace 参考实现相同的打分
    # 面积 - 2×到图像中心的距离²（偏好居中大框），再按打分降序取前 max_num
    if max_num > 0 and len(boxes) > max_num:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        center = img_center_xy or (input_size / 2, input_size / 2)
        offset_sq = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
        order = np.argsort(area - offset_sq * 2)[::-1][:max_num]
    else:
        order = scores.argsort()[::-1]
    boxes, kps, scores = boxes[order], kps[order], scores[order]

    results = []
    for i in range(len(boxes)):
        results.append({
            "bbox": boxes[i].astype(np.float32),
            "kps": kps[i].astype(np.float32).reshape(5, 2),
            "score": float(scores[i]),
        })
    return results

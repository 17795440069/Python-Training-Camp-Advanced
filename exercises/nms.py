# exercises/nms.py
"""
练习：非极大值抑制 (Non-Maximum Suppression, NMS)

描述：
实现目标检测中常用的 NMS 算法，用于去除重叠度高的冗余边界框。

请补全下面的函数 `calculate_iou` 和 `nms`。
"""
import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比 (IoU)。
    边界框格式：[x_min, y_min, x_max, y_max]

    Args:
        box1 (np.array): 第一个边界框 [x1_min, y1_min, x1_max, y1_max]。
        box2 (np.array): 第二个边界框 [x2_min, y2_min, x2_max, y2_max]。

    Return:
        float: IoU 值。
    """
    # 请在此处编写代码
    # (与 iou.py 中的练习相同，可以复用代码或导入)
    # 提示：计算交集面积和并集面积，然后相除。

    # 获取两个矩形坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]

    # 计算box1和box2的面积
    s1 = (x1max - x1min + 1.0) * (y1max - y1min + 1.0)
    s2 = (x2max - x2min + 1.0) * (y2max - y2min + 1.0)

    # 计算两相交矩形坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)

    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)

    # 计算相交矩阵长宽和面积
    inter_h = np.maximum(xmax - xmin + 1.0, 0)
    inter_w = np.maximum(ymax - ymax + 1.0, 0)

    inter_area = inter_h * inter_w

    # 计算相并面积
    s_union = s1 + s2 - inter_area

    # 计算交并比
    iou = inter_area / s_union
    return iou
    pass

def nms(boxes, scores, iou_threshold):
    """
    执行非极大值抑制 (NMS)。

    Args:
        boxes (np.array): 边界框数组，形状 (N, 4)，格式 [x_min, y_min, x_max, y_max]。
        scores (np.array): 每个边界框对应的置信度分数，形状 (N,)。
        iou_threshold (float): IoU 阈值，用于判断是否抑制。

    Return:
        list: 保留下来（未被抑制）的边界框的索引列表。
    """
    # 请在此处编写代码
    # 提示：
    # 1. 如果 boxes 为空，直接返回空列表。
    # 2. 将 boxes 和 scores 转换为 NumPy 数组。
    # 3. 计算所有边界框的面积 areas。
    # 4. 根据 scores 对边界框索引进行降序排序 (order = np.argsort(scores)[::-1])。
    # 5. 初始化一个空列表 keep 用于存储保留的索引。
    # 6. 当 order 列表不为空时循环：
    #    a. 取出 order 中的第一个索引 i (当前分数最高的框)，加入 keep。
    #    b. 计算框 i 与 order 中剩余所有框的 IoU。
    #       (需要计算交集区域坐标 xx1, yy1, xx2, yy2 和交集面积 intersection)
    #    c. 找到 IoU 小于等于 iou_threshold 的索引 inds。
    #    d. 更新 order，只保留那些 IoU <= threshold 的框的索引 (order = order[inds + 1])。
    # 7. 返回 keep 列表。

    # 根据分数排序
    sorted_indices = np.argsort(scores)[::-1]

    keep = []
    while sorted_indices.size > 0:
        # 选择当前最高分的框
        idx = sorted_indices[0]
        keep.append(idx)

        # 计算当前框与其他所有框的IOU
        ious = np.array([calculate_iou(boxes[idx], boxes[i]) for i in sorted_indices[1:]])

        # 删除与当前框IOU大于与之的框
        remove_indices = np.where(ious > iou_threshold)[0] + 1
        sorted_indices = np.delete(sorted_indices, remove_indices)
        # 移除已经处理过的最高分框的索引
        sorted_indices = np.delete(sorted_indices, 0)
    return keep

    pass 
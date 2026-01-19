#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具类模块 - BoxInterpolator 帧间检测框补全器
"""

from typing import Dict, List, Optional


class BoxInterpolator:
    """
    帧间检测框补全器

    用于处理连续帧中偶发的漏检问题。当某一帧的检测数量明显少于前后帧时，
    通过IoU匹配找出漏检的框并进行补全。
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_box_count_diff: int = 1,
        interpolate_mode: str = "linear"
    ):
        """
        Args:
            iou_threshold: IoU阈值，用于判断两个框是否为同一目标
            min_box_count_diff: 最小框数量差异，超过此值才触发补全
            interpolate_mode: 补全模式
                - "linear": 线性插值，取前后帧框坐标的中点（默认）
                - "union": 并集模式，取前后帧框的最小外接矩形
        """
        self.iou_threshold = iou_threshold
        self.min_box_count_diff = min_box_count_diff
        self.interpolate_mode = interpolate_mode
        self.stats = {
            "frames_interpolated": 0,
            "boxes_added": 0
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

    @staticmethod
    def _polygon_to_bbox(polygon: list) -> tuple:
        """将四边形多边形转换为矩形边界框 (x1, y1, x2, y2)"""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _calculate_iou(box1: tuple, box2: tuple) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def _interpolate_box(box1: list, box2: list, alpha: float = 0.5) -> list:
        """
        在两个框之间进行线性插值

        Args:
            box1: 第一个框的四个顶点
            box2: 第二个框的四个顶点
            alpha: 插值系数，0.5表示取中间值

        Returns:
            插值后的框
        """
        result = []
        for p1, p2 in zip(box1, box2):
            x = p1[0] * (1 - alpha) + p2[0] * alpha
            y = p1[1] * (1 - alpha) + p2[1] * alpha
            result.append([x, y])
        return result

    @staticmethod
    def _union_box(box1: list, box2: list) -> list:
        """
        计算两个框的并集（最小外接矩形）

        Args:
            box1: 第一个框的四个顶点
            box2: 第二个框的四个顶点

        Returns:
            包含两个框的最小外接矩形（四个顶点）
        """
        # 获取所有顶点的坐标
        all_xs = [p[0] for p in box1] + [p[0] for p in box2]
        all_ys = [p[1] for p in box1] + [p[1] for p in box2]

        # 计算最小外接矩形
        min_x, max_x = min(all_xs), max(all_xs)
        min_y, max_y = min(all_ys), max(all_ys)

        # 返回四个顶点（左上、右上、右下、左下）
        return [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]

    def _generate_interpolated_box(self, box1: list, box2: list) -> list:
        """
        根据 interpolate_mode 生成补全框

        Args:
            box1: 前一帧的框
            box2: 后一帧的框

        Returns:
            补全后的框
        """
        if self.interpolate_mode == "union":
            return self._union_box(box1, box2)
        else:  # "linear" 或其他
            return self._interpolate_box(box1, box2, 0.5)

    def _find_matching_box(self, target_box: list, candidates: list) -> Optional[int]:
        """
        在候选框列表中找到与目标框匹配的框

        Args:
            target_box: 目标框
            candidates: 候选框列表

        Returns:
            匹配框的索引，如果没找到则返回 None
        """
        target_bbox = self._polygon_to_bbox(target_box)
        best_iou = 0.0
        best_idx = None

        for idx, cand_box in enumerate(candidates):
            cand_bbox = self._polygon_to_bbox(cand_box)
            iou = self._calculate_iou(target_bbox, cand_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= self.iou_threshold:
            return best_idx
        return None

    def interpolate_frame(
        self,
        prev_boxes: list,
        curr_boxes: list,
        next_boxes: list,
        prev_detections: list = None,
        curr_detections: list = None,
        next_detections: list = None
    ) -> tuple:
        """
        对当前帧进行补全

        通过比较前后帧，找出当前帧可能漏检的框并补全。

        Args:
            prev_boxes: 前一帧的检测框列表
            curr_boxes: 当前帧的检测框列表
            next_boxes: 后一帧的检测框列表
            prev_detections: 前一帧的检测结果（可选）
            curr_detections: 当前帧的检测结果（可选）
            next_detections: 后一帧的检测结果（可选）

        Returns:
            (补全后的boxes, 补全后的detections, 是否有补全)
        """
        if not prev_boxes or not next_boxes:
            return curr_boxes, curr_detections, False

        # 检查是否需要补全：当前帧的框数量少于前后帧
        prev_count = len(prev_boxes)
        curr_count = len(curr_boxes)
        next_count = len(next_boxes)

        # 如果当前帧的数量不是明显少于前后帧，不需要补全
        min_neighbor = min(prev_count, next_count)
        if curr_count >= min_neighbor - self.min_box_count_diff + 1:
            return curr_boxes, curr_detections, False

        # 找出在前后帧都存在，但当前帧缺失的框
        added_boxes = []
        added_detections = []

        for prev_idx, prev_box in enumerate(prev_boxes):
            # 检查这个框在当前帧是否存在
            curr_match = self._find_matching_box(prev_box, curr_boxes)
            if curr_match is not None:
                continue  # 当前帧已有匹配，不需要补全

            # 检查这个框在后一帧是否存在
            next_match = self._find_matching_box(prev_box, next_boxes)
            if next_match is None:
                continue  # 后一帧也没有，可能是真的消失了

            # 前后帧都有，但当前帧没有 -> 需要补全
            # 根据模式生成补全框
            next_box = next_boxes[next_match]
            interpolated_box = self._generate_interpolated_box(prev_box, next_box)

            added_boxes.append(interpolated_box)

            # 构造补全的检测结果
            if prev_detections and next_detections:
                prev_det = prev_detections[prev_idx] if prev_idx < len(prev_detections) else None
                next_det = next_detections[next_match] if next_match < len(next_detections) else None

                # 使用前一帧的文本和置信度（取较低的置信度）
                if prev_det and next_det:
                    confidence = min(prev_det.get("confidence", 0.5), next_det.get("confidence", 0.5))
                    text = prev_det.get("text", "")
                elif prev_det:
                    confidence = prev_det.get("confidence", 0.5)
                    text = prev_det.get("text", "")
                else:
                    confidence = 0.5
                    text = ""

                added_detections.append({
                    "bbox": interpolated_box,
                    "text": text,
                    "confidence": confidence,
                    "interpolated": True,
                    "interpolate_mode": self.interpolate_mode
                })

        if not added_boxes:
            return curr_boxes, curr_detections, False

        # 合并结果
        new_boxes = list(curr_boxes) + added_boxes
        new_detections = list(curr_detections) if curr_detections else []
        new_detections.extend(added_detections)

        # 更新统计
        self.stats["frames_interpolated"] += 1
        self.stats["boxes_added"] += len(added_boxes)

        return new_boxes, new_detections, True

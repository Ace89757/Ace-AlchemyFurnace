# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import numpy as np
import os.path as osp
from typing import Dict, List, Optional, Union

from mmengine.fileio import load
from mmengine.logging import MMLogger

from alchemy.registry import METRICS
from mmdet3d.evaluation import do_eval
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


F = 3


def alchemy_kitti_eval(gt_annos, dt_annos, current_classes, eval_types=['bbox', 'bev', '3d']):
    """
    KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval. Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, 'must contain at least one evaluation type'

    if 'aos' in eval_types:
        assert 'bbox' in eval_types, 'must evaluate bbox when evaluating aos'

    # 不同类别在easy, moderate, hard下设置不同的阈值(车辆类别阈值设置的较高, 人相关的类别比较难预测, 阈值设置的较低)
    overlap_0_7 = np.array([
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],     # iou@.7: easy
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],     # iou@.7: moderate
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]      # iou@.7: hard
        ])

    overlap_0_5 = np.array([
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.5],     # iou@.5: easy
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],  # iou@.5: moderate
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]   # iou@.5: hard
        ])

    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 6]

    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }

    # 将需要评估的类别转换成数字
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]

    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)

    # 根据实际评估的类别选择iou阈值
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]

    result = ''

    # check whether alpha is valid, 决定是否计算AOS
    compute_aos = False
    pred_alpha = False
    valid_alpha_gt = False

    for anno in dt_annos:
        mask = (anno['alpha'] != -10)
        if anno['alpha'][mask].shape[0] != 0:
            pred_alpha = True
            break

    for anno in gt_annos:
        if anno['alpha'][0] != -10:
            valid_alpha_gt = True
            break

    compute_aos = (pred_alpha and valid_alpha_gt)
    if compute_aos:
        eval_types.append('aos')

    # 计算AP11 和 AP40指标
    mAP11_bbox, mAP11_bev, mAP11_3d, mAP11_aos, \
    mAP40_bbox, mAP40_bev, mAP40_3d, mAP40_aos = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, eval_types)

    # ----------- AP11 Results ------------
    ret_dict = {}
    difficulty = ['easy', 'moderate', 'hard']

    line_format = '{: <12}| {: ^15} | {: ^15} | {: ^15}\n'

    # Calculate AP40
    result += '\n----------------------- AP40 Results -----------------------\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]

        result += '{:*^62}\n'.format(f' {curcls_name} (AP R40) ')

        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += (line_format.format('IoU', *[f"@{round(x, 2)} ({eval_types[idx]})" for idx, x in enumerate(min_overlaps[i, :, j])]))  # easy moderate hard
            result += '{:=^62}\n'.format('=')
            result += line_format.format('', *difficulty)
            result += '{:-^62}\n'.format('-')

            if mAP40_bbox is not None:
                result += line_format.format('BBOX', *[round(x, F) for x in mAP40_bbox[j, :, i]])

            if mAP40_bev is not None:
                result += line_format.format('BEV', *[round(x, F) for x in mAP40_bev[j, :, i]])

            if mAP40_3d is not None:
                result += line_format.format('3D', *[round(x, F) for x in mAP40_3d[j, :, i]])

            if compute_aos and mAP40_aos is not None:
                result += line_format.format('AOS', *[round(x, F) for x in mAP40_aos[j, :, i]])

            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'

                if mAP40_3d is not None:
                    ret_dict[f'{curcls_name}_3D_AP40_{postfix}'] = mAP40_3d[j, idx, i]

                if mAP40_bev is not None:
                    ret_dict[f'{curcls_name}_BEV_AP40_{postfix}'] = mAP40_bev[j, idx, i]

                if mAP40_bbox is not None:
                    ret_dict[f'{curcls_name}_2D_AP40_{postfix}'] = mAP40_bbox[j, idx, i]
                
                if compute_aos and mAP40_aos is not None:
                    ret_dict[f'{curcls_name}_AOS_AP40_{postfix}'] = mAP40_aos[j, idx, i]

            result += '\n'

    # calculate mAP40 over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += '{:*^62}\n'.format(f' Overall (AP R40) ')
        result += (line_format.format('Difficulty', *difficulty))
        result += '{:-^62}\n'.format('-')

        if mAP40_bbox is not None:
            mAP40_bbox = mAP40_bbox.mean(axis=0)
            result += line_format.format('BBOX AP40', *[round(x, F) for x in mAP40_bbox[:, 0]])

        if mAP40_bev is not None:
            mAP40_bev = mAP40_bev.mean(axis=0)
            result += line_format.format('BEV AP40', *[round(x, F) for x in mAP40_bev[:, 0]])

        if mAP40_3d is not None:
            mAP40_3d = mAP40_3d.mean(axis=0)
            result += line_format.format('3D AP40', *[round(x, F) for x in mAP40_3d[:, 0]])

        if compute_aos:
            mAP40_aos = mAP40_aos.mean(axis=0)
            result += line_format.format('AOS AP40', *[round(x, F) for x in mAP40_aos[:, 0]])

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            if mAP40_3d is not None:
                ret_dict[f'Overall_3D_AP40_{postfix}'] = mAP40_3d[idx, 0]

            if mAP40_bev is not None:
                ret_dict[f'Overall_BEV_AP40_{postfix}'] = mAP40_bev[idx, 0]

            if mAP40_bbox is not None:
                ret_dict[f'Overall_2D_AP40_{postfix}'] = mAP40_bbox[idx, 0]
            
            if compute_aos and mAP11_aos is not None:
                ret_dict[f'Overall_AOS_AP40_{postfix}'] = mAP11_aos[idx, 0]

    return result, ret_dict


@METRICS.register_module()
class AlchemyMono3dMetric(KittiMetric):
    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'alchemy_kitti'
        super(AlchemyMono3dMetric, self).__init__(
            ann_file=ann_file,
            metric=metric,
            pcd_limit_range=pcd_limit_range,
            pklfile_prefix=pklfile_prefix,
            backend_args=backend_args,
            default_cam_key=default_cam_key,
            submission_prefix=submission_prefix,
            format_only=format_only,
            collect_device=collect_device, prefix=prefix)

        self.num = 0
        self.best_metrics = {}

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of the metrics, and the values are corresponding results.
        """
        self.num += 1

        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        # 评估时读取的gt-location不是cam_instance中的location, 是原始的location, 是底边中心点
        # cam-instance中的location是中心点
        self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)

        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            logger.info(f'results are saved in {osp.dirname(self.submission_prefix)}')
            return metric_dict

        gt_annos = [
            self.data_infos[result['sample_idx']]['kitti_annos']
            for result in results
        ]

        for metric in self.metrics:
            ap_dict = self.kitti_evaluate(
                result_dict,
                gt_annos,
                metric=metric,
                logger=logger,
                classes=self.classes)

            logger.info('------------------ Metrics ------------------')
            for metric_name in ap_dict:
                metric_dict[metric_name] = ap_dict[metric_name]

                # 记录最优结果
                if 'Overall' not in metric_name:
                    continue

                if metric_name not in self.best_metrics:
                    self.best_metrics[metric_name] = (self.num, ap_dict[metric_name])
                else:
                    if ap_dict[metric_name] > self.best_metrics[metric_name][1]:
                        self.best_metrics[metric_name] = (self.num, ap_dict[metric_name])
                
                current_str = '{: <40}'.format(f'{metric_name}: {round(ap_dict[metric_name], F)}')
                best_str = f'(best: {self.best_metrics[metric_name][1]} [{self.best_metrics[metric_name][0]}])'
                logger.info(f'{current_str}{best_str}')

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return metric_dict

    def kitti_evaluate(self, results_dict: dict, gt_annos: List[dict], metric: Optional[str] = None, classes: Optional[List[str]] = None, logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """
        Evaluation in KITTI protocol.

        Args:
            results_dict (dict): Formatted results of the dataset.
            gt_annos (List[dict]): Contain gt information of each sample.
            metric (str, optional): Metrics to be evaluated. Defaults to None.
            classes (List[str], optional): A list of class name. Defaults to None.
            logger (MMLogger, optional): Logger used for printing related information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        ap_dict = dict()
        for name in results_dict:
            if name == 'pred_instances' or metric == 'img_bbox':
                continue
            else:
                result_table, ap_dict_ = alchemy_kitti_eval(gt_annos, results_dict[name], classes, eval_types=['bbox', 'bev', '3d'])

                logger.info('----------------------- KITTI Resutls -----------------------')
                for line in result_table.split('\n'):
                    logger.info(line)
                
                for ap_type, ap in ap_dict_.items():
                    ap_dict[ap_type] = float(round(ap, F))

        return ap_dict






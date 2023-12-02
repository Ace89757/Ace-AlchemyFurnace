# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import math
import tempfile
import itertools
import numpy as np
import os.path as osp

from collections import OrderedDict
from terminaltables import AsciiTable
from typing import Dict, List, Optional, Sequence

from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmdet.datasets.api_wrappers import COCO, COCOeval

from alchemy.registry import METRICS


F = 3


@METRICS.register_module()
class AlchemyDet2dMetric(BaseMetric):
    """
    COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including bbo. 
    Please refer to https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file. Defaults to None.
        classwise (bool): Whether to evaluate the metric class-wise. Defaults to True.
        max_dets (Sequence[int]): Numbers of proposals to be evaluated. Defaults to (100, 300, 1000).
        format_only (bool): Format the output results without perform evaluation. 
                            It is useful when you want to format the result to a specific format and submit it to the test server. Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes the file path and the prefix of filename, e.g., "a/b/prefix". 
                                        If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from different ranks during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'gpu'.
        prefix (str, optional): The prefix that will be added in the metric names to disambiguate homonymous metrics of different evaluators.
                                If prefix is not provided in the argument, self.default_prefix  will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = None

    def __init__(self,
                 format_only: bool = False,
                 backend_args: dict = None,
                 collect_device: str = 'gpu',
                 prefix: Optional[str] = None,
                 ann_file: Optional[str] = None,
                 outfile_prefix: Optional[str] = None,
                 max_dets: Sequence[int] = (100, 300, 1000),
                 object_size: Sequence[float] = (32, 64, 1e5)) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not None when format_only is True, otherwise the result files will be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        
        # load dataset
        with get_local_path(ann_file, backend_args=self.backend_args) as local_path:
            self._coco_api = COCO(local_path)

        # max dets used to compute recall or precision.
        self.max_dets = list(max_dets)

        self.area_ranges = (
            [0 ** 2, object_size[2] ** 2], 
            [0 ** 2, object_size[0] ** 2], 
            [object_size[0] ** 2, object_size[1] ** 2], 
            [object_size[1] ** 2, object_size[2] ** 2]
            )

        # iou_thrs used to compute recall or precision.
        self.iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        
        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

        # 获取每个类别的类别数量
        self.cat_counts = {}
        total_anns = 0
        for cat_info in self._coco_api.dataset['categories']:
            cat_name = cat_info['name']
            cat_id = self._coco_api.get_cat_ids(cat_names=[cat_name])[0]
            img_ids = self._coco_api.getImgIds(catIds=cat_id)
            ann_ids = self._coco_api.getAnnIds(imgIds=img_ids, catIds=cat_id, iscrowd=None)
            self.cat_counts[cat_id] = len(ann_ids)
            total_anns += len(ann_ids)
        
        # 某些类别数量可能特别少，使用类别数量权重有些不太合理，这些数量少的目标基本没有贡献, 现对这些目标的权重做转换, 降低数量多目标的权重，增加数量少目标的权重
        class_ratios = {}
        for cat_id, num_cats in self.cat_counts.items():
            class_ratios[cat_id] = math.sqrt(num_cats) / math.sqrt(total_anns)
        
        new_total_anns = sum(list(class_ratios.values()))

        self.cat_ratios = {}
        for cat_id, cat_ratio in class_ratios.items():
            self.cat_ratios[cat_id] = cat_ratio / new_total_anns
        
        self.best_metrics = {}
        self.num = 0

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict], outfile_prefix: str) -> dict:
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']

            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        return result_files

    # TODO: data_batch is no longer needed, consider adjusting the parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and predictions. 
        The processed results should be stored in ``self.results``, 
        which will be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            # parse pred
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        self.num += 1  # 记录第几次评估
        logger: MMLogger = MMLogger.get_current_instance()

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix
        
        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta['classes'])
            
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(results, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        logger.info(f'Evaluating bbox...')

        # evaluate bbox
        if 'bbox' not in result_files:
            raise KeyError('bbox is not in results!!')

        try:
            predictions = load(result_files['bbox'])
            coco_dt = self._coco_api.loadRes(predictions)

        except IndexError:
            logger.error('The testing results of the whole dataset is empty')

            return eval_results

        coco_eval = COCOeval(self._coco_api, coco_dt, 'bbox')

        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.iouThrs = self.iou_thrs

        # 设置大、中、小目标的面积
        coco_eval.params.areaRng = self.area_ranges

        # 设置最多检测框的数量
        coco_eval.params.maxDets = list(self.max_dets)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Compute per-category AP from https://github.com/facebookresearch/detectron2/
        """
        iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
        recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
        areaRng    - [...] A=4 object area ranges for evaluation
        catIds     - [all] K cat ids to use for evaluation
        maxDets    - [1 10 100] M=3 thresholds on max detections per image
        """
        precisions = coco_eval.eval['precision']   # [TxRxKxAxM]
        recalls = coco_eval.eval['recall']        # [TxKxAxM]

        # precision: (iou, recall, cls, area range, max dets)
        assert len(self.cat_ids) == precisions.shape[2]

        results_per_category = []
        avg_metrics = {}
        for idx, cat_id in enumerate(self.cat_ids):
            t = []

            # 类别所占比例
            cat_ratio = self.cat_ratios[cat_id]

            # area range index 0: all area ranges max dets index -1: typically 100 per image
            cat_info = self._coco_api.loadCats(cat_id)[0]
            t.append(f'{cat_info["name"]}')
            t.append(round(cat_ratio, F))

            t.append(self.cat_counts[cat_id])

            # cls_ap
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]

            if precision.size:
                cls_ap = np.mean(precision)
            else:
                cls_ap = 0.

            t.append(round(cls_ap, F))
            eval_results[f'{cat_info["name"]}_AP'] = round(cls_ap, F)
            if 'mAP' not in avg_metrics:
                avg_metrics['mAP'] = cls_ap * cat_ratio
            else:
                avg_metrics['mAP'] += cls_ap * cat_ratio

            # cls_ar
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            if recall.size:
                cls_ar = np.mean(recall)
            else:
                cls_ar = 0.

            t.append(round(cls_ar, F))
            eval_results[f'{cat_info["name"]}_AR'] = round(cls_ar, F)
            if 'mAR' not in avg_metrics:
                avg_metrics['mAR'] = cls_ar * cat_ratio
            else:
                avg_metrics['mAR'] += cls_ar * cat_ratio

            # f1
            cls_f1 = 2 * (cls_ap * cls_ar) / (cls_ap + cls_ar + 1e-6)
            t.append(round(cls_f1, F))
            eval_results[f'{cat_info["name"]}_F1'] = round(cls_f1, F)
            if 'mF1' not in avg_metrics:
                avg_metrics['mF1'] = cls_f1 * cat_ratio
            else:
                avg_metrics['mF1'] += cls_f1 * cat_ratio
            
            # indexes of IoU  @50 and @75
            for iou in [0, 5]:
                # ap
                p_name = f'AP@{".5" if iou == 0 else ".75"}'

                precision = precisions[iou, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    cls_ap = np.mean(precision)
                else:
                    cls_ap = 0.

                t.append(round(cls_ap, F))
                eval_results[f'{cat_info["name"]}_{p_name}'] = round(cls_ap, F)
                if f'm{p_name}' not in avg_metrics:
                    avg_metrics[f'm{p_name}'] = cls_ap * cat_ratio
                else:
                    avg_metrics[f'm{p_name}'] += cls_ap * cat_ratio

                # ar
                r_name = f'R@{".5" if iou == 0 else ".75"}'

                cls_r = recalls[iou, idx, 0, -1]
                t.append(round(cls_r, F))
                if f'mA{r_name}' not in avg_metrics:
                    avg_metrics[f'mA{r_name}'] = cls_r * cat_ratio
                else:
                    avg_metrics[f'mA{r_name}'] += cls_r * cat_ratio

                eval_results[f'{cat_info["name"]}_{r_name}'] = round(cls_r, F)

            # indexes of area of small, median and large
            names = ['small', 'medium', 'large']
            for area in [1, 2, 3]:
                # ap
                p_name = f'AP@{names[area - 1]}'

                precision = precisions[:, :, idx, area, -1]
                precision = precision[precision > -1]
                if precision.size:
                    cls_ap = np.mean(precision)
                else:
                    cls_ap = 0.

                t.append(round(cls_ap, F))
                eval_results[f'{cat_info["name"]}_{p_name}'] = round(cls_ap, F)
                if f'm{p_name}' not in avg_metrics:
                    avg_metrics[f'm{p_name}'] = cls_ap * cat_ratio
                else:
                    avg_metrics[f'm{p_name}'] += cls_ap * cat_ratio

                # ar
                r_name = f'AR@{names[area - 1]}'
                recall = recalls[:, idx, area, -1]
                recall = recall[recall > -1]
                if recall.size:
                    cls_ar = round(np.mean(recall), F)
                else:
                    cls_ar = 0.

                t.append(round(cls_ar, F))
                eval_results[f'{cat_info["name"]}_{r_name}'] = round(cls_ar, F)
                if f'm{r_name}' not in avg_metrics:
                    avg_metrics[f'm{r_name}'] = cls_ar * cat_ratio
                else:
                    avg_metrics[f'm{r_name}'] += cls_ar * cat_ratio

            results_per_category.append(tuple(t))

        num_columns = len(results_per_category[0])
        results_flatten = list(itertools.chain(*results_per_category))
        headers = [
            'category', 
            'weight',
            'counts',
            'AP', 'AR', 'F1', 
            'AP@.5', 'R@.5', 
            'AP@.75', 'R@.75', 
            'AP@samll', 'AR@small', 
            'AP@medium', 'AR@medium', 
            'AP@large', 'AR@large'
            ]

        results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)

        logger.info('------------------ Class-wise Metric ------------------')
        for table_line in table.table.split('\n'):
            logger.info(table_line)

        # 平均指标
        logger.info('------------------ Average Metric ------------------')
        logger.info('====> alchemy:')
        # 不是直接平均, 根据类别数量多少, 来设置权重
        for metric_name, avg_value in avg_metrics.items():
            avg_value = round(avg_value, F)

            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = (self.num, avg_value)
            else:
                if avg_value > self.best_metrics[metric_name][1]:
                    self.best_metrics[metric_name] = (self.num, avg_value)
            
            eval_results[metric_name] = avg_value

            current_str = '{: <24}'.format(f'{metric_name}: {avg_value}')
            best_str = f'(best: {self.best_metrics[metric_name][1]} [{self.best_metrics[metric_name][0]}])'
            logger.info(f'{current_str}{best_str}')
        
        logger.info('====> coco:')
        coco_names = [
            'mAP', 'mAP@.5', 'mAP@.75', 
            'mAP@small', 'mAP@medium', 'mAP@large', 
            f'mAR@{self.max_dets[0]}', f'mAR@{self.max_dets[1]}', f'mAR@{self.max_dets[2]}', 
            'mAR@small', 'mAR@medium', 'mAR@large'
            ]

        for idx, name in enumerate(coco_names):
            coco_val = round(coco_eval.stats[idx], F)

            if f'(coco) {name}' not in self.best_metrics:
                self.best_metrics[f'(coco) {name}'] = (self.num, coco_val)
            else:
                if coco_val > self.best_metrics[f'(coco) {name}'][1]:
                    self.best_metrics[f'(coco) {name}'] = (self.num, coco_val)

            current_str = '{: <28}'.format(f'(coco) {name}: {coco_val}')
            best_str = f'(best: {self.best_metrics[f"(coco) {name}"][1]} [{self.best_metrics[f"(coco) {name}"][0]}])'
            logger.info(f'{current_str}{best_str}')

            eval_results[f'(coco) {name}'] = coco_val
        logger.info('----------------------------------------------------')

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

import json
import itertools
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def mcbench_eval(gt_json_path, dt_json_path, eval_type='instance'):
    weighted_ap50, accuracy = None, None

    # ========== Load GT and original predictions ==========
    coco = COCO(gt_json_path)
    prediction = coco.loadRes(dt_json_path)

    cocoEval = COCOeval(coco, prediction, "bbox")
    cocoEval.evaluate()

    # ========== Re-group category IDs for paired images based on optimal matching ==========
    image_more_gt = coco.getImgIds(catIds=coco.getCatIds('group 2'))
    for gt_img_id in image_more_gt:
        pair_gt_img_id = coco.loadImgs(gt_img_id)[0]['pair_image_id']

        # Get all GT categories for the image pair
        gt_cats_list = [ann['category_id'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=gt_img_id))]
        gt_cats_list.extend([ann['category_id'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=pair_gt_img_id))])
        gt_cats_list = list(set(gt_cats_list))

        # Get predicted category groups
        dt_groups_list = [ann['category_id'] for ann in prediction.loadAnns(prediction.getAnnIds(imgIds=gt_img_id))]
        dt_groups_list.extend(
            [ann['category_id'] for ann in prediction.loadAnns(prediction.getAnnIds(imgIds=pair_gt_img_id))])
        dt_groups_list = list(set(dt_groups_list))

        # Find the best permutation of GT categories to match predicted category groups
        all_permutations = list(itertools.permutations(gt_cats_list, len(dt_groups_list)))
        all_permu_iou_list = []

        for permu in all_permutations:
            total_iou = []
            for group_id, gt_cat in enumerate(permu, start=1):
                dt = cocoEval._dts.get((gt_img_id, group_id), [])
                gt = cocoEval._gts.get((gt_img_id, gt_cat), [])
                dt_pair = cocoEval._dts.get((pair_gt_img_id, group_id), [])
                gt_pair = cocoEval._gts.get((pair_gt_img_id, gt_cat), [])

                if len(dt) > 0:
                    g = [g['bbox'] for g in gt]
                    d = [d['bbox'] for d in dt]
                    iscrowd = [int(o['iscrowd']) for o in gt]
                    ious = maskUtils.iou(d, g, iscrowd)
                    if len(ious) > 0:
                        # find optimal matching using Hungarian algorithm
                        cost_matrix = -ious
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        for r, c in zip(row_ind, col_ind):
                            total_iou.append(ious[r, c])

                if len(dt_pair) > 0:
                    g_pair = [g_['bbox'] for g_ in gt_pair]
                    d_pair = [d_['bbox'] for d_ in dt_pair]
                    iscrowd_pair = [int(o['iscrowd']) for o in gt_pair]
                    ious_pair = maskUtils.iou(d_pair, g_pair, iscrowd_pair)
                    if len(ious_pair) > 0:
                        # find optimal matching using Hungarian algorithm
                        cost_matrix_pair = -ious_pair
                        row_ind_pair, col_ind_pair = linear_sum_assignment(cost_matrix_pair)
                        for r, c in zip(row_ind_pair, col_ind_pair):
                            total_iou.append(ious_pair[r, c])

            if len(total_iou) == 0:
                all_permu_iou_list.append(np.zeros(1))
            else:
                all_permu_iou_list.append(sum(total_iou) / len(total_iou))

        if len(all_permu_iou_list) == 0:
            select_permu = []
        else:
            select_permu = all_permutations[all_permu_iou_list.index(max(all_permu_iou_list))]

        # Update category_id in res_dict based on the optimal matching
        with open(dt_json_path, 'r') as file:
            res_dict = json.load(file)
        for old_group, new_group in enumerate(select_permu, start=1):
            for d in res_dict:
                if d.get('category_id') == old_group and d.get('image_id') == gt_img_id:
                    d['category_id'] = new_group
                if d.get('category_id') == old_group and d.get('image_id') == pair_gt_img_id:
                    d['category_id'] = new_group

    # ========== Instance-level evaluation ==========
    coco_dt = coco.loadRes(res_dict)
    if eval_type == 'instance' or eval_type == 'all':
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Get the number of GT annotations per category
        category_ids = coco.getCatIds()
        gt_counts = {cat_id: len(coco.getAnnIds(catIds=[cat_id])) for cat_id in category_ids}

        per_class_results = coco_eval.eval['precision']

        # Set indices for IoU, area range, and max dets
        iou_index = 0  # IoU threshold (0.5)
        area_index = 0  # All area ranges
        max_dets_index = 2  # For maxDet=100

        # Compute weighted average precision
        weighted_ap50 = 0
        total_gt = sum(gt_counts.values())

        for idx, cat_id in enumerate(category_ids):
            category_precision = per_class_results[iou_index, :, idx, area_index, max_dets_index]
            valid_precisions = category_precision[category_precision > -1]
            if valid_precisions.size > 0:
                mean_precision = np.mean(valid_precisions)
                weighted_precision = (gt_counts[cat_id] / total_gt) * mean_precision
                weighted_ap50 += weighted_precision

    # ========== Image-level evaluation ==========
    if eval_type == 'image' or eval_type == 'all':
        img_ids = coco.getImgIds()

        # Save category information using Dict
        cat_per_img_gt = {}
        cat_per_img_dt = {}
        for img_id in img_ids:
            # Ground-truth categories per image
            gt_ann_ids = coco.getAnnIds(imgIds=[img_id])
            gt_anns = coco.loadAnns(gt_ann_ids)
            cat_per_img_gt[img_id] = list(set([ann['category_id'] for ann in gt_anns]))

            # Predicted categories per image
            dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)
            cat_per_img_dt[img_id] = list(set([dt['category_id'] for dt in dt_anns]))

        correct = 0
        # TODO: implementation for #image > 2
        for i in range(0, len(cat_per_img_gt), 2):
            gt_num1 = len(cat_per_img_gt[img_ids[i]])
            dt_num1 = len(cat_per_img_dt[img_ids[i]])
            gt_num2 = len(cat_per_img_gt[img_ids[i + 1]])
            dt_num2 = len(cat_per_img_dt[img_ids[i + 1]])

            gt_num1 = 1 if gt_num1 > 1 else gt_num1
            dt_num1 = 1 if dt_num1 > 1 else dt_num1
            gt_num2 = 1 if gt_num2 > 1 else gt_num2
            dt_num2 = 1 if dt_num2 > 1 else dt_num2

            # Check if each image contains at least one instance
            # A sample is considered as correct if all images is correct
            if gt_num1 == dt_num1 and gt_num2 == dt_num2:
                correct += 1
        accuracy = correct / (len(cat_per_img_gt) / 2)

    return weighted_ap50, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MC-Bench Evaluation')
    parser.add_argument('--eval_type', type=str, default="all", help='Evaluation type selected from [all, instance, image]')
    parser.add_argument('--gt_json_path', type=str, default="./MC-Bench_coco_format.json", help='Path to MC-Bench annotation')
    parser.add_argument('--dt_json_path', type=str, required=True, help='Path to your COCO-format prediction JSON file')
    args = parser.parse_args()

    ap50, acc = mcbench_eval(args.gt_json_path, args.dt_json_path, args.eval_type)
    if ap50 is not None:
        print(f"Weighted Average Precision (IoU=0.5): {ap50}")
    if acc is not None:
        print(f"Average Accuracy (IoU=0.5): {acc}")


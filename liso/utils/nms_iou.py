from typing import List

import numpy as np
from liso.kabsch.shape_utils import Shape

import torch  # isort:skip
import iou3d_nms_cuda  # isort:skip # import torch before this: https://stackoverflow.com/questions/65710713/importerror-libc10-so-cannot-open-shared-object-file-no-such-file-or-director


def hard_limit_detections(non_batched_pred_boxes, max_num_centerpoint_preds):
    top_confident_detection_idxs = torch.argsort(
        torch.squeeze(non_batched_pred_boxes.probs, dim=-1),
        dim=0,
        descending=True,
    )[: min(max_num_centerpoint_preds, non_batched_pred_boxes.pos.shape[0])]
    mask_predictions = torch.zeros_like(non_batched_pred_boxes.valid)
    mask_predictions[top_confident_detection_idxs] = True
    non_batched_pred_boxes.valid = mask_predictions
    non_batched_pred_boxes = non_batched_pred_boxes.drop_padding_boxes()
    return non_batched_pred_boxes


def perform_nms_on_shapes(
    pred_visu_boxes: Shape,
    max_num_boxes: int,
    overlap_threshold: float,
    use_cuda=True,
    pre_nms_max_num_boxes=-1,  # to not overload memory
):
    nms_suppressed_pred_boxes = []
    to_be_nmsed_boxes = pred_visu_boxes.clone()
    for batch_idx in range(to_be_nmsed_boxes.pos.shape[0]):
        non_batched_pred_boxes = to_be_nmsed_boxes[batch_idx].drop_padding_boxes()
        if (
            pre_nms_max_num_boxes > 0
            and non_batched_pred_boxes.shape[0] > pre_nms_max_num_boxes
        ):
            keep_mask = torch.zeros_like(non_batched_pred_boxes.valid)
            conf_idxs_desc = torch.argsort(
                non_batched_pred_boxes.probs, dim=0, descending=True
            )
            keep_mask[conf_idxs_desc[:pre_nms_max_num_boxes]] = True
            non_batched_pred_boxes.valid = non_batched_pred_boxes.valid & keep_mask
            non_batched_pred_boxes = non_batched_pred_boxes.drop_padding_boxes()

        if use_cuda:
            nms_pred_box_idxs = iou_based_nms(
                non_batched_pred_boxes, overlap_threshold=overlap_threshold
            )
        else:
            nms_pred_box_idxs = shapely_nms(
                non_batched_pred_boxes, overlap_threshold=overlap_threshold
            )
        det_boxes = non_batched_pred_boxes[nms_pred_box_idxs]
        det_boxes = hard_limit_detections(det_boxes, max_num_boxes)
        nms_suppressed_pred_boxes.append(det_boxes)

    nms_pred_boxes = Shape.from_list_of_shapes(nms_suppressed_pred_boxes)
    # TEST THAT NMS does not invent new objects!
    # for batch_idx, boxes in enumerate(nms_pred_boxes):
    #     for box_idx in range(boxes.pos.shape[0]):
    #         if boxes.valid[box_idx]:
    #             assert (
    #                 (boxes.pos[box_idx] == to_be_nmsed_boxes[batch_idx].pos).all(dim=-1)
    #             ).any(), f"batch {batch_idx}, box {box_idx} cannot be found before nms!"
    return nms_pred_boxes


def compute_shapely_iou(box_a, box_b):
    intersection = box_a.intersection(box_b).area
    union = box_a.union(box_b).area
    if union > 0.0:
        return intersection / union
    else:
        return 0.0


@torch.no_grad()
def iou_based_nms(
    objects: Shape,
    overlap_threshold: float,
    pre_nms_max_boxes: int = None,
    post_nms_max_boxes: int = None,
) -> List[int]:
    assert len(objects.probs.shape) == 2, objects.probs.shape
    assert objects.probs.shape[-1] == 1, objects.probs.shape
    assert objects.valid.all(), "can't handle padding boxes"
    boxes_conv = convert_shapes_to_dense_3d(objects.clone())

    assert boxes_conv.shape[-1] == 7, boxes_conv.shape
    keep = rotate_nms_pcdet(
        boxes_conv,
        torch.squeeze(objects.probs, dim=-1),
        overlap_threshold,
        pre_maxsize=pre_nms_max_boxes,
        post_max_size=post_nms_max_boxes,
    )

    return keep


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))
    ).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(
        boxes_a.contiguous(), boxes_b.contiguous(), ans_iou
    )

    return ans_iou


@torch.no_grad()
def box_iou_matrix(
    boxes_a: Shape,
    boxes_b: Shape,
    iou_mode: str = "iou_bev",
) -> List[int]:
    assert len(boxes_a.probs.shape) == 2, boxes_a.probs.shape
    assert boxes_a.probs.shape[-1] == 1, boxes_a.probs.shape
    assert boxes_a.valid.all(), "can't handle padding boxes"
    assert len(boxes_b.probs.shape) == 2, boxes_b.probs.shape
    assert boxes_b.probs.shape[-1] == 1, boxes_b.probs.shape
    assert boxes_b.valid.all(), "can't handle padding boxes"
    boxes_conv_a = convert_shapes_to_dense_3d(boxes_a.clone())
    boxes_conv_b = convert_shapes_to_dense_3d(boxes_b.clone())

    assert boxes_conv_a.shape[-1] == 7, boxes_conv_a.shape
    assert boxes_conv_b.shape[-1] == 7, boxes_conv_b.shape
    if boxes_a.shape[0] != 0 and boxes_b.shape[0] != 0:
        if iou_mode == "iou_bev":
            iou_mat = torch.cuda.FloatTensor(
                torch.Size((boxes_a.shape[0], boxes_b.shape[0]))
            ).zero_()
            iou3d_nms_cuda.boxes_iou_bev_gpu(
                boxes_conv_a.float().contiguous(),
                boxes_conv_b.float().contiguous(),
                iou_mat,
            )
        elif iou_mode == "iou_3d":
            bev_overlap_area_mat = torch.cuda.FloatTensor(
                torch.Size((boxes_a.shape[0], boxes_b.shape[0]))
            ).zero_()
            iou3d_nms_cuda.boxes_overlap_bev_gpu(
                boxes_conv_a.float().contiguous(),
                boxes_conv_b.float().contiguous(),
                bev_overlap_area_mat,
            )
            num_boxes_a = boxes_a.shape[0]
            num_boxes_b = boxes_b.shape[0]

            # find overlapping height segment along z dimension
            # min(upper_a, upper_b) - max(lower_a, lower_b)
            lower_pt_a = boxes_a.pos[:, 2] - 0.5 * boxes_a.dims[:, 2]
            lower_pt_b = boxes_b.pos[:, 2] - 0.5 * boxes_b.dims[:, 2]

            upper_pt_a = boxes_a.pos[:, 2] + 0.5 * boxes_a.dims[:, 2]
            upper_pt_b = boxes_b.pos[:, 2] + 0.5 * boxes_b.dims[:, 2]

            # broadcast everything to [num_boxes_a, num_boxes_b]
            lower_pt_a_mat = lower_pt_a[..., None].repeat((1, num_boxes_b))
            lower_pt_b_mat = lower_pt_b[None, ...].repeat((num_boxes_a, 1))

            upper_pt_a_mat = upper_pt_a[..., None].repeat((1, num_boxes_b))
            upper_pt_b_mat = upper_pt_b[None, ...].repeat((num_boxes_a, 1))

            upper_pts_mat = torch.min(upper_pt_a_mat, upper_pt_b_mat)
            lower_pts_mat = torch.max(lower_pt_a_mat, lower_pt_b_mat)

            height_overlap_mat = upper_pts_mat - lower_pts_mat
            is_pos_overlap = height_overlap_mat > 0.0

            intersection_volume = torch.where(
                is_pos_overlap, bev_overlap_area_mat * height_overlap_mat, 0.0
            )

            volume_boxes_a = torch.prod(boxes_a.dims, dim=-1)
            volume_boxes_b = torch.prod(boxes_b.dims, dim=-1)

            union_volume = (
                volume_boxes_a[..., None]
                + volume_boxes_b[None, ...]
                - intersection_volume
            )

            iou_mat = intersection_volume / torch.clip(
                union_volume, min=torch.finfo(torch.float32).eps
            )

        else:
            raise NotImplementedError(iou_mode)

    else:
        iou_mat = torch.zeros(
            (boxes_a.shape[0], boxes_b.shape[0]), device=boxes_conv_a.device
        )
    return iou_mat


@torch.no_grad()
def shapely_nms(objects: Shape, overlap_threshold: float):
    objects_np = objects.drop_padding_boxes().numpy()
    shapely_objs = objects_np.get_shapely_contour()

    sort_idxs = np.argsort(np.squeeze(objects_np.probs), axis=-1)[::-1]
    result: List[int] = []
    for idx in sort_idxs:
        suppress = False
        for r in result:
            iou = compute_shapely_iou(shapely_objs[r], shapely_objs[idx])
            if iou > overlap_threshold:
                suppress = True
                break
        if not suppress:
            result.append(idx)
    assert (objects.valid).all(), "padding not supported"
    return np.array(result)


def convert_shapes_to_dense_3d(boxes: Shape):
    dense_boxes = torch.cat(
        [
            pad_attr_to_3d_if_necessary(boxes.pos, 0.0),
            pad_attr_to_3d_if_necessary(boxes.dims, 1.0),
            boxes.rot,  # heading
        ],
        dim=-1,
    )
    dense_boxes = torch.where(
        boxes.valid[..., None], dense_boxes, torch.zeros_like(dense_boxes)
    )
    return dense_boxes


def pad_attr_to_3d_if_necessary(pos_or_dims: torch.FloatTensor, padding_value: float):
    if pos_or_dims.shape[-1] == 2:
        return torch.cat(
            [pos_or_dims, padding_value * torch.ones_like(pos_or_dims[..., [0]])],
            dim=-1,
        )
    elif pos_or_dims.shape[-1] == 3:
        return pos_or_dims
    else:
        raise NotImplementedError("can't handle shape", pos_or_dims.shape)


def rotate_nms_pcdet(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    :param boxes: (N, 5) [x, y, z, l, w, h, theta]
    :param scores: (N)
    :param thresh:
    :return:
    """
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    keep = 0 * torch.LongTensor(boxes.size(0))

    if len(boxes) == 0:
        num_out = 0
    else:
        num_out = iou3d_nms_cuda.nms_gpu(boxes.float(), keep, thresh)

    selected = order[keep[:num_out].cuda()].contiguous()

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected

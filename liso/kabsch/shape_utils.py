import pprint
from typing import Tuple, Union

import numpy as np
import torch
from liso.utils.torch_transformation import (
    homogenize_pcl,
    numpy_compose_matrix,
    torch_compose_matrix,
    torch_decompose_matrix,
)
from shapely.affinity import rotate, translate
from shapely.geometry import Point, box

UNKNOWN_CLASS_ID = torch.iinfo(torch.int32).max
INVALID_CLASS_ID = UNKNOWN_CLASS_ID - 1


class Shape:
    _numeric_float_keys = ("pos", "dims", "rot", "probs", "velo")
    _numeric_int_keys = ("class_id", "difficulty")

    def __init__(
        self,
        pos,
        dims,
        rot,
        probs,
        velo=None,
        valid=None,
        class_id=None,
        difficulty=None,
    ) -> None:
        assert pos.shape[-1] in (1, 2, 3), pos.shape
        assert dims.shape[-1] in (1, 2, 3), dims.shape
        assert probs.shape[-1] == 1, probs.shape
        self.pos = pos
        self.dims = dims
        self.rot = rot
        self.probs = probs
        if valid is not None:
            self.valid = valid
        elif valid is None and torch.is_tensor(probs):
            self.valid = torch.ones_like(probs[..., 0], dtype=torch.bool)
        elif valid is None and isinstance(probs, np.ndarray):
            self.valid = np.ones_like(probs[..., 0], dtype=bool)
        else:
            raise NotImplementedError(
                "valid is not None but type of probs is unhandeled"
            )

        if velo is not None:
            assert velo.shape[-1] in (1, 2, 3), velo.shape
            self.velo = velo
        elif velo is None and torch.is_tensor(pos):
            self.velo = torch.zeros_like(probs)
        elif velo is None and isinstance(probs, np.ndarray):
            self.velo = np.zeros_like(probs)
        else:
            raise NotImplementedError("velo is not None but type of pos is unhandeled")

        if class_id is not None:
            assert class_id.shape[-1] == 1, class_id.shape
            self.class_id = class_id
        elif class_id is None and torch.is_tensor(pos):
            self.class_id = UNKNOWN_CLASS_ID * torch.ones_like(
                pos[..., :1], dtype=torch.int32
            )
        elif class_id is None and isinstance(pos, np.ndarray):
            self.class_id = UNKNOWN_CLASS_ID * np.ones_like(
                pos[..., :1], dtype=np.int32
            )
        else:
            raise NotImplementedError(
                "class id is not None but type of pos is unhandeled"
            )

        if difficulty is not None:
            assert difficulty.shape[-1] == 1, difficulty.shape
            self.difficulty = difficulty
        elif difficulty is None and torch.is_tensor(pos):
            self.difficulty = torch.ones_like(pos[..., :1], dtype=torch.int)
        elif difficulty is None and isinstance(pos, np.ndarray):
            self.difficulty = np.zeros_like(pos[..., :1], dtype=np.int32)
        else:
            raise NotImplementedError(
                "class id is not None but type of pos is unhandeled"
            )

        assert len(self.valid.shape) == len(self.probs.shape) - 1
        assert len(self.valid.shape) == len(self.dims.shape) - 1
        assert len(self.valid.shape) == len(self.rot.shape) - 1

    @staticmethod
    def createEmpty():
        return Shape(
            pos=np.empty((0, 3)),
            dims=np.empty((0, 3)),
            rot=np.empty((0, 1)),
            probs=np.empty((0, 1)),
            valid=np.zeros((0), dtype=bool),
            class_id=np.zeros((0, 1), dtype=np.int32),
            difficulty=np.zeros((0, 1), dtype=np.int32),
            velo=np.empty((0, 1)),
        )

    @property
    def shape(self):
        return self.valid.shape

    @staticmethod
    def from_list_of_shapes(
        shapes_list, numeric_padding_value=np.nan, int_padding_value=INVALID_CLASS_ID
    ):
        if all([shape.__dict__["valid"].shape == () for shape in shapes_list]):
            valid_masks = torch.stack(
                [shape.__dict__["valid"] for shape in shapes_list]
            )
            batched_entries = {"valid": valid_masks}
        else:
            batched_entries = {
                "valid": torch.nn.utils.rnn.pad_sequence(
                    [shape.__dict__["valid"] for shape in shapes_list],
                    batch_first=True,
                    padding_value=False,
                )
            }
        for key in Shape._numeric_float_keys:
            batched_entries[key] = torch.nn.utils.rnn.pad_sequence(
                [shape.__dict__[key] for shape in shapes_list],
                batch_first=True,
                padding_value=numeric_padding_value,
            )
        for key in Shape._numeric_int_keys:
            batched_entries[key] = torch.nn.utils.rnn.pad_sequence(
                [shape.__dict__[key] for shape in shapes_list],
                batch_first=True,
                padding_value=int_padding_value,
            )
        return Shape(**batched_entries)

    @staticmethod
    def from_list_of_npy_shapes(shapes_list):
        assert all(
            [shape.__dict__["valid"].shape == () for shape in shapes_list]
        ), "batching is not supported for numpy!"
        stacked_shape_attrs = {}
        for key in Shape._numeric_float_keys + Shape._numeric_int_keys + ("valid",):
            stacked_shape_attrs[key] = np.stack(
                [shape.__dict__[key] for shape in shapes_list],
            )
        return Shape(**stacked_shape_attrs)

    def assert_attr_shapes_compatible(
        self,
    ):
        valid_mask_shape = self.valid.shape
        for k, v in self.__dict__.items():
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                if k == "valid":
                    continue
                assert v.shape[:-1] == valid_mask_shape, (
                    f"Shape incompatible for key {k}",
                    self.print_attr_shapes(),
                )

    def print_attr_shapes(self):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: None")

    def detach(self):
        detached_tensors = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                detached_tensors[k] = v.detach()
        return Shape(**detached_tensors)

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            dont_map = ()
        elif isinstance(device_or_dtype, torch.dtype) or device_or_dtype in (
            np.float32,
            np.float64,
        ):
            dont_map = ("valid",) + self._numeric_int_keys
        else:
            raise NotImplementedError(f"Don't know what to do with {device_or_dtype}")
        for k, v in self.__dict__.items():
            if torch.is_tensor(v) and k not in dont_map:
                self.__dict__[k] = v.to(device_or_dtype)
            elif isinstance(v, np.ndarray) and k not in dont_map:
                self.__dict__[k] = v.astype(device_or_dtype)
        return self

    def cpu(self):
        for k, v in self.__dict__.items():
            self.__dict__[k] = v.to("cpu")
        return self

    def numpy(self):
        clone = self.clone()
        for k, v in clone.__dict__.items():
            if torch.is_tensor(v):
                clone.__dict__[k] = v.detach().cpu().numpy()
        return clone

    def __getitem__(self, key):
        if (
            isinstance(key, (int, np.int64))
            or key is None
            or (
                torch.is_tensor(key)
                and key.dtype in (torch.int, torch.long, torch.bool)
            )
            or all(isinstance(x, (int, np.int64)) for x in key)
            or key.dtype == bool  # boolean masks are okay
        ):
            batch_values = {}
            batch_values["rot"] = None
            for k, v in self.__dict__.items():
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    batch_values[k] = v[key]
            return Shape(**batch_values)
        else:
            raise NotImplementedError(f"don't know what to do with key {key}")

    def clone(self):
        cloned_entries = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                cloned_entries[k] = v.clone()
            elif isinstance(v, np.ndarray):
                cloned_entries[k] = v.copy()
            elif k in ("rot", "valid") and v is None:
                cloned_entries[k] = None
            else:
                raise AssertionError(f"dont know what to do with {k} of type {type(v)}")
        return Shape(**cloned_entries)

    def drop_padding_boxes(self):
        if len(self.shape) == 1:
            valid_only_entries = {}
            valid_only_entries["rot"] = None
            if torch.is_tensor(self.valid):
                valid_mask = self.valid.clone()
            elif isinstance(self.valid, np.ndarray):
                valid_mask = self.valid.copy()
            else:
                raise NotImplementedError("only tensors and ndarrays supported")
            for k, v in self.__dict__.items():
                if torch.is_tensor(v):
                    assert (
                        k == "valid" or len(v.shape) == 2
                    ), f"{k} batching not supported {v.shape}"
                    valid_only_entries[k] = v.clone()[valid_mask]
                elif isinstance(v, np.ndarray):
                    assert (
                        k == "valid" or len(v.shape) == 2
                    ), f"{k} batching not supported {v.shape}"
                    valid_only_entries[k] = np.copy(v)[valid_mask]
            return Shape(**valid_only_entries)
        else:
            shapes = Shape.from_list_of_shapes(
                [self[i].drop_padding_boxes() for i in range(self.shape[0])]
            )
            return shapes

    def get_poses(self) -> torch.FloatTensor:
        is_unbatched_input = len(self.pos.shape) == 2

        if is_unbatched_input:
            pos = self.pos[None, ...]
            rot = self.rot[None, ...]
        else:
            assert len(self.pos.shape) == 3, self.pos.shape
            pos = self.pos
            rot = self.rot
        if torch.is_tensor(self.pos):
            t_z = None if self.pos.shape[-1] == 2 else pos[..., 2].to(torch.double)
            if self.rot is None or self.rot.shape[-1] == 0:
                pose = torch_compose_matrix(
                    pos[..., 0].to(torch.double),
                    pos[..., 1].to(torch.double),
                    torch.zeros_like(pos[..., 0], dtype=torch.double),
                    t_z=t_z,
                )
            else:
                assert torch.all(torch.isfinite(rot))
                pose = torch_compose_matrix(
                    pos[..., 0].to(torch.double),
                    pos[..., 1].to(torch.double),
                    rot[..., 0].to(torch.double),
                    t_z=t_z,
                )
        else:
            t_z = None if self.pos.shape[-1] == 2 else pos[..., 2].astype(np.double)
            if self.rot is None or self.rot.shape[-1] == 0:
                pose = numpy_compose_matrix(
                    pos[..., 0].astype(np.float64),
                    pos[..., 1].astype(np.float64),
                    np.zeros_like(pos[..., 0], dtype=np.float64),
                    t_z=t_z,
                )
            else:
                assert np.all(np.isfinite(rot))
                pose = numpy_compose_matrix(
                    pos[..., 0].astype(np.float64),
                    pos[..., 1].astype(np.float64),
                    rot[..., 0].astype(np.float64),
                    t_z=t_z,
                )

        if is_unbatched_input:
            pose = pose[0, ...]

        return pose

    def cat(self, other, dim: int):
        concatted_vals = {}
        concatted_vals["rot"] = None
        for key in self.__dict__:
            if self.__dict__[key] is not None:
                assert len(self.__dict__[key].shape) == len(
                    other.__dict__[key].shape
                ), (self.__dict__[key].shape, other.__dict__[key].shape)
                concatted_vals[key] = torch.cat(
                    [self.__dict__[key], other.__dict__[key]], dim=dim
                )
        return Shape(**concatted_vals)

    def get_shapely_contour(self):
        retval = []
        assert len(self.pos.shape) == 2, "can't handle batched content"
        for pos, dims, rot in zip(self.pos, self.dims, self.rot):
            if self.dims.shape[-1] == 1:
                rc = Point(pos[0], pos[1]).buffer(dims[0] / 2)
                retval.append(rc)
            else:
                c = box(-dims[0] / 2.0, -dims[1] / 2.0, dims[0] / 2.0, dims[1] / 2.0)
                rc = rotate(c, rot, use_radians=True)
                retval.append(translate(rc, pos[0], pos[1]))
        return retval

    def change_order_confidence_descending(self):
        assert self.probs.shape[-1] == 1, self.probs.shape
        assert (
            len(self.pos.shape) == 2
        ), "can't handle batched inputs due to nan padding values etc"
        if torch.is_tensor(self.probs):
            assert torch.all(
                torch.isfinite(self.probs)
            ), "dont know how to sort infinite values"
            target_order = torch.squeeze(
                torch.argsort(self.probs, dim=0, descending=True), dim=-1
            )
        elif isinstance(self.probs, np.ndarray):
            assert np.all(
                np.isfinite(self.probs)
            ), "dont know how to sort infinite values"
            target_order = np.squeeze(
                np.argsort(self.probs, axis=0, descending=True), axis=-1
            )
        else:
            raise NotImplementedError(type(self.probs))

        for key in self.__dict__:
            if self.__dict__[key] is not None:
                self.__dict__[key] = self.__dict__[key][target_order]

    @staticmethod
    def get_bottom_corner_idxs():
        return (0, 1, 4, 5)

    def get_box_corners(self):
        assert self.dims.shape[-1] == 3, self.dims.shape
        assert self.pos.shape[-1] == 3, self.pos.shape

        unit_cube = (
            0.5
            * np.array(
                (
                    (1.0, -1.0, -1.0),  # front right bottom
                    (1.0, 1.0, -1.0),  # front left bottom
                    (1.0, 1.0, 1.0),  # front left top
                    (1.0, -1.0, 1.0),  # front right top
                    (-1.0, -1.0, -1.0),  # rear right bottom
                    (-1.0, 1.0, -1.0),  # rear left bottom
                    (-1.0, 1.0, 1.0),  # rear left top
                    (-1.0, -1.0, 1.0),  # rear right top
                )
            )[None, ...]
        )
        draw_line_sequence = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        is_tensor = torch.is_tensor(self.dims)
        if is_tensor:
            unit_cube = (
                torch.from_numpy(unit_cube).to(self.dims.dtype).to(self.dims.device)
            )
        is_batched = len(self.dims.shape) == 3
        if is_batched:
            unit_cube = unit_cube[None, ...]
        box_corners = unit_cube * self.dims[..., None, :]
        if is_tensor:
            box_corners_homog = torch.cat(
                [box_corners, torch.ones_like(box_corners[..., [0]])], dim=-1
            )
            s_T_box = self.get_poses()
            box_corners_sensor = torch.einsum(
                "...ij,...cj->...ci", s_T_box, box_corners_homog.to(s_T_box.dtype)
            )[..., :3]
        else:
            box_corners_homog = np.concatenate(
                [box_corners, np.ones_like(box_corners[..., [0]])], dim=-1
            )
            s_T_box = self.get_poses()
            box_corners_sensor = np.einsum(
                "...kij,...kcj->...kci", s_T_box, box_corners_homog
            )[..., :3]

        return box_corners_sensor, draw_line_sequence

    def set_padding_val_to(self, value=0.0, int_value=INVALID_CLASS_ID):
        for key in Shape._numeric_float_keys:
            if torch.is_tensor(self.__dict__[key]):
                self.__dict__[key] = torch.where(
                    self.valid[..., None],
                    self.__dict__[key],
                    torch.tensor(value).to(
                        self.__dict__[key].device, self.__dict__[key].dtype
                    ),
                )
            else:
                self.__dict__[key] = np.where(self.valid, self.__dict__[key], value)

        for key in Shape._numeric_int_keys:
            if torch.is_tensor(self.__dict__[key]):
                self.__dict__[key] = torch.where(
                    self.valid[..., None],
                    self.__dict__[key],
                    torch.tensor(int_value).to(
                        self.__dict__[key].device, self.__dict__[key].dtype
                    ),
                )
            else:
                self.__dict__[key] = np.where(self.valid, self.__dict__[key], int_value)

    def __str__(self):
        s = pprint.pformat(self.__dict__)
        return s

    def to_tensor(self):
        shape_dict = {k: torch.from_numpy(v) for k, v in self.__dict__.items()}
        return Shape(**shape_dict)

    def transform(self, new_T_old: torch.FloatTensor):
        boxes_new_cosy = self.clone()

        old_T_box = self.get_poses()

        poses_new = new_T_old @ old_T_box

        pos_new, rot_new = torch_decompose_matrix(poses_new)

        boxes_new_cosy.pos = pos_new
        boxes_new_cosy.rot = rot_new

        boxes_new_cosy.assert_attr_shapes_compatible()
        return boxes_new_cosy

    @torch.no_grad()
    def get_points_in_box_bool_mask(
        self,
        pcl: Union[np.ndarray, torch.FloatTensor],
        box_dims_bloat_factor: float = 1.0,
        return_points_in_box_coords=False,
    ) -> Union[
        Union[
            Tuple[np.ndarray, np.ndarray],
            Tuple[torch.Tensor, torch.Tensor],
        ],
        np.ndarray,
        torch.Tensor,
    ]:
        assert pcl.shape[-1] == 3, pcl.shape
        assert len(self.shape) == len(pcl.shape) - 1, (self.shape, pcl.shape)
        if len(self.shape) == 2:
            # must have same batch dim
            assert self.shape[0] == pcl.shape[0], (self.shape, pcl.shape)
        assert torch.is_tensor(self.pos) == torch.is_tensor(
            pcl
        ), "need either both np arrays or tensors!"
        sensor_T_box = self.get_poses()

        pcl_homog = homogenize_pcl(pcl[..., :3])
        relevant_box_dims = box_dims_bloat_factor * self.dims
        if torch.is_tensor(pcl_homog):
            pts_homog_in_box = torch.einsum(
                "...kij,...nj->...nki",
                torch.linalg.inv(sensor_T_box).to(torch.float),
                pcl_homog,
            )
            assert torch.all(torch.isfinite(pts_homog_in_box))
            pt_is_in_box = torch.all(
                torch.abs(pts_homog_in_box[..., 0:3]) < 0.5 * relevant_box_dims,
                dim=-1,
            )
        else:
            pts_homog_in_box = np.einsum(
                "...kij,...nj->...nki",
                np.linalg.inv(sensor_T_box),
                pcl_homog,
            )
            assert np.all(np.isfinite(pts_homog_in_box))
            pt_is_in_box = np.all(
                np.abs(pts_homog_in_box[..., 0:3]) < 0.5 * relevant_box_dims,
                axis=-1,
            )
        if return_points_in_box_coords:
            return pt_is_in_box, pts_homog_in_box
        else:
            return pt_is_in_box

    def into_list_of_shapes(self):
        box_list = []
        for box_idx in range(self.shape[0]):
            boxes = self[box_idx]
            boxes = boxes.drop_padding_boxes()
            box_list.append(boxes)
        return box_list


def is_boxes_clearly_in_bev_range(
    boxes: Shape, bev_range_m: Union[torch.FloatTensor, np.ndarray]
) -> Union[torch.BoolTensor, np.ndarray]:
    assert len(bev_range_m) == 2, bev_range_m
    if torch.is_tensor(boxes.pos):
        box_xy_pos = torch.abs(boxes.pos[..., :2]) - boxes.dims[..., [0]] / 2
        is_in_range = torch.all(torch.abs(box_xy_pos) < bev_range_m / 2, dim=-1)
    else:
        box_xy_pos = np.abs(boxes.pos[..., :2]) - boxes.dims[..., [0]] / 2
        is_in_range = np.all(np.abs(box_xy_pos) < bev_range_m / 2, dim=-1)

    return is_in_range


def extract_motion_in_pred_box_coordinates(
    pred_boxes_a: Shape,
    fg_kabsch_trafos: torch.DoubleTensor,
    bg_kabsch_trafo: torch.DoubleTensor,
) -> Tuple[torch.DoubleTensor, torch.DoubleTensor]:
    b0_deltaT_b1 = extract_box_motion_transform_without_sensor_odometry(
        pred_boxes_a, fg_kabsch_trafos, bg_kabsch_trafo
    )

    box_trans, box_rot = torch_decompose_matrix(b0_deltaT_b1)
    # box_trans = torch.einsum(
    #     "bsij,j-> bsi",
    #     b0_deltaT_b1,
    #     torch.tensor(
    #         [0.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=b0_deltaT_b1.device
    #     ),
    # )[..., 0:3]
    return box_trans, box_rot


def extract_box_motion_transform_without_sensor_odometry(
    pred_boxes_a: Shape,
    fg_kabsch_trafos: torch.DoubleTensor,
    bg_kabsch_trafo: torch.DoubleTensor,
) -> torch.DoubleTensor:
    # need to replicate this using kabsch only
    # s0_T_box0 = gt_boxes_t0.get_poses()
    # s0_T_s1 = bg_kabsch_trafo
    # box0_T_s0 = torch.linalg.inv(s0_T_box0)
    # s1_T_box1 = gt_boxes_t1.get_poses()
    # b0_deltaT_b1 = box0_T_s0 @ s0_T_s1 @ s1_T_box1
    # gt_transl, _ = torch_decompose_matrix(b0_deltaT_b1)
    s0_T_box0 = pred_boxes_a.get_poses()
    s0_T_s1 = torch.linalg.inv(bg_kabsch_trafo)
    box0_T_s0 = torch.linalg.inv(s0_T_box0)

    s1_T_box1 = fg_kabsch_trafos @ s0_T_box0
    b0_deltaT_b1 = box0_T_s0 @ s0_T_s1 @ s1_T_box1
    # s0_Tt0_box = pred_boxes_a.get_poses()  # current pose
    # box_centric_kabsch_trafos = (
    #     torch.linalg.inv(s0_Tt0_box) @ (fg_kabsch_trafos - bg_kabsch_trafo) @ s0_Tt0_box
    # )
    return b0_deltaT_b1


def soft_align_box_flip_orientation_with_motion_trafo(
    boxes: Shape,
    fg_kabsch_trafos: torch.FloatTensor,
    bg_kabsch_trafo: torch.FloatTensor,
    no_align_for_displacement_below_m=0.1,
    full_align_for_displacement_above_m=0.3,
):
    # if box motion > no_align_for_disp_below:
    # flip box if motion is opposite to orientation
    # also, align box to motion direction
    assert no_align_for_displacement_below_m < full_align_for_displacement_above_m, (
        no_align_for_displacement_below_m,
        full_align_for_displacement_above_m,
    )
    box_translation, _ = extract_motion_in_pred_box_coordinates(
        boxes, fg_kabsch_trafos, bg_kabsch_trafo
    )
    box_displacement_m = torch.linalg.norm(box_translation[..., :2], dim=-1)
    box_needs_flip = (box_translation[..., 0] < 0.0) & (
        box_displacement_m > no_align_for_displacement_below_m
    )
    box_translation[..., :2] = torch.where(
        box_needs_flip[..., None], -box_translation[..., :2], box_translation[..., :2]
    )
    assert boxes.rot.shape[-1] == 1, boxes.rot.shape
    boxes.rot = torch.where(box_needs_flip[..., None], boxes.rot + np.pi, boxes.rot)

    alignment_ratio = (box_displacement_m - no_align_for_displacement_below_m) / (
        full_align_for_displacement_above_m - no_align_for_displacement_below_m
    )
    alignment_ratio = torch.clip(alignment_ratio, min=0.0, max=1.0)[..., None]
    delta_angle = torch.atan2(box_translation[..., [1]], box_translation[..., [0]])
    boxes.rot = boxes.rot + alignment_ratio * delta_angle

    box_velo = torch.zeros_like(box_translation)
    box_velo[..., 0] = box_displacement_m
    boxes.velo = box_velo
    return boxes

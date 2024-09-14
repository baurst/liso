from collections import abc, defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from liso.datasets.kitti.kitti_range_image_projection_helper import (
    kitti_pcl_projection_get_rows_cols,
)
from liso.datasets.nuscenes.analyse_boxes import voxelize_pcl
from liso.kabsch.kabsch_mask import (
    batched_render_gaussian_kabsch_mask,
    render_hard_kabsch_mask,
)
from liso.kabsch.shape_utils import Shape
from liso.tracker.augm_box_db_utils import load_sanitize_box_augmentation_database
from liso.tracker.mined_box_db_utils import load_mined_boxes_db
from liso.transformations.transformations import compose_matrix, decompose_matrix
from liso.utils.bev_utils import get_bev_setup_params
from liso.utils.cloud_utils import CloudLoaderSaver
from liso.utils.numpy_scatter import scatter_mean_nd_numpy
from liso.utils.torch_transformation import (
    homogenize_flow,
    homogenize_pcl,
    torch_compose_matrix,
    torch_decompose_matrix,
)
from liso.visu.pcl_image import create_occupancy_pcl_image
from skimage.morphology import binary_dilation, disk


# LidarSample = namedtuple("Sample", ("idx", "sample_name", "timestamp", "full_path"))
class LidarSample:
    def __init__(self, idx, sample_name, timestamp, full_path):
        self.idx = idx
        self.sample_name = sample_name
        self.timestamp = timestamp
        self.full_path = full_path


KITTI_MOVABLE_CLASSES = ("Car", "Pedestrian", "Cyclist")

KITTI_MAP_TO_SIMPLE_CLASSES = {
    "Car": "Car",
    "PassengerCar": "Car",
    "Pedestrian": "Pedestrian",
    "Person": "Pedestrian",
    "Van": "Car",
    "Truck": "Car",
    "Person_sitting": "Pedestrian",
    "Cyclist": "Cyclist",
    "Tram": "Car",
}

KITTI_IGNORE_NON_MOVABLE_CLASSMAPPING = {
    "Unknown": None,
    "DontCare": None,
    "Car": "movable",
    "PassengerCar": "movable",
    "Pedestrian": "movable",
    "Person": "movable",
    "Van": "movable",
    "Truck": "movable",
    "Person_sitting": None,
    "Cyclist": "movable",
    "Tram": "movable",
    "Misc": None,
    "LargeVehicle": "movable",
}


def worker_init_fn(worker_id):
    np.random.seed(4 + worker_id)


def add_lidar_rows_to_kitti_sample(
    sample_content, time_keys: Tuple[str], pcl_key="pcl_"
):
    for tk in time_keys:
        pcl_key = f"pcl_{tk}"
        if pcl_key in sample_content:
            pcl = sample_content[f"pcl_{tk}"]
            row_idxs, _ = kitti_pcl_projection_get_rows_cols(pcl)
            sample_content[f"lidar_rows_{tk}"] = row_idxs.astype(np.uint8)


def change_dict_keys(obj, convert):
    """
    Recursively goes through the dictionary obj and replaces keys with the convert function.
    """
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, dict):
        new = obj.__class__()
        for k, v in obj.items():
            new[convert(k)] = change_dict_keys(v, convert)
    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(change_dict_keys(v, convert) for v in obj)
    else:
        return obj
    return new


def get_centermaps_output_grid_size(cfg, output_grid_size):
    if cfg.network.name == "centerpoint":
        ds_factor = get_centermaps_downsampling_factor(cfg)
        if cfg.network.centerpoint.reduce_receptive_field == 1:
            ds_factor //= 2
        elif cfg.network.centerpoint.reduce_receptive_field == 0:
            pass
        else:
            raise NotImplementedError()
        return output_grid_size // ds_factor
    elif cfg.network.name == "transfusion":
        centermaps_output_grid_size = (
            output_grid_size // cfg.network.transfusion.out_size_factor
        )
        return centermaps_output_grid_size
    elif cfg.network.name in ("flow_cluster_detector", "echo_gt"):
        centermaps_output_grid_size = output_grid_size // 2
        return centermaps_output_grid_size
    else:
        return None


def get_centermaps_downsampling_factor(cfg):
    ds_factor = 4 if cfg.network.centerpoint.use_baseline_parameters else 8
    return ds_factor


def infer_ground_label_using_cone(
    pcl, cone_z_threshold__m: float = -1.70, cone_angle__deg: float = 0.8
):
    assert 0.0 <= cone_angle__deg <= 10.0  # 10 deg arbitrary high value as sanity check
    if cone_angle__deg > 0.0:
        cone_angle = cone_angle__deg / 180.0 * np.pi
        d_xy = np.linalg.norm(pcl[..., 0:2], axis=-1)
        z_t_thresh = cone_z_threshold__m + np.tan(cone_angle) * d_xy
        is_ground = pcl[..., 2] < z_t_thresh
    else:
        is_ground = pcl[..., 2] < cone_z_threshold__m
    return is_ground


def recursive_npy_dict_to_torch(sample):
    if torch.is_tensor(sample):
        return sample
    elif isinstance(sample, dict):
        return {k: recursive_npy_dict_to_torch(v) for k, v in sample.items()}
    elif isinstance(sample, np.ndarray):
        return torch.from_numpy(sample)
    elif isinstance(sample, Shape):
        # shape_dict = {k: torch.from_numpy(v) for k, v in sample.__dict__.items()}
        return sample.to_tensor()
    else:
        raise ValueError("unknown type of sample", type(sample))


def downsample_dict(data_dict, keep_mask, downsample_keys):
    for k, v in data_dict.items():
        if isinstance(v, abc.Mapping):
            data_dict[k] = downsample_dict(v, keep_mask, downsample_keys)
        elif k in downsample_keys:
            data_dict[k] = v[keep_mask]
        else:
            pass
    return data_dict


def lidar_dataset_collate_fn(data_list):
    sample_datas_t0, sample_datas_t1, augm_sample_datas_t0, meta_datas = list(
        zip(*data_list)
    )

    # train_datas = {k: [dic[k] for dic in train_datas] for k in train_datas[0]}
    # train_data = {k: torch.stack(v, dim=0) for k, v in train_datas.items()}
    sample_data_t0 = collate_list_data(sample_datas_t0)
    sample_data_t1 = collate_list_data(sample_datas_t1)
    augm_sample_data_t0 = collate_list_data(augm_sample_datas_t0)
    meta_data = defaultdict(list)
    for md in meta_datas:
        for key, val in md.items():
            meta_data[key].append(val)

    return sample_data_t0, sample_data_t1, augm_sample_data_t0, meta_data


def draw_heat_regression_maps(
    boxes: Shape,  # numpy!
    grid_size: np.ndarray,
    bev_range_m: np.ndarray,
    box_pred_cfg: Dict[str, Dict[str, str]],
    per_obj_prob_scale: np.ndarray = None,
    normalize_gaussian=False,
):
    assert bev_range_m.shape == (2,), bev_range_m.shape
    if np.count_nonzero(boxes.valid) > 0:
        per_slot_prob_heatmap = batched_render_gaussian_kabsch_mask(
            box_x=boxes.pos[None, :, 0],
            box_y=boxes.pos[None, :, 1],
            box_len=boxes.dims[None, :, 0],
            box_w=boxes.dims[None, :, 1],
            box_theta=boxes.rot[None, :, 0],
            bev_range_x=bev_range_m[0],
            bev_range_y=bev_range_m[1],
            img_shape=grid_size,
            normalize_gaussian=normalize_gaussian,
        )
        per_slot_prob_heatmap = np.squeeze(per_slot_prob_heatmap, axis=0)
        occupancy_thresh = 0.01
        occupancy_mask_f32 = (per_slot_prob_heatmap > occupancy_thresh).astype(
            np.float32
        )[..., None]
        # START SECTION DIMS
        if box_pred_cfg.dimensions_representation.method == "predict_abs_size":
            box_dims = boxes.dims
        elif box_pred_cfg.dimensions_representation.method == "predict_log_size":
            assert box_pred_cfg.activations.dims == "exp", box_pred_cfg.activations.dims
            box_dims = np.log(boxes.dims)
        else:
            raise NotImplementedError(box_pred_cfg.dimensions_representation.method)
        dims_map = occupancy_mask_f32 * box_dims[:, None, None, :]
        # END DIMS

        # START SECTION ROT
        if box_pred_cfg.rotation_representation.method == "vector":
            sin_yaw, cos_yaw = np.sin(boxes.rot), np.cos(boxes.rot)
            sincos_yaw = np.concatenate([sin_yaw, cos_yaw], axis=-1)
            rot_map = occupancy_mask_f32 * sincos_yaw[:, None, None, :]
        elif box_pred_cfg.rotation_representation.method in ("direct", "class_bins"):
            rot_map = occupancy_mask_f32 * boxes.rot[:, None, None, :]
        else:
            raise NotImplementedError(box_pred_cfg.rotation_representation.method)
        # END ROT

        # START SECTION POS
        if box_pred_cfg.position_representation.method in (
            "global_absolute",
            "local_relative_offset",
        ):
            pos_map = occupancy_mask_f32 * boxes.pos[:, None, None, :]
        elif box_pred_cfg.position_representation.method == "local_relative_offset":
            raise AssertionError("this did not work so well, dropped it")
            # # assert box_pred_cfg.activations.pos in ("None","none",None), box_pred_cfg.activations.pos
            # # TEST
            # # bp = np.array(
            # #     [
            # #         [0.0, 0.0, 0.0],
            # #     ]
            # # )
            # # b_range = np.array([8, 8, 4])
            # # gr_size = np.array([4, 4, 1])
            # # box_center_voxel_coors, _ = voxelize_pcl(bp, b_range, gr_size)
            # # box_center_pillar_coors = box_center_voxel_coors[..., :2]

            # # box_pillar_center_coord_m = -b_range[:2] / 2 + (
            # #     box_center_pillar_coors + 0.5
            # # ) * (b_range[:2] / gr_size[:2])

            # # delta_pillar_center_box_pos = bp[..., :2] - box_pillar_center_coord_m
            # # assert np.all(
            # #     np.abs(delta_pillar_center_box_pos) <= b_range[:2] / gr_size[:2]
            # # )
            # # rel_box_pos = np.concatenate(
            # #     [delta_pillar_center_box_pos, bp[..., 2:]], axis=-1
            # # )

            # # END TEST
            # box_center_voxel_coors, _ = voxelize_pcl(
            #     boxes.pos,
            #     np.concatenate([bev_range_m, np.array([100.0])]),
            #     np.concatenate([grid_size, np.array([1])]),
            # )
            # box_center_pillar_coors = box_center_voxel_coors[..., :2]

            # box_pillar_center_coord_m = -bev_range_m / 2 + (
            #     box_center_pillar_coors + 0.5
            # ) * (bev_range_m / grid_size)

            # delta_pillar_center_box_pos = boxes.pos[..., :2] - box_pillar_center_coord_m
            # assert np.all(
            #     np.abs(delta_pillar_center_box_pos) <= bev_range_m / grid_size
            # )
            # rel_box_pos = np.concatenate(
            #     [delta_pillar_center_box_pos, boxes.pos[..., 2:]], axis=-1
            # )
            # pos_map = occupancy_mask_f32 * rel_box_pos[:, None, None, :]

        else:
            raise NotImplementedError(box_pred_cfg.position_representation.method)
        # END POS

        assert boxes.velo.shape[-1] == 1, boxes.velo.shape
        velo_map = occupancy_mask_f32 * boxes.velo[:, None, None, :]
        if per_obj_prob_scale is not None:
            assert not normalize_gaussian
            assert per_obj_prob_scale.shape[-1] == 1, per_obj_prob_scale.shape
            per_slot_prob_heatmap = (
                per_obj_prob_scale[:, :, None] * per_slot_prob_heatmap
            )
        prob_heatmap = np.max(per_slot_prob_heatmap, axis=0)[..., None]

        hottest_object_mask = (
            per_slot_prob_heatmap.max(axis=0, keepdims=1) == per_slot_prob_heatmap
        )
        maps = {
            "probs": prob_heatmap,
            "dims": (hottest_object_mask[..., None] * dims_map).sum(axis=0),
            "pos": (hottest_object_mask[..., None] * pos_map).sum(axis=0),
            "rot": (hottest_object_mask[..., None] * rot_map).sum(axis=0),
            "velo": (hottest_object_mask[..., None] * velo_map).sum(axis=0),
        }
        gt_center_mask = np.squeeze(
            create_occupancy_pcl_image(boxes.pos[boxes.valid], bev_range_m, grid_size)
            > 0.5,
            axis=-1,
        )
    else:
        if box_pred_cfg.rotation_representation.method == "vector":
            rot_dims = (2,)
        elif box_pred_cfg.rotation_representation.method in ("direct", "class_bins"):
            rot_dims = (1,)
        else:
            raise NotImplementedError(box_pred_cfg.rotation_representation.method)
        maps = {
            "probs": np.zeros(tuple(grid_size) + (1,)),
            "dims": np.zeros(tuple(grid_size) + (3,)),
            "pos": np.zeros(tuple(grid_size) + (3,)),
            "rot": np.zeros(tuple(grid_size) + rot_dims),
            "velo": np.zeros(tuple(grid_size) + (1,)),
        }
        gt_center_mask = np.zeros(tuple(grid_size), dtype=bool)

    maps = {k: v.astype(np.float32) for k, v in maps.items()}
    maps["center_bool_mask"] = gt_center_mask
    return maps


def recursive_defaultdict_to_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = recursive_defaultdict_to_dict(v)
    return dict(d)


def list_of_dict_to_dict_of_list(in_list):
    if all(torch.is_tensor(el) for el in in_list):
        return in_list
    else:
        res = defaultdict(list)
        {res[key].append(sub[key]) for sub in in_list for key in sub}
        return recursive_defaultdict_to_dict(res)


def list_of_dicts_of_dicts_to_dict_of_dicts_of_lists(in_data):
    out_dict = list_of_dict_to_dict_of_list(in_data)
    nested = False
    for v in out_dict.values():
        is_nested = any(isinstance(i, abc.Mapping) for i in v)
        nested = nested | is_nested
        if nested:
            break
    if nested:
        return {k: list_of_dict_to_dict_of_list(v) for k, v in out_dict.items()}
    else:
        return out_dict


def collate_list_data(sample_datas_t0):
    nested_dict_of_lists = list_of_dicts_of_dicts_to_dict_of_dicts_of_lists(
        sample_datas_t0
    )
    for k, v in nested_dict_of_lists.items():
        change_k_v(k, nested_dict_of_lists, v)
    return nested_dict_of_lists


def change_k_v(key, parent_dict, value):
    if key in ("pcl_ta", "pcl_tb", "pcl_tx") and isinstance(
        value, abc.Mapping
    ):  # , "pillar_coors", "point_flow", "moving_mask"):
        padded_pcls = torch.nn.utils.rnn.pad_sequence(
            value["pcl"], batch_first=True, padding_value=np.nan
        )
        padded_pillar_coors = torch.nn.utils.rnn.pad_sequence(
            value["pillar_coors"],
            batch_first=True,
            padding_value=-1,
        )
        padding_mask = torch.logical_not(torch.isnan(padded_pcls).sum(-1))
        parent_dict[key] = {
            "pcl": padded_pcls,
            "pcl_is_valid": padding_mask,
            "pillar_coors": padded_pillar_coors,
        }
        pillar_coords_padding_mask = torch.logical_not(
            (padded_pillar_coors == -1).sum(-1)
        )
        assert torch.all(padding_mask == pillar_coords_padding_mask)
    if isinstance(value, abc.Mapping):
        for sub_key, sub_value in value.items():
            change_k_v(sub_key, value, sub_value)
    else:
        if key in ("pillar_coors", "pcl"):
            # these can only be processed jointly as dict
            pass

        elif key in ("moving_mask", "point_has_valid_flow_label"):
            parent_dict[key] = torch.nn.utils.rnn.pad_sequence(
                parent_dict[key],
                batch_first=True,
                padding_value=False,
            )
        elif all(isinstance(val, Shape) for val in value):
            parent_dict[key] = Shape.from_list_of_shapes(value)
        elif key in ("flow_ta_tb", "flow_tb_ta"):
            parent_dict[key] = torch.nn.utils.rnn.pad_sequence(
                parent_dict[key], batch_first=True, padding_value=np.nan
            )
        elif "pcl_full" in key or "lidar_rows" in key:
            parent_dict[key] = value
        elif "track_ids_mask" in key:
            parent_dict[key] = torch.nn.utils.rnn.pad_sequence(
                parent_dict[key],
                batch_first=True,
                padding_value=0,
            )
        else:
            parent_dict[key] = torch.stack(value, dim=0)


class LidarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        mode: str,
        use_geom_augmentation: bool,
        use_skip_frames: str,
        path_to_augmentation_db: Union[Path, str],
        path_to_mined_boxes_db: Union[Path, str],
        for_tracking: bool,
        shuffle: bool,
        need_flow: bool,
    ) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.cfg = cfg
        self.mode = mode
        self.use_geom_augmentation = use_geom_augmentation
        self.need_flow = need_flow
        self.path_to_augmentation_db = path_to_augmentation_db
        self.path_to_mined_boxes_db = path_to_mined_boxes_db
        self.box_augm_db = None
        self.mined_boxes_db = None
        self.dataset_sequence_is_messed_up = False
        self.loader_saver_helper = None  # need a seperate connection here

        assert self.mode in ("train", "val", "test"), self.mode
        if shuffle:
            # we only shuffle the val dataset so that we get the same samples during
            # online validation
            assert self.mode in (
                "val",
                "test",
            ), "please shuffle train using loader, not dataset!"
        assert not (
            (self.mode == "val" or self.mode == "test") and self.use_geom_augmentation
        ), "we don't want to augment validation data"

        if hasattr(cfg.data, "odom_source"):
            assert self.cfg.data.odom_source in (
                "kiss_icp",
                "gt",
            ), self.cfg.data.odom_source

        assert use_skip_frames in ("only", "never", "both"), use_skip_frames
        if for_tracking:
            assert (
                use_skip_frames == "never"
            ), "tracking and skipping breaks continuity of samples"
            assert not shuffle, "shuffling breaks continuity"
        self.for_tracking = for_tracking
        self.data_use_skip_frames = use_skip_frames

        (
            self.bev_range_m_np,
            self.img_grid_size_np,
            self.bev_pixel_per_meter_res_np,
            self.pcl_bev_center_coords_homog_np,
            torch_tensors,
        ) = get_bev_setup_params(cfg)

        for param_name, param in torch_tensors.items():
            setattr(self, param_name, param)

        if self.cfg.data.limit_pillar_height:
            self.height_range_m_np = np.array(
                self.cfg.data.pillar_height_range_m, np.float32
            )
        else:
            self.height_range_m_np = np.array([-np.inf, np.inf], np.float32)
        self.centermaps_output_grid_size = get_centermaps_output_grid_size(
            self.cfg, self.img_grid_size_np
        )

        if self.cfg.data.augmentation.boxes.active:
            self.box_augm_cfg = self.cfg.data.augmentation.boxes

        self.need_reverse_time_sample_data = self.cfg.network.name == "slim"

    def initialize_loader_saver_if_necessary(self):
        if self.loader_saver_helper is None:
            self.loader_saver_helper = CloudLoaderSaver()

    def initialize_dbs_if_necessary(self):
        # not to be called during __init__, but during first loading of data
        # since then it will be shared across all datalaoder threads
        # causing copy on read (refcount)
        # see github issue https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814

        if self.path_to_mined_boxes_db is not None and self.mined_boxes_db is None:
            self.load_set_mined_boxes_db_member()
        if self.path_to_augmentation_db is not None and self.box_augm_db is None:
            self.load_set_augmentation_db_member()

    def load_set_mined_boxes_db_member(self):
        self.mined_boxes_db = load_mined_boxes_db(self.path_to_mined_boxes_db)

    def load_set_augmentation_db_member(self):
        self.cfg.optimization.rounds.setdefault(
            "confidence_threshold_for_augmentation_strictness_factor", 1.0
        )
        self.box_augm_db = load_sanitize_box_augmentation_database(
            self.path_to_augmentation_db,
            self.cfg.optimization.rounds.confidence_threshold_mined_boxes
            * self.cfg.optimization.rounds.confidence_threshold_for_augmentation_strictness_factor,
        )

    def get_pred_flow_path(self):
        pred_flow_path = Path(
            self.cfg.data.paths[self.cfg.data.source].slim_flow[
                self.cfg.data.flow_source
            ]["local"]
        )

        return pred_flow_path

    def get_consecutive_sample_idxs_for_sequence(
        self,
        sequence_idx: int,
    ) -> List[LidarSample]:
        raise NotImplementedError("subclass needs to implement this")

    def __len__(self) -> int:
        return len(self.sample_files)

    def load_add_mined_boxes_to_sample_content(
        self,
        db_key: str,
        sample_content: Dict[str, torch.FloatTensor],
    ) -> Dict[str, torch.FloatTensor]:
        if db_key in self.mined_boxes_db:
            extra_boxes = self.mined_boxes_db[db_key]
            extra_boxes = Shape(**extra_boxes["raw_box"])
            box_confident_enough = (
                np.squeeze(extra_boxes.probs, axis=-1)
                >= self.cfg.optimization.rounds.confidence_threshold_mined_boxes
            )
            extra_boxes.valid = extra_boxes.valid & box_confident_enough
            extra_boxes = extra_boxes.drop_padding_boxes()
            extra_boxes.probs = np.ones_like(extra_boxes.probs)
        else:
            print(f"Found no mined boxes in sample {db_key}")
            if len(self.mined_boxes_db) > 0:
                some_key = next(iter(self.mined_boxes_db))
                assert type(some_key) is type(db_key), (
                    "type mismatch",
                    type(some_key),
                    type(db_key),
                )
                if len(some_key) != len(db_key):
                    print(
                        f"Warning: DB key size does not match! {len(some_key)} vs {len(db_key)}"
                    )
            extra_boxes = Shape.createEmpty()
        sample_content["mined"] = {"objects_t0": extra_boxes}

    def load_add_flow_to_sample_content(
        self,
        fname,
        sample_content,
        src_key,
        target_key,
        use_path_stem_only=True,
        specific_pred_flow_path: Path = None,
    ):
        if specific_pred_flow_path is None:
            if use_path_stem_only:
                specific_pred_flow_path = self.pred_flow_path.joinpath(
                    Path(fname).stem + ".npz"
                )
            else:
                specific_pred_flow_path = self.pred_flow_path.joinpath(
                    Path(fname).with_suffix(".npz")
                )

        if not specific_pred_flow_path.exists():
            print(
                f"Warning - file {specific_pred_flow_path} for flow source {self.cfg.data.flow_source} was not found!"
            )
            return
        pred_content = self.loader_saver_helper.load_sample(
            specific_pred_flow_path, np.load, allow_pickle=True
        )
        flow_source_grid_range_m_np = np.append(
            pred_content["bev_range_m"], np.array(1000.0)
        )
        if self.use_geom_augmentation and self.cfg.data.augmentation.active:
            max_allowed_radius = 0.5 * np.linalg.norm(
                0.5 * flow_source_grid_range_m_np[:2]
            )
        else:
            max_allowed_radius = np.min(0.5 * flow_source_grid_range_m_np[:2])
        assert (
            0.5 * np.max(self.bev_range_m_np) <= max_allowed_radius
        ), "cannot gather flow predictions outside of bev range - load predictiosn with larger bev or reduce cfg.data.bev_range_m"

        grid_size = np.append(pred_content["bev_raw_flow_t0_t1"].shape[:2], np.array(1))

        sample_content[self.cfg.data.flow_source] = {}
        for source_time_key, target_time_key in (
            (src_key, target_key),
            (target_key, src_key),
        ):
            voxel_coors, in_range = voxelize_pcl(
                sample_content[f"pcl_{source_time_key}"],
                flow_source_grid_range_m_np,
                grid_size,
            )
            pillar_coors = voxel_coors[:, :2]
            flow2d = np.nan * np.ones(pillar_coors.shape, dtype=np.float32)
            bev_flow = pred_content[f"bev_raw_flow_{source_time_key}_{target_time_key}"]

            # bev_flow_new = self.fill_in_flow_from_valid_pillar_neighbors(bev_flow)
            # bev_flow_new = self.fill_in_flow_from_valid_pillar_neighbors_array_style(
            #     bev_flow
            # )

            bev_flow = self.expand_valid_bev_flow_to_zero_flow_neighbor_pillars(
                bev_flow
            )

            in_range_pillar_coors = pillar_coors[in_range]
            flow2d[in_range] = bev_flow[
                in_range_pillar_coors[..., 0], in_range_pillar_coors[..., 1]
            ]
            flow2d[~in_range] = np.mean(flow2d[in_range], axis=0)
            # some NANs seem to have slipped through
            # probably we are using another voxelize method somewhere else
            # so just use zeros for default flow
            # num_bad_flows = np.count_nonzero(~np.isfinite(flow2d[in_range]))
            # if num_bad_flows > 0:
            #     print(
            #         "{0}: Found {1} NANs in sample {2}".format(
            #             datetime.now(), num_bad_flows, fname
            #         )
            #     )
            #     flow2d = np.where(np.isfinite(flow2d), flow2d, 0.0)

            flow3d = np.concatenate([flow2d, np.zeros_like(flow2d[:, :1])], axis=-1)
            sample_content[self.cfg.data.flow_source][
                f"flow_{source_time_key}_{target_time_key}"
            ] = flow3d

    def expand_valid_bev_flow_to_zero_flow_neighbor_pillars(self, bev_flow: np.ndarray):
        # this should fix any small numerical imprecisions during pillarization
        # but not change flow of occupied cells

        pillar_has_zero_flow = (bev_flow == 0.0).all(axis=-1)
        bev_flow = np.ma.masked_array(
            bev_flow,
            # true indicates invalid data
            mask=np.stack([pillar_has_zero_flow, pillar_has_zero_flow], axis=-1),
        )
        for shift in (-1, 1):
            for axis in (0, 1):
                bev_flow_shifted = np.roll(bev_flow, shift=shift, axis=axis)
                idx = ~bev_flow_shifted.mask * bev_flow.mask
                bev_flow[idx] = bev_flow_shifted[idx]
        bev_flow = bev_flow.filled(fill_value=0.0)
        # pillar_has_zero_flow = (bev_flow == 0.0).all(axis=-1)
        return bev_flow

    def kitti_extract_boxes_for_timestamp(self, sample_content, src_key):
        box_key = f"objects_{src_key}"
        if (
            box_key in sample_content["gt"]
            and sample_content["gt"][box_key]["poses"].shape[0] > 0
        ):
            objects = sample_content["gt"][box_key]
            sensor_T_box = objects["poses"]
            obj_dims = objects["size"]
            obj_pos_ta = sensor_T_box[:, 0:3, 3]
            obj_rot_ta = np.stack(
                [decompose_matrix(trafo)[2][2] for trafo in sensor_T_box],
                axis=0,
            )[..., None]
            obj_probs_ta = np.ones_like(obj_rot_ta)
            obj_velo_ta = np.zeros_like(obj_probs_ta)
            class_ids = self.get_label_idxs_from_label_name(objects["category"])[
                ..., None
            ]
            gt_boxes_ta = Shape(
                pos=obj_pos_ta,
                dims=obj_dims,
                rot=obj_rot_ta,
                probs=obj_probs_ta,
                class_id=class_ids,
                valid=np.ones_like(np.squeeze(obj_probs_ta, axis=-1), dtype=bool),
                velo=obj_velo_ta,
            )
            class_names = objects["category"]

        else:
            gt_boxes_ta = Shape.createEmpty()
            class_names = np.array([], dtype=str)
        return gt_boxes_ta, class_names

    def get_label_idxs_from_label_name(self, label_names: str) -> np.ndarray:
        del label_names
        raise NotImplementedError("subclass needs to implement this")

    def extract_boxes_for_timestamp(
        self,
        sample_content: Dict[str, np.ndarray],
        src_key: str,
        target_key: str,
    ) -> Shape:
        raise NotImplementedError("this is dataset specific")

    def assemble_sample_data(
        self,
        sample_content: Dict[str, np.ndarray],
        src_key: str,
        target_key: str,
        src_trgt_time_delta_s: float,
        remap_src_target_keys_to_ta_tb=True,
    ):
        if f"pcl_{src_key}" not in sample_content:
            return {}

        if self.cfg.network.name in (
            "transfusion",
            "centerpoint",
            "pointrcnn",
            "pointpillars",
            "slim",
            "echo_gt",
            "flow_cluster_detector",
        ):
            sample_content = self.pillarize_points_remove_ground_add_bev_ghm_occupancy(
                sample_content, src_key=src_key, target_key=target_key
            )
            self.add_bev_flow(sample_content, "gt", src_key, target_key)
            if (
                self.need_flow
                and self.cfg.data.flow_source != "gt"
                and self.mode == "train"
            ):
                self.add_bev_flow(
                    sample_content, self.cfg.data.flow_source, src_key, target_key
                )
            pcl_homog = homogenize_pcl(sample_content[f"pcl_{src_key}"]["pcl"][:, :3])
            if (
                f"odom_{target_key}_{src_key}" in sample_content["gt"]
                and f"flow_{src_key}_{target_key}" in sample_content["gt"]
            ):
                sample_content["gt"]["moving_mask"] = (
                    np.linalg.norm(
                        np.einsum(
                            "ij,kj->ki",
                            sample_content["gt"][f"odom_{target_key}_{src_key}"]
                            - np.eye(4),
                            pcl_homog,
                        )[..., 0:3]
                        - sample_content["gt"][f"flow_{src_key}_{target_key}"],
                        axis=-1,
                    )
                    > self.cfg.data.non_rigid_flow_threshold_mps * src_trgt_time_delta_s
                )
            if "mined" in sample_content:
                mined_boxes = sample_content["mined"].pop(f"objects_{src_key}", None)
                if mined_boxes is not None:
                    sample_content["mined"]["boxes"] = mined_boxes
                    if self.cfg.network.name not in ("pointrcnn", "pointpillars"):
                        centermaps_ta = draw_heat_regression_maps(
                            mined_boxes,
                            self.centermaps_output_grid_size,
                            self.bev_range_m_np,
                            per_obj_prob_scale=np.ones_like(mined_boxes.probs),
                            box_pred_cfg=self.cfg.box_prediction,
                        )
                    else:
                        centermaps_ta = {}
                    for k, v in centermaps_ta.items():
                        sample_content["mined"][f"centermaps_{k}"] = v
                sample_content["mined"].pop(f"objects_{target_key}", None)

            gt_boxes, gt_class_names = self.extract_boxes_for_timestamp(
                sample_content,
                src_key,
                target_key,
            )
            gt_object_is_movable = np.array(
                [self.object_is_movable(el) for el in gt_class_names], dtype=bool
            )
            gt_boxes.valid = gt_boxes.valid & gt_object_is_movable
            gt_boxes = gt_boxes.drop_padding_boxes()
            homog_pcl = homogenize_pcl(
                sample_content[f"pcl_full_no_ground_{src_key}"][:, :3]
            )
            (
                sample_content["gt"]["boxes_nusc"],
                gt_box_has_points,
            ) = self.filter_objects_to_bev_non_empty(
                deepcopy(gt_boxes),
                homog_pcl,
                filter_range_m=50.0,
                filter_bev=False,
            )

            gt_boxes, _ = self.filter_objects_to_bev_non_empty(
                gt_boxes, homog_pcl, box_has_points_inside=gt_box_has_points
            )

            obj_speed_ta_tb = np.linalg.norm(gt_boxes.velo, axis=-1)
            sample_content["gt"]["boxes"] = gt_boxes
            sample_content["gt"].pop("objects", None)

            sample_content["gt"][
                "point_has_valid_flow_label"
            ] = self.get_has_valid_scene_flow_label(sample_content, src_key)

            if (
                self.cfg.network.name in ("centerpoint", "transfusion")
                and self.cfg.loss.supervised.centermaps.active
            ):
                per_obj_prob_heatmap_scale = self.select_centermaps_target_confidence(
                    gt_boxes, obj_speed_ta_tb
                )
                centermaps_ta = draw_heat_regression_maps(
                    gt_boxes,
                    self.centermaps_output_grid_size,
                    self.bev_range_m_np,
                    per_obj_prob_scale=per_obj_prob_heatmap_scale,
                    box_pred_cfg=self.cfg.box_prediction,
                )
                for k, v in centermaps_ta.items():
                    sample_content["gt"][f"centermaps_{k}"] = v

                if f"kitti_ignore_region_boxes_{src_key}" in sample_content["gt"]:
                    ignore_boxes = sample_content["gt"][
                        f"kitti_ignore_region_boxes_{src_key}"
                    ]

                    ignore_region_is_true_mask = (
                        self.create_true_where_ignore_region_mask(
                            ignore_boxes,
                        )
                    )

                    sample_content["gt"][
                        "ignore_region_is_true_mask"
                    ] = ignore_region_is_true_mask

        elif self.cfg.network.name in ("ogc",):
            raise AssertionError(
                "you do not filter objects to be in BEV for OGC -> problematic?"
            )
        else:
            raise NotImplementedError(self.cfg.network.name)
        if remap_src_target_keys_to_ta_tb:
            sample_content = change_dict_keys(
                sample_content, lambda x: x.replace(src_key, "ta")
            )
            sample_content = change_dict_keys(
                sample_content, lambda x: x.replace(target_key, "tb")
            )
        sample_content["src_trgt_time_delta_s"] = np.array(src_trgt_time_delta_s)
        drop_keys = (
            "flow_tb_ta",
            "is_ground_tb",
            "is_ground_tx",
            "semantics_ta",
            "semantics_tb",
            "is_ground_ta",
        )
        for drop_key in drop_keys:
            sample_content["gt"].pop(drop_key, None)
        return sample_content

    def select_centermaps_target_confidence(self, gt_boxes, obj_speed_ta_tb):
        if self.cfg.loss.supervised.centermaps.confidence_target == "gaussian":
            per_obj_prob_heatmap_scale = np.ones_like(gt_boxes.probs)
        else:
            raise NotImplementedError(
                self.cfg.loss.supervised.centermaps.confidence_target
            )

        return per_obj_prob_heatmap_scale

    def get_has_valid_scene_flow_label(
        self, _sample_content: Dict[str, np.ndarray], _src_key: str
    ):
        raise NotImplementedError("subclass needs to implement this!")

    def create_true_where_ignore_region_mask(self, ignore_boxes: Shape):
        if torch.is_tensor(ignore_boxes.pos):
            ignore_boxes_np = ignore_boxes.clone().detach().cpu().numpy()
        else:
            ignore_boxes_np = ignore_boxes
        ignore_region_is_true_mask = np.zeros(
            self.centermaps_output_grid_size, dtype=bool
        )
        for ignore_box in ignore_boxes_np:
            this_box_mask_float, _ = render_hard_kabsch_mask(
                box_x=ignore_box.pos[0],
                box_y=ignore_box.pos[1],
                box_len=ignore_box.dims[0],
                box_w=ignore_box.dims[1],
                box_theta=ignore_box.rot[0],
                bev_range_x=self.bev_range_m_np[0],
                bev_range_y=self.bev_range_m_np[1],
                img_shape=self.centermaps_output_grid_size,
            )
            ignore_region_is_true_mask = ignore_region_is_true_mask | (
                this_box_mask_float > 0.5
            )
        return ignore_region_is_true_mask

    def move_keys_to_subdict(
        self,
        sample_content: Dict[str, np.ndarray],
        move_these_keys=(
            "objects",
            "odom",
            "flow",
            "semantics",
            "is_ground",
            "track_ids_mask",
        ),
        subdict_target_key="gt",
        drop_substr_from_moved_keys="",
    ):
        restructured_smaple_content = {}

        restructured_smaple_content[subdict_target_key] = {}
        for k in sample_content:
            move_key = False
            for move_key_to_gt in move_these_keys:
                if move_key_to_gt in k:
                    move_key = True
                    break
            if move_key:
                restructured_smaple_content[subdict_target_key][
                    k.replace(drop_substr_from_moved_keys, "")
                ] = sample_content[k]

            else:
                restructured_smaple_content[k] = sample_content[k]
        return restructured_smaple_content

    def voxelize_sample(self, pcl_np):
        bev_range_m_np = np.append(self.bev_range_m_np, np.array(1000.0))

        grid_size = np.append(self.img_grid_size_np, np.array(1))
        pointwise_voxel_coords_all_pts, point_is_in_range = voxelize_pcl(
            pcl_np, bev_range_m_np, grid_size
        )
        point_is_in_pillar_limits = (self.height_range_m_np[0] < pcl_np[:, 2]) & (
            pcl_np[:, 2] < self.height_range_m_np[1]
        )
        point_is_in_range = point_is_in_range & point_is_in_pillar_limits
        pointwise_pillar_coors_all_pts = pointwise_voxel_coords_all_pts[..., 0:2]
        return pointwise_pillar_coors_all_pts, point_is_in_range

    def drop_intensities_from_pcls_in_sample(self, sample_content):
        for time_key in ("t0", "t1", "t2"):
            pcl_time_key = f"pcl_{time_key}"
            if pcl_time_key in sample_content:
                assert sample_content[pcl_time_key].shape[-1] == 4, sample_content[
                    pcl_time_key
                ].shape[-1]
                sample_content[pcl_time_key] = sample_content[pcl_time_key][:, :3]

    def drop_points_on_kitti_vehicle(self, sample_content, src_key, target_key):
        half_vehicle_size = 0.5 * np.array([5.0, 2.5, 3.0])
        for sk, tk in ((src_key, target_key), (target_key, src_key)):
            downsample_keys = self.get_sample_data_downsample_keys(sk, tk)
            if f"pcl_{sk}" in sample_content:
                is_vehicle_point = (
                    np.abs(sample_content[f"pcl_{sk}"][:, :3])
                    < half_vehicle_size[None, ...]
                ).all(axis=-1)
                sample_content = downsample_dict(
                    sample_content, ~is_vehicle_point, downsample_keys
                )

        return sample_content

    def filter_objects_to_bev_non_empty(
        self,
        objects: Shape,
        pcl_homog: np.ndarray,
        filter_bev: bool = True,
        filter_range_m: float = None,
        box_has_points_inside: np.ndarray = None,
    ):
        if np.count_nonzero(objects.valid) > 0:
            if box_has_points_inside is None:
                point_is_in_box = get_points_in_boxes_mask(
                    objects, pcl_homog, use_double_precision=False
                )
                box_has_points_inside = point_is_in_box.sum(axis=0) > 0
            else:
                # use precomputed box_has_points_inside
                assert box_has_points_inside.shape == objects.shape, (
                    box_has_points_inside.shape,
                    objects.shape,
                )
            objects.valid = objects.valid & box_has_points_inside
            objects = objects.drop_padding_boxes()

            if np.count_nonzero(objects.valid) > 0:
                if filter_bev:
                    object_is_in_bev = self.object_is_in_bev_range(objects)
                else:
                    object_is_in_bev = np.ones_like(objects.valid)

                if filter_range_m is not None:
                    object_is_in_range = (
                        np.linalg.norm(objects.pos, axis=-1) < filter_range_m
                    )
                else:
                    object_is_in_range = np.ones_like(objects.valid)

                keep_boxes_mask = np.logical_and.reduce(
                    (
                        objects.valid,
                        object_is_in_bev,
                        object_is_in_range,
                    )
                )
                objects.valid = keep_boxes_mask
                objects = objects.drop_padding_boxes()

        return objects, box_has_points_inside

    def pillarize_points_remove_ground_add_bev_ghm_occupancy(
        self, sample_content, src_key, target_key
    ):
        for sk, tk in ((src_key, target_key), (target_key, src_key)):
            if f"pcl_{sk}" in sample_content:
                sample_content[f"pcl_full_w_ground_{sk}"] = np.copy(
                    sample_content[f"pcl_{sk}"]
                )
                no_ground_pcl_sample = self.remove_ground_points_from_sample(
                    {
                        f"pcl_{sk}": np.copy(sample_content[f"pcl_{sk}"]),
                        "gt": {
                            f"is_ground_{sk}": sample_content["gt"][f"is_ground_{sk}"]
                        },
                    },
                    sk,
                    tk,
                    downsample_keys=(f"pcl_{sk}",),
                )
                sample_content[f"pcl_full_no_ground_{sk}"] = no_ground_pcl_sample[
                    f"pcl_{sk}"
                ]
                sample_content = self.pillarize_bev(sample_content, sk, tk)
                sample_content = self.remove_ground_points_from_sample(
                    sample_content,
                    sk,
                    tk,
                )
                flow_key = f"flow_{sk}_{tk}"
                if flow_key in sample_content["gt"]:
                    assert (
                        sample_content[f"pcl_{sk}"].shape[0]
                        == sample_content["gt"][f"flow_{sk}_{tk}"].shape[0]
                    )

                # we now have 3 types on the point cloud
                # pcl_ta:                   does NOT have ground, is limited to BEV
                #                           this corresponds to the additional attributes
                #                           * pillar_coors_ta
                #                           * flow_ta_tb
                #                           * lidar_rows_ta
                # pcl_full_w_ground_ta:     does have ground, is NOT limited to BEV
                # pcl_full_no_ground_ta:    does NOT have ground, is NOT limited to BEV
                self.add_bev_ground_height_occupancy_maps(sample_content, sk)
                self.move_pcl_pillar_coors_to_subdict(sample_content, sk)
        return sample_content

    def move_pcl_pillar_coors_to_subdict(self, sample_content: Dict, sk: str):
        pcl = sample_content.pop(f"pcl_{sk}")
        sample_content[f"pcl_{sk}"] = {
            "pcl": pcl,
            "pillar_coors": sample_content[f"pillar_coors_{sk}"],
        }
        del sample_content[f"pillar_coors_{sk}"]

    def get_object_velocity_in_obj_coords(self, odom_ta_tb, obj_pose_ta, obj_pose_tb):
        dyn_flow_trafo = obj_pose_tb @ np.linalg.inv(obj_pose_ta) - np.eye(4)
        stat_flow_trafo = np.linalg.inv(odom_ta_tb) - np.eye(4)
        obj_pos_ta = obj_pose_ta[:, 0:2, 3]

        obj_norig_flow_trafo = dyn_flow_trafo - stat_flow_trafo
        obj_pos_homog_ta = np.concatenate(
            [
                obj_pos_ta,
                np.zeros_like(obj_pos_ta[..., [0]]),
                np.ones_like(obj_pos_ta[..., [0]]),
            ],
            axis=-1,
        )
        norig_flow_sensor_cosy = np.einsum(
            "nij,nj->ni", obj_norig_flow_trafo, obj_pos_homog_ta
        )[..., :3]

        norig_flow_sensor_cosy_homog = np.concatenate(
            [
                norig_flow_sensor_cosy,
                np.zeros_like(norig_flow_sensor_cosy[..., [0]]),
            ],
            axis=-1,
        )
        norig_flow_obj_cosy = np.einsum(
            "nij,nj->ni", obj_pose_ta, norig_flow_sensor_cosy_homog
        )[..., :3]

        return norig_flow_obj_cosy

    def pillarize_bev(
        self,
        sample_content,
        src_key,
        target_key,
    ):
        downsample_keys_t0 = self.get_sample_data_downsample_keys(src_key, target_key)
        pcl_t0 = sample_content[f"pcl_{src_key}"]

        pillar_coors_t0, point_is_in_range_t0 = self.voxelize_sample(pcl_t0)
        sample_content[f"pillar_coors_{src_key}"] = pillar_coors_t0

        sample_content = downsample_dict(
            sample_content, point_is_in_range_t0, downsample_keys_t0
        )

        return sample_content

    def remove_ground_points_from_sample(
        self, sample_content, src_key, target_key, downsample_keys=None
    ):
        pcl_t0 = sample_content[f"pcl_{src_key}"]
        is_ground_jpc = sample_content["gt"][f"is_ground_{src_key}"]

        is_ground_height_based = infer_ground_label_using_cone(
            pcl_t0,
            cone_z_threshold__m=self.cfg.data.ground_height_map.ground_threshold,
        )

        # is_ground_agreement = is_ground & is_ground_height_based
        is_ground = is_ground_jpc | is_ground_height_based
        if downsample_keys is None:
            downsample_keys_t0 = self.get_sample_data_downsample_keys(
                src_key, target_key
            )
        else:
            downsample_keys_t0 = downsample_keys
        sample_content = downsample_dict(sample_content, ~is_ground, downsample_keys_t0)
        return sample_content

    @staticmethod
    @lru_cache(maxsize=10)
    def get_sample_data_downsample_keys(src_key, target_key):
        return (
            f"flow_{src_key}_{target_key}",
            f"pcl_{src_key}",
            f"lidar_rows_{src_key}",
            f"semantics_{src_key}",
            f"pillar_coors_{src_key}",
            f"is_ground_{src_key}",
            f"track_ids_mask_{src_key}",
        )

    def add_bev_flow(
        self, sample_content, point_flow_src_key, time_src_key, time_target_key
    ):
        flow_key = f"flow_{time_src_key}_{time_target_key}"
        if flow_key in sample_content[point_flow_src_key]:
            flow = sample_content[point_flow_src_key][flow_key]
            flow_gt_bev = scatter_mean_nd_numpy(
                indices=sample_content[f"pcl_{time_src_key}"]["pillar_coors"],
                shape=tuple(self.img_grid_size_np) + (3,),
                updates=flow,
            )
            sample_content[point_flow_src_key][
                f"flow_bev_{time_src_key}_{time_target_key}"
            ] = flow_gt_bev.astype(np.float32)

    def add_bev_ground_height_occupancy_maps(self, sample_content, src_key):
        occupancy_map_f32 = np.zeros(self.img_grid_size_np)
        occupancy_map_f32[
            sample_content[f"pillar_coors_{src_key}"][:, 0],
            sample_content[f"pillar_coors_{src_key}"][:, 1],
        ] = 1.0
        sample_content[f"occupancy_f32_{src_key}"] = occupancy_map_f32[
            None, ...
        ].astype(np.float32)

    def object_is_movable(self, obj) -> bool:
        raise NotImplementedError("subclass needs to implement this!")

    def object_is_in_bev_range(self, obj):
        x_in_range = 0.5 * self.bev_range_m_np[0] >= np.abs(obj.pos[:, 0])
        y_in_range = 0.5 * self.bev_range_m_np[1] >= np.abs(obj.pos[:, 1])
        return x_in_range & y_in_range

    def drop_unused_timed_keys_from_sample(
        self, sample_content, src_key, target_key, delete_target_key
    ):
        delete_no_matter_what = [
            "lidar_intensities",
            "track_ids_mask",
            "semantics",
        ]
        for key in list(sample_content.keys()):
            if any([el in key for el in delete_no_matter_what]):
                sample_content.pop(key, None)

        drop_these_keys = [
            "meta_data_t0",
            f"flow_{src_key}_{delete_target_key}",
            f"flow_{delete_target_key}_{src_key}",
            f"flow_{target_key}_{delete_target_key}",
            f"flow_{delete_target_key}_{target_key}",
            f"pcl_{delete_target_key}",
            f"is_ground_{delete_target_key}",
            f"odom_{src_key}_{delete_target_key}",
            f"odom_{delete_target_key}_{src_key}",
            f"odom_{target_key}_{delete_target_key}",
            f"odom_{delete_target_key}_{target_key}",
            f"objects_{delete_target_key}",
        ]

        if self.pure_inference_mode:
            drop_these_additional_keys = [
                f"flow_{src_key}_{target_key}",
                f"flow_{target_key}_{src_key}",
                f"pcl_{target_key}",
                f"is_ground_{target_key}",
                f"semantics_{target_key}",
                f"odom_{src_key}_{target_key}",
                f"odom_{target_key}_{src_key}",
            ]
            drop_these_keys += drop_these_additional_keys
        for delete_key in drop_these_keys:
            sample_content.pop(delete_key, None)

    def add_reverse_odometry_to_sample(self, sample_content):
        # add all missing reverse odometries:
        odom_sources = {"gt", self.cfg.data.odom_source}
        for odom_source in odom_sources:
            add_odom_target_times = ("t1", "t2", "tx")
            for src_key in ("t0", "t1", "t2"):
                for odom_target_time in add_odom_target_times:
                    odom_tgt_src_key = f"odom_{odom_target_time}_{src_key}"
                    odom_src_tgt_key = f"odom_{src_key}_{odom_target_time}"
                    if (
                        odom_tgt_src_key not in sample_content[odom_source]
                        and odom_src_tgt_key in sample_content[odom_source]
                    ):
                        sample_content[odom_source][odom_tgt_src_key] = np.linalg.inv(
                            sample_content[odom_source][odom_src_tgt_key]
                        )

    def augment_sample_content(
        self,
        sample_content,
        src_key,
        target_key,
        dataset_name: str,
    ):
        assert (
            not self.for_tracking
        ), "different calls will use different augm transforms, this breaks everything!"
        odom_key_src_trgt = f"odom_{src_key}_{target_key}"
        key_flow_src_trgt = f"flow_{src_key}_{target_key}"
        key_flow_trgt_src = f"flow_{target_key}_{src_key}"
        pcl_src_key = f"pcl_{src_key}"
        pcl_trgt_key = f"pcl_{target_key}"
        pcl_x_key = "pcl_tx"
        odom_key_t0_tx = "odom_t0_tx"
        odom_key_tx_t0 = "odom_tx_t0"
        assert (
            f"pcl_full_no_ground_{src_key}" not in sample_content
        ), "will not be augmented!"
        assert (
            f"pcl_full_w_ground_{src_key}" not in sample_content
        ), "will not be augmented!"
        assert (
            f"pcl_full_no_ground_{target_key}" not in sample_content
        ), "will not be augmented!"
        assert (
            f"pcl_full_w_ground_{target_key}" not in sample_content
        ), "will not be augmented!"
        # # before:
        # gt_flow_bef = np.copy(sample_content["gt"][key_flow_src_trgt])
        # slim_flow_bef = np.copy(sample_content["slim_bev_120m"][key_flow_src_trgt])
        # is_nan = np.any(~np.isfinite(slim_flow_bef), axis=-1)
        # epe_before = np.linalg.norm(slim_flow_bef - gt_flow_bef, axis=-1)[
        #     ~is_nan
        # ].mean()

        augSensor_T_sensor = get_augmentation_transform(
            max_symm_rot_deg=self.cfg.data.augmentation.rotation.max_rot_deg,
            max_sensor_pos_offset_m=self.cfg.data.augmentation.translation.max_sensor_pos_offset_m,
            max_xy_scale_delta=None,
        )
        pcl_src = sample_content[pcl_src_key]
        sample_content[pcl_src_key] = self.transform_pcl_maybe_with_intensity(
            pcl_src, augSensor_T_sensor
        )
        pcl_trgt = sample_content[pcl_trgt_key]
        sample_content[pcl_trgt_key] = self.transform_pcl_maybe_with_intensity(
            pcl_trgt, augSensor_T_sensor
        )
        if pcl_x_key in sample_content:
            assert "flow_t0_tx" not in sample_content, "not augmented, add below!"
            sample_content[pcl_x_key] = self.transform_pcl_maybe_with_intensity(
                sample_content[pcl_x_key], augSensor_T_sensor
            )
        for odom_source in {"gt", self.cfg.data.odom_source}:
            if odom_key_t0_tx in sample_content[odom_source]:
                sample_content[odom_source][odom_key_t0_tx] = (
                    augSensor_T_sensor
                    @ sample_content[odom_source][odom_key_t0_tx]
                    @ np.linalg.inv(augSensor_T_sensor)
                )
                sample_content[odom_source][odom_key_tx_t0] = np.linalg.inv(
                    sample_content[odom_source][odom_key_t0_tx]
                )
            if odom_key_src_trgt in sample_content[odom_source]:
                sample_content[odom_source][odom_key_src_trgt] = (
                    augSensor_T_sensor
                    @ sample_content[odom_source][odom_key_src_trgt]
                    @ np.linalg.inv(augSensor_T_sensor)
                )
                sample_content[odom_source][
                    f"odom_{target_key}_{src_key}"
                ] = np.linalg.inv(sample_content[odom_source][odom_key_src_trgt])
        for data_category in {"gt", self.cfg.data.flow_source}:
            if data_category in sample_content:
                if key_flow_src_trgt in sample_content[data_category]:
                    sample_content[data_category][key_flow_src_trgt] = np.einsum(
                        "ij,nj->ni",
                        augSensor_T_sensor,
                        homogenize_flow(
                            sample_content[data_category][key_flow_src_trgt]
                        ),
                    )[..., 0:3].astype(np.float32)

                if key_flow_trgt_src in sample_content[data_category]:
                    sample_content[data_category][key_flow_trgt_src] = np.einsum(
                        "ij,nj->ni",
                        augSensor_T_sensor,
                        homogenize_flow(
                            sample_content[data_category][key_flow_trgt_src]
                        ),
                    )[..., 0:3].astype(np.float32)
        if dataset_name in ("kitti", "nuscenes"):
            if "objects" in sample_content["gt"]:
                for obj in sample_content["gt"]["objects"]:
                    for timestamp in ("t0", "t1", "t2"):
                        pose_key = f"pose_{timestamp}"
                        if pose_key in obj:
                            obj[pose_key] = augSensor_T_sensor @ obj[pose_key]
        elif dataset_name == "kitti_object":
            for timestamp in ("t0", "t1", "t2"):
                obj_time_key = f"objects_{timestamp}"
                if obj_time_key in sample_content["gt"]:
                    sample_content["gt"][obj_time_key]["poses"] = (
                        augSensor_T_sensor @ sample_content["gt"][obj_time_key]["poses"]
                    )
            for timestamp in ("t0", "t1", "t2"):
                ignore_box_time_key = f"kitti_ignore_region_boxes_{timestamp}"
                if ignore_box_time_key in sample_content["gt"]:
                    ignore_region_box_poses = sample_content["gt"][
                        ignore_box_time_key
                    ].get_poses()
                    updated_ignore_box_poses = (
                        augSensor_T_sensor @ ignore_region_box_poses
                    )
                    aug_ignore_pos, augm_ignore_yaw = torch_decompose_matrix(
                        torch.from_numpy(updated_ignore_box_poses)
                    )
                    sample_content["gt"][
                        ignore_box_time_key
                    ].pos = aug_ignore_pos.numpy()
                    sample_content["gt"][
                        ignore_box_time_key
                    ].rot = augm_ignore_yaw.numpy()

        elif dataset_name in ("waymo", "av2"):
            self.augment_objects_from_category_with_trafo(
                sample_content, augSensor_T_sensor, "gt"
            )
        else:
            raise NotImplementedError(dataset_name)
        if "mined" in sample_content:
            self.augment_objects_from_category_with_trafo(
                sample_content, augSensor_T_sensor, "mined"
            )
        # # after
        # gt_flow = sample_content["gt"][key_flow_src_trgt]
        # slim_flow = sample_content["slim_bev_120m"][key_flow_src_trgt]
        # is_nan = np.any(~np.isfinite(slim_flow), axis=-1)
        # epe_after = np.linalg.norm(slim_flow - gt_flow, axis=-1)[~is_nan].mean()
        # assert np.allclose(epe_before, epe_after)

    def augment_objects_from_category_with_trafo(
        self, sample_content, augSensor_T_sensor, category_key
    ):
        for timestamp in ("t0", "t1", "t2"):
            # assert (
            #     f"boxes_{timestamp}" not in sample_content[category_key]
            # ), sample_content[category_key].keys()

            possible_obj_keys = (f"objects_{timestamp}", f"boxes_{timestamp}")
            if all(
                [
                    obj_key in sample_content[category_key]
                    for obj_key in possible_obj_keys
                ]
            ):
                print("WARNING: Multiple Box/object istances found- check this!")
            for obj_key in possible_obj_keys:
                if obj_key in sample_content[category_key]:
                    objs = sample_content[category_key][obj_key]
                    obj_poses = objs.get_poses()
                    augm_obj_poses = augSensor_T_sensor @ obj_poses
                    # augm_angles, augm_transl = zip(*[decompose_matrix(trafo)[2:3] for trafo in augm_obj_poses])
                    transl, rots = torch_decompose_matrix(
                        torch.from_numpy(augm_obj_poses)
                    )
                    objs.pos = transl.numpy()
                    objs.rot = rots.numpy()
                    sample_content[category_key][obj_key] = objs

    def transform_pcl_maybe_with_intensity(
        self, pcl_src: np.ndarray, augSensor_T_sensor: np.ndarray
    ):
        assert augSensor_T_sensor.shape == (4, 4), augSensor_T_sensor.shape
        pcl_has_intensity = pcl_src.shape[-1] == 4
        if pcl_has_intensity:
            intensity = pcl_src[:, [-1]]
            pcl_3d = pcl_src[:, :3]
        else:
            pcl_3d = pcl_src
        transformed_pcl_3d = np.einsum(
            "ij,nj->ni", augSensor_T_sensor, homogenize_pcl(pcl_3d)
        )[..., 0:3].astype(np.float32)
        if pcl_has_intensity:
            final_transformed_pcl = np.concatenate(
                [transformed_pcl_3d, intensity], axis=-1
            )
        else:
            final_transformed_pcl = transformed_pcl_3d
        return final_transformed_pcl

    def select_time_keys(self):
        src_key = "t0"
        delete_map = {
            "t1": "t2",
            "t2": "t1",
        }  # gives you the key to delete for your chosen target frame
        if self.mode == "train":
            if self.data_use_skip_frames == "only":
                target_key = "t2"
            elif self.data_use_skip_frames == "never":
                target_key = "t1"
            elif self.data_use_skip_frames == "both":
                target_key = np.random.choice(list(delete_map.keys()))
            else:
                raise NotImplementedError(self.data_use_skip_frames)

        elif self.mode == "val" or self.mode == "test":
            target_key = "t1"
        else:
            raise NotImplementedError(self.mode)

        if self.pure_inference_mode:
            target_key = "t1"

        time_delta_between_frames_sec = {"t1": 0.1, "t2": 0.2}[target_key]

        delete_target_key = delete_map[target_key]
        return src_key, target_key, delete_target_key, time_delta_between_frames_sec

    @torch.no_grad()
    def get_motion_based_centermaps(
        self, sample_data_a: Dict[str, torch.FloatTensor]
    ) -> Dict[str, torch.FloatTensor]:
        assert self.cfg.loss.supervised.centermaps.active
        assert not (
            self.cfg.loss.supervised.hungarian.active
            and self.cfg.network.name == "centerpoint"
        )

        motion_based_centermaps = {}
        if self.cfg.data.flow_source not in sample_data_a:
            # we only have slim flow for train dataset
            assert self.mode == "val", ("need train targets for mode: ", self.mode)
            return motion_based_centermaps
        return motion_based_centermaps

    def create_augmented_sample_from_box_snippet_db(
        self, src_trgt_time_delta_s, sample_data_ta, prediscovered_boxes: Shape = None
    ):
        num_augm_objs = np.random.randint(
            low=1, high=self.box_augm_cfg.max_num_objs + 1
        )

        bev_occupancy_bool_map = torch.zeros(tuple(self.img_grid_size_np), dtype=bool)
        pillar_coors = sample_data_ta["pcl_ta"]["pillar_coors"].long()
        bev_occupancy_bool_map[
            pillar_coors[:, 0],
            pillar_coors[:, 1],
        ] = True
        size_single_pillar_m = 1 / self.bev_pixel_per_meter_res_np
        min_obj_center_dist_from_occupied_pillars_m = self.box_augm_cfg.setdefault(
            "min_obj_center_dist_from_occupied_pillars_m", 2.0
        )
        assert size_single_pillar_m.shape == (
            2,
        ), "mean calculation will not work below"
        pixel_dilation_radius = int(
            min_obj_center_dist_from_occupied_pillars_m / size_single_pillar_m.mean()
        )
        footprint = disk(radius=max(3, pixel_dilation_radius))
        valid_augm_box_loc_mask = ~binary_dilation(
            bev_occupancy_bool_map.numpy(), footprint=footprint
        )
        allowed_box_locs = self.pcl_bev_center_coords_homog_np[valid_augm_box_loc_mask]
        augm_loc_idxs = np.random.choice(
            np.arange(allowed_box_locs.shape[0]),
            size=num_augm_objs,
            replace=False,
        )
        augm_box_locations_xy = torch.from_numpy(allowed_box_locs[augm_loc_idxs][:, :2])
        augm_box_locations_xy += (
            0.5 - torch.rand_like(augm_box_locations_xy)
        ) * size_single_pillar_m
        obj_idxs = np.random.choice(
            np.arange(len(self.box_augm_db["pcl_in_box_cosy"])),
            size=num_augm_objs,
            replace=True,
        )
        box_dims = self.box_augm_db["boxes"][obj_idxs].dims
        box_z_pos_old = self.box_augm_db["boxes"][obj_idxs].pos[..., [2]]
        box_z_pos_new = 0.5 * (torch.rand((num_augm_objs, 1)) - 0.5) + box_z_pos_old

        box_rot = 2 * np.pi * (torch.rand((num_augm_objs, 1)) - 0.5)
        box_pos = torch.cat(
            [augm_box_locations_xy, box_z_pos_new],
            dim=-1,
        )
        sensor_Trand_box = torch_compose_matrix(
            t_x=box_pos[None, ..., 0],
            t_y=box_pos[None, ..., 1],
            theta_z=box_rot[None, ..., 0],
            t_z=None,
        )[0].numpy()

        extra_boxes = Shape(
            pos=box_pos,
            dims=box_dims,
            rot=box_rot,
            probs=torch.ones_like(box_rot),
        )
        extra_pcl = []
        extra_flows = []

        for i, obj_idx in enumerate(obj_idxs):
            pcl_in_box = np.copy(self.box_augm_db["pcl_in_box_cosy"][obj_idx])
            if self.box_augm_cfg.use_raydrop_augm:
                per_pt_row_idxs = self.box_augm_db["lidar_rows"][obj_idx].astype(
                    np.int32
                )
                keep_this_point = self.layer_based_raydrop_augm(per_pt_row_idxs)

                if np.count_nonzero(keep_this_point) == 0:
                    # we don't want to drop all points!
                    pass
                else:
                    pcl_in_box = pcl_in_box[keep_this_point]
                    box_Ts_sensor = self.box_augm_db["box_T_sensor"][obj_idx]
                    pcl_sensor = np.concatenate(
                        [
                            np.einsum(
                                "ij,nj->ni",
                                np.linalg.inv(box_Ts_sensor),
                                homogenize_pcl(pcl_in_box[:, :3]),
                            )[..., :3],
                            pcl_in_box[..., [-1]],
                        ],
                        axis=-1,
                    )
                    keep_this_point = self.resolution_raydrop_augmentation(pcl_sensor)
                    if np.count_nonzero(keep_this_point) == 0:
                        # we don't want to drop all points!
                        pass
                    else:
                        pcl_sensor = pcl_sensor[keep_this_point]

            elif self.box_augm_cfg.max_points_dropout != 0.0:
                num_pts_in_box = pcl_in_box.shape[0]
                num_points_to_keep = max(
                    1,
                    int(
                        num_pts_in_box
                        * (
                            1.0
                            - np.random.rand() * self.box_augm_cfg.max_points_dropout
                        )
                    ),
                )
                chosen_point_idxs = np.random.choice(
                    np.arange(start=0, stop=num_pts_in_box, step=1, dtype=int),
                    num_points_to_keep,
                    replace=False,
                )

                pcl_in_box = np.copy(pcl_in_box)[chosen_point_idxs]

            else:
                pass
            flip_x = 1 if np.random.rand() < 0.5 else -1
            flip_y = 1 if np.random.rand() < 0.5 else -1
            scale_x = 1.0 - self.box_augm_cfg.max_scale_delta * (
                2 * np.random.rand() - 1.0
            )
            scale_y = 1.0 - self.box_augm_cfg.max_scale_delta * (
                2 * np.random.rand() - 1.0
            )
            scale_z = 1.0 - self.box_augm_cfg.max_scale_delta * (
                2 * np.random.rand() - 1.0
            )
            flip_trafo = np.eye(4)

            flip_trafo[0, 0] = flip_x * scale_x
            flip_trafo[1, 1] = flip_y * scale_y
            flip_trafo[2, 2] = scale_z

            pcl_sensor = np.einsum(
                "ij,nj->ni",
                sensor_Trand_box[i] @ flip_trafo,
                homogenize_pcl(pcl_in_box[:, :3]),
            )[:, :3]
            extra_flow = self.box_augm_cfg.min_artificial_obj_velo + np.random.rand(
                *pcl_sensor.shape
            ) * (
                self.box_augm_cfg.max_artificial_obj_velo
                - self.box_augm_cfg.min_artificial_obj_velo
            )
            extra_boxes.velo[i] = torch.from_numpy(
                np.mean(
                    np.linalg.norm(extra_flow[:, :3], axis=-1, keepdims=True), axis=0
                )
            )
            pcl_sensor_w_intensity = np.concatenate(
                [pcl_sensor, pcl_in_box[:, [-1]]], axis=-1
            )
            extra_pcl.append(pcl_sensor_w_intensity)
            if self.need_flow:
                extra_flows.append(extra_flow)

        extra_pcl = torch.from_numpy(
            np.concatenate(extra_pcl, axis=0).astype(np.float32)
        )
        if prediscovered_boxes is not None:
            extra_boxes = extra_boxes.cat(prediscovered_boxes, dim=0)
            assert torch.all(extra_boxes.probs == 1.0)
        else:
            prediscovered_boxes = Shape.createEmpty().to_tensor()

        augm_base_data = {
            "gt": {
                "boxes": extra_boxes,
            },
            "pcl_ta": torch.cat(
                [sample_data_ta["pcl_ta"]["pcl"].clone(), extra_pcl], axis=0
            ),
            "pcl_full_w_ground_ta": torch.cat(
                [sample_data_ta["pcl_full_w_ground_ta"].clone(), extra_pcl], axis=0
            ),
            "pcl_full_no_ground_ta": torch.cat(
                [sample_data_ta["pcl_full_no_ground_ta"].clone(), extra_pcl], axis=0
            ),
            "src_trgt_time_delta_s": torch.tensor(src_trgt_time_delta_s),
        }
        if "odom_ta_tb" in sample_data_ta["gt"]:
            # kitti object we don;t have this
            augm_base_data["gt"]["odom_ta_tb"] = sample_data_ta["gt"][
                "odom_ta_tb"
            ].clone()
        if self.cfg.data.flow_source not in augm_base_data:
            # if we use slim flow, we need extra dict
            augm_base_data[self.cfg.data.flow_source] = {}
        if self.need_flow:
            extra_flows = torch.from_numpy(
                np.concatenate(extra_flows, axis=0).astype(np.float32)
            )

            augm_base_data[self.cfg.data.flow_source]["flow_ta_tb"] = torch.cat(
                [
                    sample_data_ta[self.cfg.data.flow_source]["flow_ta_tb"].clone(),
                    extra_flows,
                ],
                axis=0,
            )

        augm_base_data = self.pillarize_bev(
            augm_base_data, src_key="ta", target_key="tb"
        )
        self.move_pcl_pillar_coors_to_subdict(augm_base_data, sk="ta")
        # if self.cfg.loss.supervised.centermaps.active:
        #     augm_base_data["gt"].update(
        #         self.get_motion_based_centermaps(augm_base_data)
        #     )
        if self.cfg.network.name not in ("pointrcnn", "pointpillars"):
            per_obj_prob_heatmap_scale = self.select_centermaps_target_confidence(
                extra_boxes, np.linalg.norm(extra_boxes.velo, axis=-1)
            )
            cluster_centermaps_ta = draw_heat_regression_maps(
                extra_boxes.numpy(),  # creates a clone internally
                self.centermaps_output_grid_size,
                self.bev_range_m_np,
                per_obj_prob_scale=per_obj_prob_heatmap_scale,
                box_pred_cfg=self.cfg.box_prediction,
            )
        else:
            cluster_centermaps_ta = {}

        augm_base_data[self.cfg.data.train_on_box_source] = {
            "boxes": extra_boxes,
            "prediscovered_boxes": prediscovered_boxes,
            **{
                f"centermaps_{k}": torch.from_numpy(v)
                for k, v in cluster_centermaps_ta.items()
            },
        }
        if "kitti_ignore_region_boxes_ta" in sample_data_ta["gt"]:
            ignore_boxes = sample_data_ta["gt"]["kitti_ignore_region_boxes_ta"]
            ignore_region_is_true_mask = self.create_true_where_ignore_region_mask(
                ignore_boxes,
            )
            augm_base_data["gt"]["ignore_region_is_true_mask"] = torch.from_numpy(
                ignore_region_is_true_mask
            )
        return augm_base_data

    def layer_based_raydrop_augm(self, per_pt_row_idxs: np.ndarray):
        keep_every_nth_row = np.random.choice([1, 2, 3])
        keep_row_start_idx = np.random.choice(np.arange(0, keep_every_nth_row))
        keep_this_point = (
            (per_pt_row_idxs - keep_row_start_idx) % keep_every_nth_row
        ) == 0

        return keep_this_point

    def resolution_raydrop_augmentation(self, pcl_sensor):
        range_m = np.linalg.norm(pcl_sensor[:, :3], axis=-1)

        azimuth_angle = np.arctan2(pcl_sensor[:, 1], pcl_sensor[:, 0])
        elevation_angle = np.arccos(pcl_sensor[:, 2] / np.maximum(0.00001, range_m))

        discretization_resolution = 2 * np.pi / np.random.choice([600, 900, 1200, 1500])

        azi_idx = (azimuth_angle / discretization_resolution).astype(np.int)
        ele_idx = (elevation_angle / discretization_resolution).astype(np.int)

        spherical_drop_ratio = np.random.choice([1, 2])

        keep_this_azimuth = (azi_idx % spherical_drop_ratio) == 0
        keep_this_elevation = (ele_idx % spherical_drop_ratio) == 0
        keep_this_point = keep_this_azimuth & keep_this_elevation
        return keep_this_point

    def create_augmented_sample_from_flow_cluster_detector_and_box_snippet_db(
        self,
        src_trgt_time_delta_s,
        sample_data_ta,
    ):
        augm_sample_ta = {
            "src_trgt_time_delta_s": torch.tensor(src_trgt_time_delta_s),
        }
        if (
            "supervised_on_clusters" in self.cfg.loss.supervised
            and self.cfg.loss.supervised.supervised_on_clusters.active
        ) or self.cfg.data.augmentation.boxes.active:
            prediscovered_boxes = sample_data_ta.get(
                self.cfg.data.train_on_box_source, {}
            ).get("boxes", None)
        else:
            assert "mined" not in sample_data_ta, sample_data_ta["mined"].keys()
            prediscovered_boxes = None

        if self.box_augm_db is not None:
            augm_sample_ta = self.create_augmented_sample_from_box_snippet_db(
                src_trgt_time_delta_s,
                sample_data_ta,
                prediscovered_boxes=prediscovered_boxes,
            )
        return augm_sample_ta


def get_weighted_random_sampler_dropping_samples_without_boxes(
    path_to_mined_boxes_db: Path,
    extra_loader_kwargs: Dict[str, bool],
    train_dataset: LidarDataset,
    ordered_keys_for_mining_db,
):
    mined_boxes_db = load_mined_boxes_db(path_to_mined_boxes_db)
    num_mined_boxes_per_sample = []
    for fname in ordered_keys_for_mining_db:
        if fname in mined_boxes_db:
            num_mined_boxes = mined_boxes_db[fname]["raw_box"]["valid"].shape[0]
        else:
            num_mined_boxes = 0
        num_mined_boxes_per_sample.append(num_mined_boxes)

    num_mined_boxes_per_sample = np.array(num_mined_boxes_per_sample)
    has_boxes = num_mined_boxes_per_sample > 0

    print(
        f"Keeping {np.count_nonzero(has_boxes)}/{len(ordered_keys_for_mining_db)} samples that have boxes in them!"
    )
    if not has_boxes.any():
        print("Warning: No samples with boxes found!")
        sample_weights = has_boxes + 1e-7
    else:
        sample_weights = has_boxes.astype(np.float64)
    sample_weights = sample_weights / np.sum(sample_weights)
    weighted_random_sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights)
    )

    assert extra_loader_kwargs.pop("shuffle"), "RandomSampler will shuffle data!"

    assert not train_dataset.shuffle, "shuffling within dataset might break everything!"

    return weighted_random_sampler


def get_augmentation_transform(
    max_symm_rot_deg: float,
    max_sensor_pos_offset_m: float,
    max_xy_scale_delta: float = None,
):
    delta_rot_deg = -max_symm_rot_deg + 2 * np.random.rand() * max_symm_rot_deg

    sensor_pos_offset_angle_rad = np.random.rand() * np.pi * 2.0
    sensor_pos_offset_m = np.random.rand() * max_sensor_pos_offset_m
    sensor_pos_offset_vec_m = sensor_pos_offset_m * np.array(
        [
            np.cos(sensor_pos_offset_angle_rad),
            np.sin(sensor_pos_offset_angle_rad),
            0.0,
        ]
    )

    if max_xy_scale_delta is None:
        scale = [1.0, 1.0, 1.0]
    else:
        scale_xy_abs = 1.0 + max_xy_scale_delta * (2 * np.random.rand() - 1)
        scale = [scale_xy_abs, scale_xy_abs, 1.0]

    augSensor_T_sensor = compose_matrix(
        angles=[0.0, 0.0, np.deg2rad(delta_rot_deg)],
        translate=sensor_pos_offset_vec_m,
        scale=scale,
    )

    return augSensor_T_sensor


def get_points_in_boxes_mask(
    objects: Shape,
    pcl_homog: Union[np.ndarray, torch.FloatTensor],
    return_pcl_in_box_cosy=False,
    use_double_precision=True,
) -> Union[np.ndarray, torch.BoolTensor]:
    assert pcl_homog.shape[-1] == 4, pcl_homog.shape
    assert len(pcl_homog.shape) == 2, pcl_homog.shape
    sensor_T_box = objects.get_poses()
    if torch.is_tensor(pcl_homog):
        assert use_double_precision, "not implemented for torch"
        assert torch.all(pcl_homog[:, -1] == 1.0)
        pcl_box = torch.einsum(
            "kij, nj->nki",
            torch.linalg.inv(sensor_T_box),
            pcl_homog.to(sensor_T_box.dtype),
        ).to(pcl_homog.dtype)
        point_is_in_box = torch.all(
            torch.abs(pcl_box[:, :, 0:3]) < 0.5 * objects.dims[None, ...], dim=-1
        )

    else:
        assert np.all(pcl_homog[:, -1] == 1.0)
        box_T_sensor = np.linalg.inv(sensor_T_box)
        if not use_double_precision:
            box_T_sensor = box_T_sensor.astype(np.float32)
        pcl_box = np.einsum("kij, nj->nki", box_T_sensor, pcl_homog)
        point_is_in_box = np.all(
            np.abs(pcl_box[:, :, 0:3]) < 0.5 * objects.dims[None, ...], axis=-1
        )
    if return_pcl_in_box_cosy:
        return point_is_in_box, pcl_box
    else:
        return point_is_in_box

from typing import Dict, List

import torch
from liso.kabsch.shape_utils import Shape
from liso.tracker.tracking_helpers import aggregate_odometry_to_world_poses


class NotATracker:
    def __init__(self) -> None:
        self.boxes_sensor_ti = []
        # self.propagated_box_poses_to_sensor_ti = []
        # self.propagated_box_poses_to_sensor_tiii = []
        self.detection_ids_ti = []
        self.sti_T_stii = []
        self.max_det_id_counter = 0
        self.per_box_extra_attributes_dict = []

        # tracking stuff:
        self.w_Ts_sti = None
        self.max_track_id_counter = 0
        self.has_tracked = False

    def update(
        self,
        boxes_tii_s: Shape,  # actual detections
        predicted_box_poses_stiii: None,
        predicted_box_poses_sti: None,
        odom_stii_stiii: torch.FloatTensor,
        per_box_extra_attributes_tii: List[Dict[str, str]] = None,
    ):
        assert len(boxes_tii_s.pos.shape) == 2, (
            "batching not supported",
            boxes_tii_s.pos.shape,
        )
        assert len(odom_stii_stiii.shape) == 2, (
            "batching not supported",
            odom_stii_stiii.shape,
        )
        assert torch.all(
            boxes_tii_s.valid
        ), "can't handle invalid boxes -> drop_invalid_boxes()"
        self.boxes_sensor_ti.append(boxes_tii_s.detach().cpu())
        self.sti_T_stii.append(odom_stii_stiii.detach().cpu())

        self.per_box_extra_attributes_dict.append(per_box_extra_attributes_tii)

        det_ids = (
            self.max_det_id_counter
            + 1
            + torch.arange(
                start=0,
                end=boxes_tii_s.valid.shape[0],
                device=boxes_tii_s.valid.device,
                dtype=torch.long,
            )
        )
        assert boxes_tii_s.valid.shape[0] == det_ids.shape[0]
        if det_ids.numel() > 0:
            self.max_det_id_counter = torch.max(det_ids)
        self.detection_ids_ti.append(det_ids)

    def run_tracker(
        self,
    ):
        self.w_Ts_sti = None
        self.boxes_world_ti = []
        self.max_track_id_counter = 0

        self.w_Ts_sti = aggregate_odometry_to_world_poses(self.sti_T_stii)

        num_timesteps = len(self.boxes_sensor_ti)
        for time_idx in range(num_timesteps):
            w_T_stii = self.w_Ts_sti[time_idx]
            boxes_sensor_tii = self.boxes_sensor_ti[time_idx]
            boxes_world_ti = boxes_sensor_tii.transform(w_T_stii)
            self.boxes_world_ti.append(boxes_world_ti.clone())

        self.track_ids = self.detection_ids_ti
        self.has_tracked = True

    def get_boxes_in_world_coordinates(self):
        return self.boxes_world_ti

    def get_boxes_in_sensor_coordinates_at_each_timestamp(self):
        assert self.has_tracked, "need to run tracking first"

        boxes_sensor = []

        for boxes_world, w_T_s in zip(self.boxes_world_ti, self.w_Ts_sti):
            boxes_sensor.append(boxes_world.clone().transform(torch.linalg.inv(w_T_s)))
        self.boxes_sensor_ti = boxes_sensor
        return self.boxes_sensor_ti

    def get_extra_attributes_at_each_timestamp(self):
        return self.per_box_extra_attributes_dict

    def get_min_max_track_id(self):
        all_track_ids, _ = self.get_all_unique_track_ids_and_lengths()
        if all_track_ids.size()[0] > 0:
            return all_track_ids.min(), all_track_ids.max()
        else:
            return torch.tensor(0).to(all_track_ids.device), torch.tensor(0).to(
                all_track_ids.device
            )

    def get_all_unique_track_ids_and_lengths(self):
        all_track_ids = torch.concat(self.track_ids, dim=0)
        return torch.unique(all_track_ids, return_counts=True)

    def get_ids_lengths_of_longest_tracks(self):
        unique_track_ids, track_lens = self.get_all_unique_track_ids_and_lengths()
        longest_track_lens_order = torch.argsort(track_lens, descending=True)
        return (
            unique_track_ids[longest_track_lens_order],
            track_lens[longest_track_lens_order],
        )

    def get_box_indices_start_time_for_track_id(self, track_id):
        pad_val = -1
        track_ids_padded = torch.nn.utils.rnn.pad_sequence(
            self.track_ids, batch_first=True, padding_value=pad_val
        )
        track_id_mask = track_ids_padded == track_id
        timestamps, box_idxs = torch.where(track_id_mask)
        start_timestamp_idx = timestamps[0]
        return box_idxs, start_timestamp_idx

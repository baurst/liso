from typing import Dict, List

import numpy as np
import torch
from liso.kabsch.box_groundtruth_matching import (
    slow_greedy_match_boxes_by_desending_confidence_by_dist,
)
from liso.kabsch.shape_utils import Shape
from liso.tracker.tracking_helpers import aggregate_odometry_to_world_poses
from liso.utils.torch_transformation import torch_decompose_matrix


class FlowBasedBoxTracker:
    def __init__(
        self,
        use_propagated_boxes=False,
        box_matching_threshold_m=5.0,
        association_strategy="ours",
    ) -> None:
        self.use_propagated_boxes = use_propagated_boxes
        self.box_matching_threshold = box_matching_threshold_m

        self.boxes_sensor_ti = []
        self.propagated_box_poses_to_sensor_ti = []
        self.propagated_box_poses_to_sensor_tiii = []
        # self.detection_ids_ti = []
        self.sti_T_stii = []
        self.max_det_id_counter = 0
        self.per_box_extra_attributes_dict = []

        # tracking stuff:
        self.w_Ts_sti = None
        self.max_track_id_counter = 0
        self.has_tracked = False
        assert association_strategy in ("ours",)
        self.association_strategy = association_strategy

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
        if self.use_propagated_boxes:
            self.propagated_box_poses_to_sensor_ti.append(
                predicted_box_poses_sti.detach().cpu()
            )
            self.propagated_box_poses_to_sensor_tiii.append(
                predicted_box_poses_stiii.detach().cpu()
            )
        self.per_box_extra_attributes_dict.append(per_box_extra_attributes_tii)

        # det_ids = torch.arange(
        #     start=self.max_det_id_counter,
        #     end=self.max_det_id_counter + boxes_tii_s.valid.shape[0],
        #     device=boxes_tii_s.valid.device,
        #     dtype=torch.long,
        # )
        # if det_ids.numel() > 0:
        #     self.max_det_id_counter = torch.max(det_ids)
        # self.detection_ids_ti.append(det_ids)

    def run_tracker(self):
        self.w_Ts_sti = None
        self.boxes_world_ti = []
        self.max_track_id_counter = 0

        self.w_Ts_sti = aggregate_odometry_to_world_poses(self.sti_T_stii)

        boxes_world_ti_fwd = []
        boxes_world_ti_bwd = []
        propagated_box_poses_into_future_world = []
        propagated_box_poses_into_past_world = []
        num_timesteps = len(self.boxes_sensor_ti)
        for time_idx in range(num_timesteps):
            w_T_stii = self.w_Ts_sti[time_idx]
            boxes_sensor_tii = self.boxes_sensor_ti[time_idx]
            boxes_world_ti = boxes_sensor_tii.transform(w_T_stii)
            self.boxes_world_ti.append(boxes_world_ti.clone())
            boxes_world_ti_fwd.append(boxes_world_ti.clone())
            boxes_world_ti_bwd.append(boxes_world_ti.clone())

            if self.use_propagated_boxes:
                w_T_sti = self.w_Ts_sti[max(time_idx - 1, 0)]
                propagated_box_poses_into_past_world.append(
                    w_T_sti @ self.propagated_box_poses_to_sensor_ti[time_idx]
                )
                w_T_stiii = self.w_Ts_sti[min(time_idx + 1, num_timesteps - 1)]
                propagated_box_poses_into_future_world.append(
                    w_T_stiii @ self.propagated_box_poses_to_sensor_tiii[time_idx]
                )

        (
            boxes_world_ti_fwd,
            fwd_track_ids,
            self.max_track_id_counter,
            per_box_extra_attributes_dict_fwd,
        ) = self.track_one_way(
            boxes_world_ti_fwd,
            self.max_track_id_counter,
            self.box_matching_threshold,
            per_box_extra_attributes_dict=self.per_box_extra_attributes_dict,
            propagated_poses_into_world_past_ti=propagated_box_poses_into_past_world,
            association_strategy=self.association_strategy,
        )

        (
            boxes_world_ti_bwd,
            bwd_track_ids,
            self.max_track_id_counter,
            _,
        ) = self.track_one_way(
            boxes_world_ti_bwd[::-1],
            self.max_track_id_counter,
            self.box_matching_threshold,
            per_box_extra_attributes_dict=None,
            propagated_poses_into_world_past_ti=propagated_box_poses_into_future_world[
                ::-1
            ],
            association_strategy=self.association_strategy,
        )
        bwd_track_ids = bwd_track_ids[::-1]
        boxes_world_ti_bwd = boxes_world_ti_bwd[::-1]

        # per track_id track age
        all_fwd_track_ids = torch.concat(fwd_track_ids, dim=0)
        uniq_track_ids_fwd, track_lens_fwd = torch.unique(
            all_fwd_track_ids, return_counts=True
        )
        fwd_id_age_lookup = dict(zip(uniq_track_ids_fwd.numpy(), track_lens_fwd))
        all_bwd_track_ids = torch.concat(bwd_track_ids, dim=0)
        uniq_track_ids_bwd, track_lens_bwd = torch.unique(
            all_bwd_track_ids, return_counts=True
        )
        bwd_id_age_lookup = dict(zip(uniq_track_ids_bwd.numpy(), track_lens_bwd))

        combined_track_ids = []
        combined_track_ages = []
        combined_extra_attrs = []

        num_timestamps = len(self.boxes_world_ti)
        for time_idx in range(num_timestamps):
            num_detected_boxes = self.boxes_world_ti[time_idx].valid.shape[0]
            if num_detected_boxes > 0:
                track_ids_at_time_t_fwd = fwd_track_ids[time_idx].numpy()[
                    :num_detected_boxes
                ]
                track_ages_fwd = torch.stack(
                    [
                        fwd_id_age_lookup[track_id]
                        for track_id in track_ids_at_time_t_fwd
                    ]
                )
                track_ids_at_time_t_bwd = bwd_track_ids[time_idx].numpy()[
                    :num_detected_boxes
                ]
                track_ages_bwd = torch.stack(
                    [
                        bwd_id_age_lookup[track_id]
                        for track_id in track_ids_at_time_t_bwd
                    ]
                )
                track_ages_at_t = torch.maximum(
                    track_ages_fwd, track_ages_bwd
                )  # first, collect age only for actual detections
                extra_attrs_at_t = per_box_extra_attributes_dict_fwd[time_idx][
                    :num_detected_boxes
                ]
            else:
                track_ids_at_time_t_fwd = np.zeros((0,), dtype=np.int64)
                track_ages_at_t = torch.zeros((0,), dtype=torch.long)
                extra_attrs_at_t = []
            combined_track_ids.append(torch.from_numpy(track_ids_at_time_t_fwd))
            combined_track_ages.append(track_ages_at_t)
            combined_extra_attrs.append(extra_attrs_at_t)

        def rindex(mylist, myvalue):
            """
            Return the index of the last occurrence of myvalue in mylist.
            """
            return len(mylist) - mylist[::-1].index(myvalue) - 1

        # fill any holes in the tracks!
        for track_id in uniq_track_ids_fwd:
            occurs_at_timestamp = [
                (track_id == combined_track_ids[time_idx]).any()
                for time_idx in range(num_timestamps)
            ]
            first_occurence_time_idx = occurs_at_timestamp.index(True)
            last_occurence_time_idx = rindex(occurs_at_timestamp, True)
            if last_occurence_time_idx - first_occurence_time_idx >= 2:
                occurance_subsequence = np.array(
                    occurs_at_timestamp[
                        first_occurence_time_idx:last_occurence_time_idx
                    ]
                )
                there_are_holes_in_sequence = np.any(~occurance_subsequence)
                if there_are_holes_in_sequence:
                    hole_locations = (
                        np.where(~occurance_subsequence)[0] + first_occurence_time_idx
                    )
                    for hole_location_time_idx in hole_locations:
                        box_idx = torch.where(
                            fwd_track_ids[hole_location_time_idx] == track_id
                        )[0]
                        extrapolated_box = boxes_world_ti_fwd[hole_location_time_idx][
                            box_idx
                        ]
                        extrapolated_extra_attrs = per_box_extra_attributes_dict_fwd[
                            hole_location_time_idx
                        ][box_idx]
                        self.boxes_world_ti[
                            # fill in the missing boxes
                            hole_location_time_idx
                        ] = self.boxes_world_ti[hole_location_time_idx].cat(
                            extrapolated_box, dim=0
                        )
                        combined_track_ids[hole_location_time_idx] = torch.cat(
                            [
                                combined_track_ids[hole_location_time_idx],
                                track_id[None],
                            ]
                        )
                        combined_extra_attrs[hole_location_time_idx].append(
                            extrapolated_extra_attrs
                        )
        for time_idx in range(num_timestamps):
            assert (
                len(combined_track_ids[time_idx])
                == self.boxes_world_ti[time_idx].shape[0]
            ), (
                len(combined_track_ids[time_idx]),
                self.boxes_world_ti[time_idx].shape[0],
            )
            assert (
                len(combined_extra_attrs[time_idx])
                == self.boxes_world_ti[time_idx].shape[0]
            ), (
                len(combined_extra_attrs[time_idx]),
                self.boxes_world_ti[time_idx].shape[0],
            )
        self.track_ids = combined_track_ids
        self.has_tracked = True

    @staticmethod
    def track_one_way(
        boxes_world_tii_fwd,
        max_track_id_counter: int,
        box_matching_threshold,
        association_strategy: str,
        per_box_extra_attributes_dict=None,
        propagated_poses_into_world_past_ti=None,
    ):
        max_propagation_time = 1
        initial_track_conf = 1.0
        min_alive_track_conf = 0.0
        if per_box_extra_attributes_dict is None:
            per_box_extra_attributes_dict = [
                [
                    None,
                ]
                * boxes_world_tii_fwd[time_idx].shape[0]
                for time_idx in range(len(boxes_world_tii_fwd))
            ]

        fwd_track_ages = []
        fwd_track_confidence = []
        fwd_track_ids = []
        if boxes_world_tii_fwd[0].valid.shape[0] > 0:
            fwd_track_ids.append(
                1
                + torch.arange(
                    start=max_track_id_counter,
                    end=max_track_id_counter + boxes_world_tii_fwd[0].valid.shape[0],
                    device=boxes_world_tii_fwd[0].valid.device,
                    dtype=torch.long,
                )
            )
            max_track_id_counter = fwd_track_ids[-1].max()
        else:
            fwd_track_ids.append(torch.zeros(0, dtype=torch.long))
        fwd_track_ages.append(torch.zeros_like(fwd_track_ids[0]))
        fwd_track_confidence.append(
            initial_track_conf * torch.ones_like(fwd_track_ids[0], dtype=torch.float)
        )
        for time_idx in range(1, len(boxes_world_tii_fwd)):
            prev_boxes_ti_world_ti = boxes_world_tii_fwd[time_idx - 1]
            if time_idx >= 2:
                # we have previous tracks
                # propagate motion
                prev_track_ids = fwd_track_ids[-1]
                prevprev_boxes_ti_world_ti = boxes_world_tii_fwd[time_idx - 2]
                prevprev_track_ids = fwd_track_ids[-2]
                prev_match_prevprev_mask = (
                    prev_track_ids[..., None] == prevprev_track_ids[None, ...]
                )
                prevprev_pos, _ = torch_decompose_matrix(
                    prevprev_boxes_ti_world_ti.get_poses()
                )
                prev_boxes_world_propagated = prev_boxes_ti_world_ti.clone()
                cur_has_prev_match = prev_match_prevprev_mask.any(dim=-1)
                if cur_has_prev_match.any():
                    prev_match_idx = torch.argwhere(prev_match_prevprev_mask)[:, 1]
                    prev_boxes_world_propagated.pos[cur_has_prev_match] += (
                        prev_boxes_world_propagated.pos[cur_has_prev_match]
                        - prevprev_pos[prev_match_idx]
                    )
                else:
                    pass
            else:  # -> all track ids are different!
                prev_boxes_world_propagated = prev_boxes_ti_world_ti.clone()

            prev_track_confs = fwd_track_confidence[-1]
            prev_track_is_alive = prev_track_confs >= min_alive_track_conf

            cur_boxes_world_tii = boxes_world_tii_fwd[time_idx]
            if torch.count_nonzero(prev_track_is_alive) > 1:
                non_batched_pred_confidence = torch.squeeze(
                    # need confidence of prev boxes
                    prev_track_confs[prev_track_is_alive],
                    dim=-1,
                )
            else:
                non_batched_pred_confidence = prev_track_confs[prev_track_is_alive]
            if association_strategy == "ours":
                # we use match against propagated detection into past
                predicted_prev_box_pos, _ = torch_decompose_matrix(
                    propagated_poses_into_world_past_ti[time_idx]
                )
                box_loc_alive_prev, _ = torch_decompose_matrix(
                    prev_boxes_ti_world_ti[prev_track_is_alive].get_poses()
                )
                (
                    idxs_into_curr,
                    idxs_into_alive_prevs,
                    _,
                    matched_prevs_mask,
                    matched_currs_mask,
                ) = slow_greedy_match_boxes_by_desending_confidence_by_dist(
                    predicted_prev_box_pos,
                    box_loc_alive_prev,
                    non_batched_pred_confidence=non_batched_pred_confidence,
                    matching_threshold=box_matching_threshold,
                    match_in_nd=2,
                )
            else:
                raise NotImplementedError(association_strategy)
            matched_prevs_mask = torch.from_numpy(matched_prevs_mask)
            propagated_box_is_unmatched_and_alive = prev_track_is_alive.clone()
            propagated_box_is_unmatched_and_alive[prev_track_is_alive] = (
                propagated_box_is_unmatched_and_alive[prev_track_is_alive]
                & ~matched_prevs_mask
            )
            # propagated_box_is_unmatched_and_alive = (
            #     prev_track_is_alive & ~matched_prevs_mask
            # )
            prev_track_ids = fwd_track_ids[-1]
            prev_track_ages = fwd_track_ages[-1]
            unmatched_alive_propagated_track_ids = prev_track_ids[
                propagated_box_is_unmatched_and_alive
            ]
            unmatched_alive_propagated_track_ages = prev_track_ages[
                propagated_box_is_unmatched_and_alive
            ]
            unmatched_alive_propagated_confidence = prev_track_confs[
                propagated_box_is_unmatched_and_alive
            ]
            unmatched_alive_propagated_boxes = prev_boxes_world_propagated[
                propagated_box_is_unmatched_and_alive
            ]
            curr_track_ids = -1 * torch.ones_like(
                cur_boxes_world_tii.valid, dtype=torch.long
            )
            curr_track_ids[idxs_into_curr] = prev_track_ids[prev_track_is_alive][
                idxs_into_alive_prevs
            ]
            newborn_track_ids = (
                max_track_id_counter
                + 1
                + torch.arange(
                    start=0,
                    end=np.count_nonzero(~matched_currs_mask),
                    dtype=torch.long,
                    device=curr_track_ids.device,
                )
            )
            curr_track_ids[~matched_currs_mask] = newborn_track_ids
            new_track_confidence = initial_track_conf * torch.ones_like(
                cur_boxes_world_tii.valid, dtype=torch.float32
            )

            if association_strategy == "ours":
                unmatched_alive_damped_propagated_confidence = (
                    0.0001  # 1.0 - 1.0 / 1 = 0.0 > 0.0 -> False --> add eps
                    + unmatched_alive_propagated_confidence
                    - initial_track_conf / max_propagation_time
                )
            else:
                raise NotImplementedError(association_strategy)

            assert torch.all(curr_track_ids >= 0), curr_track_ids
            new_track_ids = torch.cat(
                [curr_track_ids, unmatched_alive_propagated_track_ids], dim=0
            )

            if newborn_track_ids.numel() > 0:
                # only update if there are actually objects
                new_max_track_id = new_track_ids.max()
            else:
                new_max_track_id = max_track_id_counter
            new_track_ages = torch.zeros_like(
                cur_boxes_world_tii.valid, dtype=torch.long
            )
            new_track_ages[matched_currs_mask] = (
                1 + prev_track_ages[prev_track_is_alive][idxs_into_alive_prevs]
            )
            new_track_ages = torch.cat(
                [new_track_ages, unmatched_alive_propagated_track_ages], dim=0
            )
            new_track_confidence = torch.cat(
                [
                    new_track_confidence,
                    unmatched_alive_damped_propagated_confidence,
                ]
            )

            boxes_world_tii_fwd[time_idx] = boxes_world_tii_fwd[time_idx].cat(
                unmatched_alive_propagated_boxes, dim=0
            )
            if torch.count_nonzero(propagated_box_is_unmatched_and_alive) > 0:
                prev_extra_attrs = np.array(per_box_extra_attributes_dict[time_idx - 1])
                unmatched_alive_propagated_extra_attrs = prev_extra_attrs[
                    propagated_box_is_unmatched_and_alive.numpy()
                ]
                per_box_extra_attributes_dict[time_idx] = np.concatenate(
                    [
                        np.array(per_box_extra_attributes_dict[time_idx]),
                        unmatched_alive_propagated_extra_attrs,
                    ]
                ).tolist()

            fwd_track_ids.append(new_track_ids)
            max_track_id_counter = new_max_track_id
            fwd_track_ages.append(new_track_ages)
            fwd_track_confidence.append(new_track_confidence)

        return (
            boxes_world_tii_fwd,
            fwd_track_ids,
            max_track_id_counter,
            per_box_extra_attributes_dict,
        )

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

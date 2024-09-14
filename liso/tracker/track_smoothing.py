import pickle
import shutil
from pathlib import Path
from typing import Any, List

import matplotlib
import numpy as np
import torch
from liso.utils.timing_utils import timeit
from matplotlib import pyplot as plt

matplotlib.use("Agg")


def torch_yaw_signed_diff(
    gt_yaw: torch.FloatTensor, pred_yaw: torch.FloatTensor, period: float = 2 * np.pi
) -> float:
    diff = (gt_yaw - pred_yaw + period / 2) % period - period / 2
    diff = torch.where(diff > np.pi, diff - (2 * np.pi), diff)  # wraparound angle
    return diff


def cauchy(logits: torch.FloatTensor) -> torch.FloatTensor:
    retval = 0.5 + 1 / np.pi * torch.atan(logits)
    return retval


def soft_sigmoid_clamp(
    x: torch.FloatTensor, a_min: float, a_max: float
) -> torch.FloatTensor:
    retval = a_min + (a_max - a_min) * cauchy(x / 100)
    return retval


MIN_TRACK_LEN_FOR_SMOOTHING = 4


class BatchedSmoothTrack(torch.nn.Module):
    def __init__(
        self,
        batched_observed_track_pos: torch.FloatTensor,
        batched_observed_track_yaw_angle_rad: torch.FloatTensor,
        time_between_frames_s: float,
    ):
        super(BatchedSmoothTrack, self).__init__()
        assert (
            len(batched_observed_track_pos.shape) == 3
        ), batched_observed_track_pos.shape
        assert (
            batched_observed_track_pos.shape[1] >= MIN_TRACK_LEN_FOR_SMOOTHING
        ), f"for jerk, need at least {MIN_TRACK_LEN_FOR_SMOOTHING} positions"
        self.initial_pos = batched_observed_track_pos[:, [0]].float().detach()
        self._smooth_pos = torch.nn.Parameter(
            batched_observed_track_pos[:, 1:, :].float(), requires_grad=True
        )
        self.smooth_rot = torch.nn.Parameter(
            batched_observed_track_yaw_angle_rad.float(), requires_grad=True
        )
        self.time_between_frames_s = time_between_frames_s

        assert self.pos.shape == batched_observed_track_pos.shape, (
            self.pos.shape,
            batched_observed_track_pos.shape,
        )

    @property
    def pos(self):
        return torch.cat([self.initial_pos, self._smooth_pos], dim=1)

    def forward(self):
        return (
            self.pos,
            self.smooth_rot,
        )


def get_pos_jerk_magnitude(pos: torch.FloatTensor) -> torch.FloatTensor:
    batch_size, num_timesteps, _ = pos.shape
    translation_jerk = torch.linalg.norm(torch.diff(pos, n=3, dim=1), dim=-1)
    _, num_jerk_timesteps = translation_jerk.shape
    num_missing_timesteps = num_timesteps - num_jerk_timesteps
    padding_values = torch.zeros((batch_size, num_missing_timesteps), device=pos.device)
    translation_jerk = torch.cat([translation_jerk, padding_values], dim=1)
    return translation_jerk


def batched_displacement_from_pos(
    pos: torch.FloatTensor, num_skip=1
) -> torch.FloatTensor:
    assert num_skip >= 1, num_skip
    disp = torch.linalg.norm(pos[:, num_skip:, :] - pos[:, :-num_skip, :], dim=-1)
    if num_skip == 1:
        disp = torch.cat([disp, disp[:, [-num_skip]]], dim=1)
    else:
        disp = torch.cat(
            [disp[:, : (num_skip // 2)], disp, disp[:, (-num_skip // 2) :]],
            dim=1,
        )
    assert pos.shape[:-1] == disp.shape, (pos.shape, disp.shape)
    return disp


@timeit
def smooth_track_jerk(
    batched_observed_pos_m: torch.FloatTensor,
    batched_valid_mask: torch.BoolTensor,
    batched_observed_yaw_angle_rad: torch.FloatTensor,
    time_between_frames_s: float,
    pos_regul_loss_weight=3.0,
    max_iters=2000,
    learning_rate=0.1,
    verbose=False,
    return_losses=False,
):
    if batched_observed_pos_m.shape[1] <= 4:
        # to optimize for min jerk, need at least 4 positions
        return (
            batched_observed_pos_m,
            batched_observed_yaw_angle_rad,
            batched_displacement_from_pos(batched_observed_pos_m),
        )
    # torch.save(observed_pos_m, "/tmp/pos")
    # torch.save(observed_yaw_angle_rad, "/tmp/yaw_angles")
    track = BatchedSmoothTrack(
        batched_observed_pos_m.clone(),
        batched_observed_yaw_angle_rad.clone(),
        time_between_frames_s,
    )
    optimizer = torch.optim.Adam(track.parameters(), lr=learning_rate)

    track.train()

    trust_track_endpoints_weight = torch.ones(
        batched_observed_pos_m.shape[:2], device=batched_observed_pos_m.device
    )
    num_timesteps_per_batch = torch.sum(batched_valid_mask, dim=1)
    # batch_size = batched_valid_mask.shape[0]
    # batch_idxs = torch.arange(batch_size, device=batched_valid_mask.device)
    # trust_track_endpoints_weight[:, :2] = 0.1
    # trust_track_endpoints_weight[batch_idxs, num_timesteps_per_batch - 1] = 0.1
    # trust_track_endpoints_weight[batch_idxs, num_timesteps_per_batch - 2] = 0.1
    if return_losses:
        losses = []
    if verbose:
        track_len = batched_observed_pos_m.shape[1]
        print(f"Track length: {track_len}")
    for _ in range(max_iters):
        optimizer.zero_grad()
        (
            smooth_pos_m,  # pos_jerk_magnitudes,
            _,  # rot_jerk_magnitudes,
        ) = track()

        per_track_pos_jerk_magnitudes = get_pos_jerk_magnitude(smooth_pos_m)

        per_track_pos_jerk_loss = per_batch_mean_loss(
            per_track_pos_jerk_magnitudes, batched_valid_mask, num_timesteps_per_batch
        )
        # avg_rot_jerk_magnitude = rot_pos_jerk_loss_tradeoff_factor * torch.mean(
        #     rot_jerk_magnitudes
        # )
        # rot_accel_penalty_loss = rot_accel_penalty * torch.mean(
        #     torch.exp(rot_accel_magnitude)
        # )
        # pos_accel_penalty_loss = pos_accel_penalty * torch.mean(
        #     torch.exp(pos_accel_magnitude)
        # )
        shift_dist_m = torch.nn.functional.mse_loss(
            smooth_pos_m, batched_observed_pos_m[:, :, :3], reduction="none"
        ).sum(dim=-1)
        per_track_pos_regularization_loss = pos_regul_loss_weight * per_batch_mean_loss(
            trust_track_endpoints_weight * shift_dist_m,
            batched_valid_mask,
            num_timesteps_per_batch,
        )
        per_track_loss = (
            per_track_pos_jerk_loss
            + per_track_pos_regularization_loss
            # + avg_rot_jerk_magnitude
            # + rot_accel_penalty
            # + pos_accel_penalty
        )
        # avg_shift_distance_m = torch.mean(shift_dist_m)
        # shift_dist_rad = torch_yaw_diff(
        #     track.smooth_rot,
        #     observed_yaw_angle_rad,
        # )
        # avg_shift_dist_rad = torch.mean(shift_dist_rad)
        # pos_regularization_loss = (
        #     observation_regularization_weight * avg_shift_distance_m
        # )
        # rot_regularization_loss = 0 * (
        #     observation_regularization_weight
        #     * avg_shift_dist_rad
        #     * rot_pos_regul_loss_tradeoff_factor
        # )
        # regularization_loss = pos_regularization_loss + rot_regularization_loss
        # rot_deviation_from_track_loss = (
        #     rot_deviations_from_track_loss_weight
        #     * torch.mean(rot_deviations_from_track)
        # )
        loss = per_track_loss.mean()
        # print(loss.detach().numpy())
        loss.backward()
        if return_losses:
            losses.append(
                {
                    "per_batch_jerk_loss": per_track_pos_jerk_loss.detach()
                    .cpu()
                    .numpy(),
                    "per_batch_loss": per_track_loss.detach().cpu().numpy(),
                    "pos_regul": per_track_pos_regularization_loss.detach()
                    .cpu()
                    .numpy(),
                }
            )
        optimizer.step()

    track_positions_m = track.pos.detach()
    rot_along_track = batched_observed_yaw_angle_rad.detach()

    min_disp_for_rot_alignment_m = 1.0
    track_rot_was_aligned = (
        torch.zeros_like(rot_along_track[..., 0], dtype=bool)
        | ~batched_valid_mask  # ignore all invalid positions
    )
    max_num_iters = min(10, track_positions_m.shape[1] // 2)
    num_iters = 0
    # we want to align the rotation of the track with the direction of the track
    # but only where the track is long enough to have a meaningful rotation
    # so we iteratively extend the time delta between track positions
    while not track_rot_was_aligned.all() and num_iters < max_num_iters:
        num_iters += 1
        track_displacement_m = batched_displacement_from_pos(
            track_positions_m, num_skip=num_iters
        )
        is_far_enough_for_rot_alignment_mask = is_far_enough_for_rot_alignment(
            track_displacement_m, min_disp_for_rot_alignment_m
        )

        track_can_be_aligned = (
            ~track_rot_was_aligned & is_far_enough_for_rot_alignment_mask
        )

        rot_along_track[track_can_be_aligned] = get_orientations_along_track(
            track_positions_m,
            pad_borders=True,
            num_skip=num_iters,
        )[..., None][track_can_be_aligned]

        track_rot_was_aligned = track_rot_was_aligned | track_can_be_aligned

    # assume constant rotation at the start and end of track
    rot_along_track[:, 0, :] = rot_along_track[:, 1, :]

    last_valid_idx = torch.sum(batched_valid_mask, dim=1) - 1
    last_valid_idx = torch.stack(
        [
            torch.arange(batched_valid_mask.shape[0]),
            last_valid_idx,
            torch.zeros_like(last_valid_idx),
        ],
        dim=-1,
    )
    second_to_last_valid_idx = last_valid_idx.clone()
    assert rot_along_track.shape[-1] == 1, rot_along_track.shape
    second_to_last_valid_idx[:, 1] -= 1
    rot_along_track[
        last_valid_idx[:, 0], last_valid_idx[:, 1], last_valid_idx[:, 2]
    ] = rot_along_track[
        second_to_last_valid_idx[:, 0],
        second_to_last_valid_idx[:, 1],
        second_to_last_valid_idx[:, 2],
    ]

    optimized_rot = rot_along_track
    optimized_velo = batched_displacement_from_pos(track_positions_m)[..., None]
    if return_losses:
        return (
            track.pos.detach(),
            optimized_rot.detach(),
            optimized_velo,
            losses,
        )

    return (
        track.pos.detach(),
        optimized_rot.detach(),
        optimized_velo,
    )


def is_far_enough_for_rot_alignment(
    track_displacement_m: torch.FloatTensor,
    min_disp_for_rot_alignment_m: float,
):
    return track_displacement_m > min_disp_for_rot_alignment_m


def car_dynamics(
    *,
    kinematics: torch.FloatTensor,
    accel: torch.FloatTensor,
    dd_heading: torch.FloatTensor,
    dt: float,
    x_idx: int,
    y_idx: int,
    heading_idx: int,
    velo_idx: int,
    hdotix: int,
    vehicle_length: torch.FloatTensor,
    max_yaw_rate: torch.FloatTensor,
    max_velocity: torch.FloatTensor,
):
    """
    kinematics [batch_size x 5]
    Based on kinematic Bicycle model.
    Note car can't go backwards.
    """
    assert kinematics.shape[-1] == 5, kinematics.shape
    newhdot = soft_sigmoid_clamp(
        kinematics[:, hdotix] + dd_heading * dt, -max_yaw_rate, max_yaw_rate
    )
    newh = (
        kinematics[:, heading_idx]
        + dt * kinematics[:, velo_idx].abs() / vehicle_length * newhdot
    )
    news = soft_sigmoid_clamp(kinematics[:, velo_idx] + accel * dt, 0.0, max_velocity)
    newy = kinematics[:, y_idx] + news * newh.sin() * dt
    newx = kinematics[:, x_idx] + news * newh.cos() * dt
    newstate = torch.empty_like(kinematics)
    newstate[:, x_idx] = newx
    newstate[:, y_idx] = newy
    newstate[:, heading_idx] = newh
    newstate[:, velo_idx] = news
    newstate[:, hdotix] = newhdot
    return newstate


class BatchedBikeModel(torch.nn.Module):
    def __init__(
        self,
        *,
        batched_observed_track_pos: torch.FloatTensor,
        batched_vehicle_length: torch.FloatTensor,
        time_between_frames_s: float,
        max_yaw_rate: float,
        max_velocity: float,
        optimize_initial_pos=True,
    ):
        batch_size, num_time_steps, _ = batched_observed_track_pos.shape
        assert (
            num_time_steps >= MIN_TRACK_LEN_FOR_SMOOTHING
        ), f"need at least {MIN_TRACK_LEN_FOR_SMOOTHING} positions for smoothing"
        self.batch_size = int(batch_size)
        self.num_time_steps = int(num_time_steps)
        super(BatchedBikeModel, self).__init__()
        assert (
            len(batched_observed_track_pos.shape) == 3
        ), batched_observed_track_pos.shape
        # state: [x, y, heading, velocity, heading_rate]

        self.propagated_states = []
        velo_mps_init = torch.linalg.norm(
            batched_observed_track_pos[:, 2:, :2]
            - batched_observed_track_pos[:, :-2, :2],
            dim=-1,
        ) / (2 * time_between_frames_s)

        self.accel_over_time = torch.nn.Parameter(
            torch.zeros_like(batched_observed_track_pos[..., 0]), requires_grad=True
        )
        self.steering_input_over_time = torch.nn.Parameter(
            torch.zeros_like(batched_observed_track_pos[..., 0]),
            requires_grad=True,
        )
        init_smooth_yaw_angles = get_orientations_along_track(
            pos=batched_observed_track_pos[:, :, :2]
        )
        init_smooth_yaw_rate = (
            torch_yaw_signed_diff(
                init_smooth_yaw_angles[:, 1:], init_smooth_yaw_angles[:, :1]
            )
            / time_between_frames_s
        )

        self.initial_pos = torch.nn.Parameter(
            batched_observed_track_pos[:, 0, 0:2], requires_grad=optimize_initial_pos
        )
        self.initial_yaw = torch.nn.Parameter(
            init_smooth_yaw_angles[:, [0]], requires_grad=True
        )
        self.initial_velo_mps = torch.nn.Parameter(
            velo_mps_init[:, [0]], requires_grad=True
        )
        self.initial_yaw_rate_radps = torch.nn.Parameter(
            init_smooth_yaw_rate[:, [0]], requires_grad=True
        )
        self.x_idx = 0
        self.y_idx = 1
        self.heading_idx = 2
        self.velo_idx = 3
        self.hdot_idx = 4

        self.time_between_frames_s = float(time_between_frames_s)
        assert batched_vehicle_length.shape == (batched_observed_track_pos.shape[0],)
        self.vehicle_length = batched_vehicle_length
        self.max_yaw_rate = max_yaw_rate
        self.max_velocity = max_velocity

    def forward(self):
        initial_state = torch.cat(
            [
                self.initial_pos,
                self.initial_yaw,
                self.initial_velo_mps,
                self.initial_yaw_rate_radps,
            ],
            dim=-1,
        )
        propagated_states = forward_compiled(
            initial_state=initial_state,
            num_time_steps=int(self.num_time_steps),
            accel_over_time=self.accel_over_time,
            steering_input_over_time=self.steering_input_over_time,
            time_between_frames_s=self.time_between_frames_s,
            x_idx=self.x_idx,
            y_idx=self.y_idx,
            heading_idx=self.heading_idx,
            velo_idx=self.velo_idx,
            hdot_idx=self.hdot_idx,
            vehicle_length=self.vehicle_length,
            max_yaw_rate=self.max_yaw_rate,
            max_velocity=self.max_velocity,
        )
        self.propagated_states = propagated_states
        return self.propagated_states

    @property
    def pos(self):
        return self.propagated_states[..., [self.x_idx, self.y_idx]]

    @property
    def rot(self):
        return self.propagated_states[..., [self.heading_idx]]

    @property
    def yaw_rate(self):
        return self.propagated_states[..., [self.hdot_idx]]

    @property
    def velo(self):
        return self.propagated_states[..., [self.velo_idx]]

    def get_pos_jerk_magnitude(self):
        translation_jerk = torch.linalg.norm(torch.diff(self.pos, n=3, dim=0), dim=-1)
        return translation_jerk


def get_orientations_along_track(
    pos: torch.FloatTensor, pad_borders=True, num_skip=2
) -> torch.FloatTensor:
    target_pts = pos[:, num_skip:, :2]  # .detach()
    src_pts = pos[:, :-num_skip, :2]  # .detach()
    dir_vecs = target_pts - src_pts
    dir_vec_len = torch.max(
        torch.linalg.norm(dir_vecs, dim=-1, keepdim=True),
        torch.tensor(0.00001, device=dir_vecs.device),
    )
    dir_vecs = dir_vecs.detach() / dir_vec_len.detach()
    track_angle = torch.atan2(dir_vecs[:, :, 1], dir_vecs[:, :, 0])
    if pad_borders:
        if num_skip == 1:
            track_angle = torch.cat([track_angle, track_angle[:, [-num_skip]]], dim=1)
        else:
            track_angle = torch.cat(
                [
                    track_angle[:, : (num_skip // 2)],
                    track_angle,
                    track_angle[:, (-num_skip // 2) :],
                ],
                dim=1,
            )

        assert pos.shape[:-1] == track_angle.shape, (pos.shape, track_angle.shape)
    return track_angle


@torch.jit.script
def forward_compiled(
    *,
    initial_state: torch.FloatTensor,
    num_time_steps: int,
    accel_over_time: torch.FloatTensor,
    steering_input_over_time: torch.FloatTensor,
    time_between_frames_s: float,
    x_idx: int,
    y_idx: int,
    heading_idx: int,
    velo_idx: int,
    hdot_idx: int,
    vehicle_length: torch.FloatTensor,
    max_yaw_rate: float,
    max_velocity: float,
) -> torch.FloatTensor:
    propagated_states = [
        initial_state,
    ]
    max_yaw_rate = torch.tensor(max_yaw_rate, device=initial_state.device)
    max_velocity = torch.tensor(max_velocity, device=initial_state.device)
    for time_idx in range(num_time_steps - 1):
        propagated_states.append(
            car_dynamics(
                kinematics=propagated_states[-1],
                accel=accel_over_time[:, time_idx],
                dd_heading=steering_input_over_time[:, time_idx],
                dt=time_between_frames_s,
                x_idx=x_idx,
                y_idx=y_idx,
                heading_idx=heading_idx,
                velo_idx=velo_idx,
                hdotix=hdot_idx,
                vehicle_length=vehicle_length,
                max_yaw_rate=max_yaw_rate,
                max_velocity=max_velocity,
            )
        )
    propagated_states = torch.stack(propagated_states, dim=0).permute((1, 0, 2))
    return propagated_states


# @torch.jit.script
# def car_dynamics_forward_over_whole_time_sequence(
#     state_over_time: List[torch.FloatTensor],
#     accel_over_time: torch.FloatTensor,
#     steering_input_over_time: torch.FloatTensor,
#     *,
#     time_between_frames_s: float,
#     num_time_steps: int,
#     vehicle_len: float,
#     max_yaw_rate: float,
#     max_velocity: float,
#     x_idx: int,
#     y_idx: int,
#     heading_idx: int,
#     velo_idx: int,
#     hdot_idx: int,
# ):
#     for time_idx in range(num_time_steps - 1):
#         state_over_time[time_idx + 1] = car_dynamics(
#             state_over_time[time_idx],
#             accel_over_time[time_idx],
#             steering_input_over_time[time_idx],
#             dt=time_between_frames_s,
#             x_idx=x_idx,
#             y_idx=y_idx,
#             heading_idx=heading_idx,
#             velo_idx=velo_idx,
#             hdotix=hdot_idx,
#             vehicle_length=vehicle_len,
#             max_yaw_rate=max_yaw_rate,
#             max_velocity=max_velocity,
#         )
#
#     return state_over_time


def per_batch_mean_loss(
    per_element_loss, valid_mask, num_elements_in_batch
) -> torch.FloatTensor:
    return (
        torch.sum(per_element_loss * valid_mask.float(), dim=-1) / num_elements_in_batch
    )


@timeit
def smooth_track_bike_model(
    *,
    batched_observed_pos_m: torch.FloatTensor,
    batched_valid_mask: torch.BoolTensor,
    batched_observed_yaw_angle_rad: torch.FloatTensor,
    batched_vehicle_length_m: torch.FloatTensor,
    time_between_frames_s: float,
    max_iters=30,
    learning_rate=0.1,
    accel_penalty_weight=0.1,
    velo_penalty_weight=0.1,
    pos_regul_loss_weight=1.0,
    max_velocity_mps=50.0,
    max_yaw_rate_radps=np.pi / 2,
    verbose=False,
    return_losses=False,
):
    # torch.autograd.set_detect_anomaly(True)
    batched_observed_pos_m = batched_observed_pos_m.clone()
    batched_observed_yaw_angle_rad = batched_observed_yaw_angle_rad.clone()
    if batched_observed_pos_m.shape[1] < MIN_TRACK_LEN_FOR_SMOOTHING:
        print(
            f"to optimize bike model, need at least {MIN_TRACK_LEN_FOR_SMOOTHING} positions"
        )
        return (
            batched_observed_pos_m,
            batched_observed_yaw_angle_rad,
            batched_displacement_from_pos(batched_observed_pos_m),
        )
    track = BatchedBikeModel(
        batched_observed_track_pos=batched_observed_pos_m,
        time_between_frames_s=time_between_frames_s,
        batched_vehicle_length=batched_vehicle_length_m,
        max_velocity=max_velocity_mps,
        max_yaw_rate=max_yaw_rate_radps,
    )
    # optimizer = torch.optim.Adam(track.parameters(), lr=learning_rate)
    optimizer = torch.optim.LBFGS(
        track.parameters(),
        lr=learning_rate,
        max_iter=20,
        # tolerance_grad=1e-5,
        # tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    track.train()

    if return_losses:
        losses = []
    if verbose:
        track_len = batched_observed_pos_m.shape[1]
        print(f"Track length: {track_len}")

    num_timesteps_per_batch = torch.sum(batched_valid_mask, dim=1)

    def forward_pass_closure(
        # observed_pos_m, accel_penalty_weight, pos_regul_loss_weight, track,
    ):
        optimizer.zero_grad()
        track.forward()

        # pos_jerk_magnitudes = track.get_pos_jerk_magnitude()
        # avg_pos_jerk_magnitude = pos_jerk_loss_weight * torch.mean(pos_jerk_magnitudes)
        # avg_pos_jerk_magnitude = torch.tensor(0.0, device=observed_pos_m.device)
        if accel_penalty_weight > 0.0:
            per_track_linear_accel_penalty = accel_penalty_weight * per_batch_mean_loss(
                track.accel_over_time**2,
                batched_valid_mask,
                num_timesteps_per_batch,
            )
            per_track_yaw_accel_penalty = accel_penalty_weight * per_batch_mean_loss(
                track.steering_input_over_time**2,
                batched_valid_mask,
                num_timesteps_per_batch,
            )
        else:
            per_track_yaw_accel_penalty = torch.zeros(
                (batched_valid_mask.shape[0],), device=batched_observed_pos_m.device
            )
            per_track_linear_accel_penalty = torch.zeros(
                (batched_valid_mask.shape[0],), device=batched_observed_pos_m.device
            )

        if velo_penalty_weight > 0.0:
            too_fast_yaw_rates = (
                torch.abs(torch.squeeze(track.yaw_rate, dim=-1)) > max_yaw_rate_radps
            )
            yaw_rate_penalty = torch.where(
                too_fast_yaw_rates,
                torch.squeeze(track.yaw_rate, dim=-1) ** 2,
                torch.zeros_like(too_fast_yaw_rates, dtype=torch.float32),
            )
            per_track_yaw_rate_penalty = per_batch_mean_loss(
                yaw_rate_penalty, batched_valid_mask, num_timesteps_per_batch
            )
        else:
            per_track_yaw_rate_penalty = torch.zeros(
                (batched_valid_mask.shape[0],), device=batched_observed_pos_m.device
            )

        # shift_dist_m = (track.pos - batched_observed_pos_m[:, :, :2]) ** 2
        shift_dist_m = torch.nn.functional.mse_loss(
            track.pos, batched_observed_pos_m[:, :, :2], reduction="none"
        ).sum(dim=-1)
        per_track_pos_regularization_loss = pos_regul_loss_weight * per_batch_mean_loss(
            shift_dist_m, batched_valid_mask, num_timesteps_per_batch
        )
        per_track_loss = (
            per_track_linear_accel_penalty
            + per_track_yaw_accel_penalty
            + per_track_yaw_rate_penalty
            + per_track_pos_regularization_loss
        )
        loss = per_track_loss.mean()
        loss.backward()
        if return_losses:
            losses.append(
                {
                    "per_batch_loss": per_track_loss.detach().cpu().numpy(),
                    "linear_accel_penalty": per_track_linear_accel_penalty.detach()
                    .cpu()
                    .numpy(),
                    "yaw_accel_penalty": per_track_yaw_accel_penalty.detach()
                    .cpu()
                    .numpy(),
                    "yaw_rate_penalty": per_track_yaw_rate_penalty.detach()
                    .cpu()
                    .numpy(),
                    "pos_regul": per_track_pos_regularization_loss.detach()
                    .cpu()
                    .numpy(),
                }
            )
        return loss

    for _ in range(max_iters):
        # optimizer.zero_grad()

        # loss = forward_pass_closure(
        #     observed_pos_m, accel_penalty_weight, pos_regul_loss_weight, track
        # )

        # loss.backward()
        optimizer.step(forward_pass_closure)

        # print(track.accel_over_time[10:20])

    optimized_pos = torch.cat([track.pos, batched_observed_pos_m[:, :, 2:]], dim=-1)
    optimized_yaw = track.rot
    optimized_velo = batched_displacement_from_pos(optimized_pos)[..., None]

    if return_losses:
        return (
            optimized_pos,
            optimized_yaw,
            optimized_velo,
            losses,
        )

    return (
        optimized_pos,
        optimized_yaw,
        optimized_velo,
    )


def main():
    smoothing_method = "bike_model"
    # smoothing_method = "jerk"

    log_dir = Path("logs_track_smoothing")
    if log_dir.exists():
        shutil.rmtree(log_dir.as_posix())
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    device = torch.device("cpu")

    loaded_boxes = []
    for box_pkl in sorted(Path("tracks_to_be_smoothed").glob("*.pkl")):
        loaded_boxes.append(pickle.load(open(box_pkl, "rb")))

    (
        batched_observed_pos_m,
        batched_observed_yaw_angle_rad,
        batched_vehicle_length_m,
        batched_valid_mask,
    ) = batch_box_data_for_batched_smoothing(
        loaded_boxes,
        device,
    )
    if smoothing_method == "bike_method":
        (
            batched_smooth_pos,
            batched_smooth_yaw,
            _,
            batched_loss,
        ) = smooth_track_bike_model(
            batched_observed_pos_m=batched_observed_pos_m,
            batched_observed_yaw_angle_rad=batched_observed_yaw_angle_rad,
            batched_vehicle_length_m=batched_vehicle_length_m,
            batched_valid_mask=batched_valid_mask,
            time_between_frames_s=0.1,
            verbose=True,
            return_losses=True,
        )
    elif smoothing_method == "jerk":
        batched_smooth_pos, batched_smooth_yaw, _, batched_loss = smooth_track_jerk(
            batched_observed_pos_m=batched_observed_pos_m,
            batched_observed_yaw_angle_rad=batched_observed_yaw_angle_rad,
            batched_valid_mask=batched_valid_mask,
            time_between_frames_s=0.1,
            verbose=True,
            return_losses=True,
        )
    else:
        raise NotImplementedError(smoothing_method)

    for idx, box_pkl in enumerate(sorted(Path("tracks_to_be_smoothed").glob("*.pkl"))):
        fname = box_pkl.stem
        plt.figure(0)
        box_stuff = pickle.load(open(box_pkl, "rb"))
        plot_track(
            box_stuff.pos,
            box_stuff.rot,
            smooth_pos_m=batched_smooth_pos[idx][batched_valid_mask[idx]].cpu(),
            smooth_yaw_rad=batched_smooth_yaw[idx][batched_valid_mask[idx]].cpu(),
            save_to=log_dir / f"{fname}_smooth_track.png",
        )
        plt.close()
        plt.cla()
        plt.clf()

        plt.figure(1)

        pyplot_list_of_dicts(
            get_nth_element_from_each_dict_entry(batched_loss, idx),
            log_dir / f"{fname}_loss.png",
        )
        plt.close()
        plt.cla()
        plt.clf()

        print("Done!")


def split_batched_padded_tensor_into_list(
    batched_tensor: torch.Tensor, valid_mask: torch.BoolTensor
) -> List[torch.Tensor]:
    return [batched_tensor[i][valid_mask[i]] for i in range(batched_tensor.shape[0])]


def batch_box_data_for_batched_smoothing(
    loaded_boxes: List[Any],
    device,
):
    batched_observed_pos_m = []
    batched_observed_yaw_angle_rad = []
    batched_vehicle_length_m = []
    batched_valid_mask = []
    for box_stuff in loaded_boxes:
        batched_observed_pos_m.append(box_stuff.pos)
        batched_valid_mask.append(
            torch.ones_like(box_stuff.pos[:, 0], dtype=torch.bool)
        )
        batched_observed_yaw_angle_rad.append(box_stuff.rot)
        batched_vehicle_length_m.append(torch.median(box_stuff.dims[:, 0]))

    batched_observed_pos_m = (
        torch.nn.utils.rnn.pad_sequence(batched_observed_pos_m, batch_first=True)
        .to(device)
        .to(torch.float32)
    )
    batched_valid_mask = torch.nn.utils.rnn.pad_sequence(
        batched_valid_mask, batch_first=True, padding_value=False
    ).to(device)
    batched_observed_yaw_angle_rad = (
        torch.nn.utils.rnn.pad_sequence(
            batched_observed_yaw_angle_rad, batch_first=True
        )
        .to(device)
        .to(torch.float32)
    )
    batched_vehicle_length_m = (
        torch.stack(batched_vehicle_length_m).to(device).to(torch.float32)
    )

    return (
        batched_observed_pos_m,
        batched_observed_yaw_angle_rad,
        batched_vehicle_length_m,
        batched_valid_mask,
    )


def get_nth_element_from_each_dict_entry(list_of_dicts, idx):
    return [{k: v[idx] for k, v in dict_el.items()} for dict_el in list_of_dicts]


def plot_track(
    observed_pos_m,
    observed_yaw_rad,
    smooth_pos_m,
    smooth_yaw_rad,
    save_to,
):
    plt.axes().set_aspect("equal", "datalim")

    plt.plot(
        observed_pos_m[..., 0].detach().cpu().numpy(),
        observed_pos_m[..., 1].detach().cpu().numpy(),
        "b",
        label="observed",
        linewidth=1,
    )
    plt.quiver(
        observed_pos_m[..., 0].detach().cpu().numpy(),
        observed_pos_m[..., 1].detach().cpu().numpy(),
        np.cos(observed_yaw_rad[..., 0].detach().cpu().numpy()),
        np.sin(observed_yaw_rad[..., 0].detach().cpu().numpy()),
        color="b",
        width=0.003,
    )
    plt.scatter(
        smooth_pos_m[..., 0].detach().cpu().numpy(),
        smooth_pos_m[..., 1].detach().cpu().numpy(),
        color="r",
        label="smoothed",
        s=1,
    )

    shifted_smooth_pos_m = smooth_pos_m + torch.tensor(
        [0.0, 10.0, 0.0], device=smooth_pos_m.device
    )
    plt.scatter(
        shifted_smooth_pos_m[..., 0].detach().cpu().numpy(),
        shifted_smooth_pos_m[..., 1].detach().cpu().numpy(),
        color="r",
        label="smoothed",
        s=1,
    )
    plt.quiver(
        shifted_smooth_pos_m[..., 0].detach().cpu().numpy(),
        shifted_smooth_pos_m[..., 1].detach().cpu().numpy(),
        np.cos(smooth_yaw_rad[..., 0].detach().cpu().numpy()),
        np.sin(smooth_yaw_rad[..., 0].detach().cpu().numpy()),
        color="r",
        width=0.003,
    )
    plt.legend()
    plt.savefig(save_to, dpi=300)


def list_of_dict_to_dict_of_lists(list_of_dicts):
    keys = list_of_dicts[0].keys()
    dict_of_lists = {k: [] for k in keys}
    for d in list_of_dicts:
        for k in keys:
            dict_of_lists[k].append(d[k])
    return dict_of_lists


def pyplot_list_of_dicts(list_of_dicts, save_to):
    dict_of_lists = list_of_dict_to_dict_of_lists(list_of_dicts)
    for k in dict_of_lists:
        plt.plot(dict_of_lists[k], label=k)
    plt.legend()
    plt.savefig(save_to, dpi=300)


if __name__ == "__main__":
    main()

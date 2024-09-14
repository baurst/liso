#!/usr/bin/env python3

from typing import Tuple

import torch
from torch import nn


class MovingAverageThreshold(nn.Module):
    def __init__(
        self,
        num_train_samples: int,
        num_moving: int,
        num_still=None,
        resolution: int = 100000,
        start_value: float = 0.5,
        value_range: Tuple[float, float] = (0.0, 1.0),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.value_range = (value_range[0], value_range[1] - value_range[0])
        self.resolution = resolution
        self.num_moving = num_moving
        self.num_still = num_still
        self.register_buffer(
            "start_value", torch.tensor(start_value, dtype=torch.float)
        )
        self.total = num_moving
        if num_still is not None:
            self.total += num_still
        assert num_train_samples > 0, num_train_samples
        avg_points_per_sample = self.total / num_train_samples
        update_weight = 1.0 / min(
            2.0 * self.total, 5_000.0 * avg_points_per_sample
        )  # update buffer roughly every 5k iterations, so 5k * points per sample for denominator
        self.register_buffer(
            "update_weight", torch.tensor(update_weight, dtype=torch.double)
        )

        if num_still is not None:
            self.register_buffer(
                "moving_counter", torch.tensor(self.num_moving, dtype=torch.long)
            )
            self.register_buffer(
                "still_counter", torch.tensor(self.num_still, dtype=torch.long)
            )

        self.register_buffer("bias_counter", torch.zeros((), dtype=torch.double))
        self.register_buffer(
            "moving_average_importance",
            torch.zeros((self.resolution,), dtype=torch.float),
        )

    def value(self):
        if self.bias_counter > 0.0:
            return self._compute_optimal_score_threshold()
        else:
            return self.start_value

    def _compute_bin_idxs(self, dynamicness_scores):
        idxs = (
            (dynamicness_scores - self.value_range[0])
            * self.resolution
            / self.value_range[1]
        ).to(torch.int)
        assert (idxs <= self.resolution).all()
        assert (idxs >= 0).all()
        idxs = torch.clamp(idxs, max=self.resolution - 1)
        assert (idxs < self.resolution).all()
        return idxs

    def _compute_improvements(
        self,
        epes_stat_flow,
        epes_dyn_flow,
        moving_mask,
    ):
        if self.num_still is None:
            assert moving_mask is None
            improvements = epes_stat_flow - epes_dyn_flow
        else:
            assert moving_mask is not None
            assert moving_mask.ndim == 1
            improvement_weight = 1.0 / torch.where(
                moving_mask, self.moving_counter, self.still_counter
            ).to(torch.float)
            improvements = (epes_stat_flow - epes_dyn_flow) * improvement_weight
        return improvements

    def _compute_optimal_score_threshold(self):
        improv_over_thresh = torch.cat(
            [
                torch.zeros(
                    (1,),
                    dtype=self.moving_average_importance.dtype,
                    device=self.moving_average_importance.device,
                ),
                torch.cumsum(self.moving_average_importance, 0),
            ],
            dim=0,
        )
        best_improv = torch.min(improv_over_thresh)
        avg_optimal_idx = torch.mean(
            torch.where(best_improv == improv_over_thresh)[0].to(torch.float)
        )
        optimal_score_threshold = (
            self.value_range[0]
            + avg_optimal_idx * self.value_range[1] / self.resolution
        )
        return optimal_score_threshold

    def _update_values(self, cur_value, cur_weight):
        cur_update_weight = (1.0 - self.update_weight) ** cur_weight
        self.moving_average_importance *= cur_update_weight.to(torch.float)
        self.moving_average_importance += (
            1.0 - cur_update_weight.to(torch.float)
        ) * cur_value
        self.bias_counter *= cur_update_weight
        self.bias_counter += 1.0 - cur_update_weight

    def update(
        self,
        epes_stat_flow,
        epes_dyn_flow,
        moving_mask,
        dynamicness_scores,
        training,
    ):
        assert isinstance(training, bool)
        if training:
            # TODO: clarify why this was on cpu()?
            # cpu() leads to error bc different devices
            epes_stat_flow = epes_stat_flow.detach()  # .cpu()
            epes_dyn_flow = epes_dyn_flow.detach()  # .cpu()
            # assert moving_mask is None
            dynamicness_scores = dynamicness_scores.detach()  # .cpu()

            assert len(epes_stat_flow.shape) == 1
            assert len(epes_dyn_flow.shape) == 1
            assert len(dynamicness_scores.shape) == 1
            improvements = self._compute_improvements(
                epes_stat_flow,
                epes_dyn_flow,
                moving_mask,
            )
            bin_idxs = self._compute_bin_idxs(dynamicness_scores)
            cur_result = torch.zeros(
                (self.resolution,), dtype=improvements.dtype, device=improvements.device
            ).scatter_add_(0, bin_idxs.to(torch.long), improvements)
            self._update_values(cur_result, epes_stat_flow.numel())
            if self.num_still is not None:
                self.moving_counter += torch.count_nonzero(moving_mask)
                self.still_counter += torch.count_nonzero(~moving_mask)
            result = self.value()
            return result
        return self.value()

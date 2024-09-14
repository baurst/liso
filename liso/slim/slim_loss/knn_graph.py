#!/usr/bin/env python3

from copy import deepcopy

import numpy as np
import pynanoflann
import torch


def knn_graph(
    x: torch.Tensor,
    *,
    index: torch.Tensor = None,
    k: int,
    batch=None,
    loop: bool = False,
    flow: str = "source_to_target",
    cosine=False,
    num_workers: int = 1,
    return_kd_tree: bool = False,
):
    # #region raise deprecation warnings for unsupported kwargs
    assert flow == "source_to_target"
    assert not cosine
    assert num_workers == 1
    # #endregion raise deprecation warnings for unsupported kwargs

    assert torch.isfinite(x).all()

    if batch is not None:
        assert not return_kd_tree
        assert index is None
        assert batch.numel() == x.shape[0]
        assert (batch[1:] >= batch[:-1]).all()
        bs = batch.max() + 1
        points_handled = 0
        all_edge_indexes = []
        for b in range(bs):
            mask = batch == b
            all_edge_indexes.append(
                knn_graph(
                    x[mask],
                    k=k,
                    batch=None,
                    loop=loop,
                    flow=flow,
                    cosine=cosine,
                    num_workers=1,
                )
            )
            if points_handled > 0:
                all_edge_indexes[-1] += points_handled
            points_handled += mask.sum()

        return torch.cat(all_edge_indexes, dim=1)

    queries = x.detach().cpu().numpy()
    if index is None:
        index = queries
        index_given = False
    else:
        index = index.detach().cpu().numpy()
        index_given = True
        assert loop

    nn = pynanoflann.KDTree(n_neighbors=k + int(not loop), metric="L2", leaf_size=20)
    nn.fit(index)

    # Get k-nearest neighbors
    _, indices = nn.kneighbors(queries)
    if not loop:
        indices = indices[:, 1:]

    assert indices.shape[-1] == k

    if index_given:
        assert indices.dtype == np.uint64
        return torch.from_numpy(indices.astype(np.int64)).to(x.device)

    edge_index = (
        np.stack(
            [
                indices,
                np.arange(indices.shape[0], dtype=indices.dtype)[:, None].repeat(
                    k, axis=-1
                ),
            ],
            axis=-1,
        )
        .reshape(-1, 2)
        .T
    )
    assert edge_index.dtype == np.uint64
    edge_index = torch.tensor(edge_index.astype(np.int64)).to(x.device)

    if return_kd_tree:
        return edge_index, deepcopy(nn)
    return edge_index

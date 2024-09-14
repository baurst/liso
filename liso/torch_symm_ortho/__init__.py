#!/usr/bin/env python3

import torch
from liso.utils.debug import print_stats


class SymmetricOrthogonalization(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def grad_compute_symmetric_orthogonalization(grad_R, U, Vh, D):
        n = D.size(-1)
        batch_shape = D.shape[:-1]
        assert grad_R.shape[-2:] == (n, n)
        assert U.shape == grad_R.shape
        assert Vh.shape == grad_R.shape
        assert D.shape == grad_R.shape[:-1]

        assert (D >= 0.0).all(), D
        dl_plus_dk = (
            D[..., :, None]
            + D[..., None, :]
            + torch.eye(n, device=D.device, dtype=D.dtype)
        )
        assert (dl_plus_dk > 0).all(), (
            print_stats("dl_plus_dk", dl_plus_dk),
            print_stats("D", D),
        )
        V = Vh.transpose(-1, -2)
        omega_ij = (
            U[..., :, None, :, None] * V[..., None, :, None, :]
            - U[..., :, None, None, :] * V[..., None, :, :, None]
        ) / dl_plus_dk[..., None, None, :, :]
        dR_per_daij = U[..., None, None, :, :] @ omega_ij @ Vh[..., None, None, :, :]
        grad_A = dR_per_daij.reshape(*batch_shape, n, n, 1, n * n) @ grad_R.reshape(
            *batch_shape, 1, 1, n * n, 1
        )
        grad_A = grad_A.squeeze(-1).squeeze(-1)
        return grad_A

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        assert input.dtype in {
            torch.float,
            torch.double,
        }, "complex cannot be handle right now, discrete datatypes not possible"
        dim = input.size(-1)
        assert (
            input.size(-2) == dim
        ), "quadratic matrices required in last two dimensions: " + str(input.shape)

        U, D, Vh = torch.linalg.svd(input)
        R = U @ Vh
        assert R.shape == input.shape, (R.shape, input.shape)
        assert D.shape == input.shape[:-1], (D.shape, R.shape)

        ctx.save_for_backward(U, Vh, D)
        return R

    @staticmethod
    def backward(ctx, grad_R):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (U, Vh, D) = ctx.saved_tensors
        grad_input = (
            SymmetricOrthogonalization.grad_compute_symmetric_orthogonalization(
                grad_R, U, Vh, D
            )
        )
        return grad_input


symmetric_orthogonalization = SymmetricOrthogonalization.apply

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_radius(mat: np.ndarray) -> float:
    eigen_values = np.linalg.eigvals(mat)
    spectral_radius = max(abs(eigen_values))
    return spectral_radius


def _tanh_saturation_distance(h: torch.Tensor) -> torch.Tensor:
    return 1 - torch.abs(h)


def _sigmoid_saturation_distance(h: torch.Tensor) -> torch.Tensor:
    return torch.min(h, 1 - h)


class VanillaRNN(nn.Module):

    def __init__(
        self,
        nin: int,
        nout: int,
        nhid: int,
        init: str = "smart_tanh",
        classif_type: str = "lastSoftmax",
        rng=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.nhid = nhid
        self.init = init
        self.classif_type = classif_type

        if init == "sigmoid":
            W_uh = rng.normal(loc=0.0, scale=0.01, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.01, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.01, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "sigmoid"

        elif init == "test":
            W_uh = rng.normal(loc=0.0, scale=0.8, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.8, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.8, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "identity"

        elif init == "basic_tanh":
            W_uh = rng.normal(loc=0.0, scale=0.1, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.1, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.1, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "tanh"

        elif init == "smart_tanh":
            W_uh = rng.normal(loc=0.0, scale=0.01, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.01, size=(nhid, nhid)).astype(np.float32)

            for dx in range(nhid):
                spng = rng.permutation(nhid)
                W_hh[dx, spng[15:]] = 0.0

            sr = spectral_radius(W_hh)
            if sr > 0:
                W_hh = (0.95 * W_hh / sr).astype(np.float32)

            W_hy = rng.normal(loc=0.0, scale=0.01, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "tanh"

        else:
            raise ValueError(f"Unknown init={init}. Choose from sigmoid, test, basic_tanh, smart_tanh")

        self.W_uh = nn.Parameter(torch.tensor(W_uh, dtype=dtype, device=device))
        self.W_hh = nn.Parameter(torch.tensor(W_hh, dtype=dtype, device=device))
        self.W_hy = nn.Parameter(torch.tensor(W_hy, dtype=dtype, device=device))
        self.b_hh = nn.Parameter(torch.tensor(b_hh, dtype=dtype, device=device))
        self.b_hy = nn.Parameter(torch.tensor(b_hy, dtype=dtype, device=device))

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "sigmoid":
            return torch.sigmoid(x)
        if self.act_name == "tanh":
            return torch.tanh(x)
        if self.act_name == "identity":
            return x
        raise RuntimeError("bad act_name")

    def act_deriv_from_h(self, h: torch.Tensor) -> torch.Tensor:
        if self.act_name == "sigmoid":
            return h * (1.0 - h)
        if self.act_name == "tanh":
            return 1.0 - h * h
        if self.act_name == "identity":
            return torch.ones_like(h)
        raise RuntimeError("bad act_name")

    def forward(self, u: torch.Tensor):

        T, B, nin = u.shape
        hidden_states = []
        prev_hidden_state = torch.zeros((B, self.nhid), dtype=u.dtype, device=u.device)

        for t in range(T):
            batch_of_tokens_at_timestamp_t = u[t]
            current_hidden_state = batch_of_tokens_at_timestamp_t @ self.W_uh + prev_hidden_state @ self.W_hh + self.b_hh
            current_hidden_state = self.act(current_hidden_state)

            hidden_states.append(current_hidden_state)
            prev_hidden_state = current_hidden_state

        hstack = torch.stack(hidden_states, dim=0)
        layer_outputs = hstack @ self.W_hy + self.b_hy

        if self.classif_type == "lastSoftmax" or self.classif_type == "lastLinear":
            logits = layer_outputs[-1]

        elif self.classif_type == "softmax":
            logits = layer_outputs.reshape((T * B, self.nout))

        return logits, hstack

    supports_omega: bool = True

    def saturation_distance_from_h(self, h: torch.Tensor) -> torch.Tensor:

        if self.act_name == "sigmoid":
            return _sigmoid_saturation_distance(h)

        return _tanh_saturation_distance(h)

    def recurrent_weight_for_rho(self) -> torch.Tensor:
        return self.W_hh

    def numpy_state(self) -> dict:
        return {
            "W_hh": self.W_hh.detach().cpu().numpy(),
            "W_uh": self.W_uh.detach().cpu().numpy(),
            "W_hy": self.W_hy.detach().cpu().numpy(),
            "b_hh": self.b_hh.detach().cpu().numpy(),
            "b_hy": self.b_hy.detach().cpu().numpy(),
            "act_name": np.array(self.act_name),
            "classif_type": np.array(self.classif_type),
            "model_type": np.array("rnn"),
        }


class GRUModel(nn.Module):

    supports_omega: bool = False

    def __init__(
        self,
        nin: int,
        nout: int,
        nhid: int,
        init: str = "smart_tanh",
        classif_type: str = "lastSoftmax",
        rng: np.random.RandomState | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()

        self.nin = int(nin)
        self.nout = int(nout)
        self.nhid = int(nhid)
        self.init = init
        self.classif_type = classif_type

        if rng is None:
            rng = np.random.RandomState(1234)

        W_uz = rng.normal(0, 0.01, size=(nin, nhid))
        W_hz = rng.normal(0, 0.01 / 2, size=(nhid, nhid))
        b_z = np.zeros((nhid,), dtype=np.float64) - 2

        W_ur = rng.normal(0, 0.01, size=(nin, nhid))
        W_hr = rng.normal(0, 0.01 / 2, size=(nhid, nhid))
        b_r = np.zeros((nhid,), dtype=np.float64)

        W_hh = rng.normal(0, 0.01, size=(nhid, nhid))
        W_uh = rng.normal(0, 0.01, size=(nin, nhid))
        b_h = np.zeros((nhid,), dtype=np.float64)

        W_hy = rng.normal(0, 0.01, size=(nhid, nout))
        b_y = np.zeros((nout,), dtype=np.float64)

        self.W_uz = nn.Parameter(torch.tensor(W_uz, dtype=dtype, device=device))
        self.W_hz = nn.Parameter(torch.tensor(W_hz, dtype=dtype, device=device))
        self.b_z = nn.Parameter(torch.tensor(b_z, dtype=dtype, device=device))

        self.W_ur = nn.Parameter(torch.tensor(W_ur, dtype=dtype, device=device))
        self.W_hr = nn.Parameter(torch.tensor(W_hr, dtype=dtype, device=device))
        self.b_r = nn.Parameter(torch.tensor(b_r, dtype=dtype, device=device))

        self.W_uh = nn.Parameter(torch.tensor(W_uh, dtype=dtype, device=device))
        self.W_hh = nn.Parameter(torch.tensor(W_hh, dtype=dtype, device=device))
        self.b_h = nn.Parameter(torch.tensor(b_h, dtype=dtype, device=device))

        self.W_hy = nn.Parameter(torch.tensor(W_hy, dtype=dtype, device=device))
        self.b_y = nn.Parameter(torch.tensor(b_y, dtype=dtype, device=device))

    def saturation_distance_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return _tanh_saturation_distance(h)

    def act_deriv_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return 1.0 - h * h

    def recurrent_weight_for_rho(self) -> torch.Tensor:
        return self.W_hh

    def numpy_state(self) -> dict:
        return {
            "W_uz": self.W_uz.detach().cpu().numpy(),
            "W_hz": self.W_hz.detach().cpu().numpy(),
            "b_z": self.b_z.detach().cpu().numpy(),
            "W_ur": self.W_ur.detach().cpu().numpy(),
            "W_hr": self.W_hr.detach().cpu().numpy(),
            "b_r": self.b_r.detach().cpu().numpy(),
            "W_uh": self.W_uh.detach().cpu().numpy(),
            "W_hh": self.W_hh.detach().cpu().numpy(),
            "b_h": self.b_h.detach().cpu().numpy(),
            "W_hy": self.W_hy.detach().cpu().numpy(),
            "b_y": self.b_y.detach().cpu().numpy(),
            "init": np.array(self.init),
            "classif_type": np.array(self.classif_type),
            "model_type": np.array("gru"),
        }

    def forward(self, u: torch.Tensor, return_extras: bool = False):

        T, B, _ = u.shape

        z_t = torch.zeros((B, self.nhid), dtype=u.dtype, device=u.device)
        r_t = torch.zeros((B, self.nhid), dtype=u.dtype, device=u.device)
        h_tilde_t = torch.zeros((B, self.nhid), dtype=u.dtype, device=u.device)
        h_t = torch.zeros((B, self.nhid), dtype=u.dtype, device=u.device)

        hidden_states = []
        preactivations_z = []
        preactivations_r = []
        preactivations_h_tilde = []

        for t in range(T):

            u_t = u[t]

            z_t = torch.sigmoid(u_t @ self.W_uz + h_t @ self.W_hz + self.b_z)
            r_t = torch.sigmoid(u_t @ self.W_ur + h_t @ self.W_hr + self.b_r)
            h_tilde_t = torch.tanh(u_t @ self.W_uh + (r_t * h_t) @ self.W_hh + self.b_h)

            h_t = (1 - z_t) * h_t + z_t * h_tilde_t

            hidden_states.append(h_t)
            preactivations_z.append(z_t)
            preactivations_r.append(r_t)
            preactivations_h_tilde.append(h_tilde_t)

        hstack = torch.stack(hidden_states, dim=0)
        layer_outputs = hstack @ self.W_hy + self.b_y

        if self.classif_type == "lastSoftmax" or self.classif_type == "lastLinear":
            logits = layer_outputs[-1]

        elif self.classif_type == "softmax":
            logits = layer_outputs.reshape((T * B, self.nout))

        else:
            raise ValueError(f"Unknown classif_type={self.classif_type}")

        if return_extras:
            extras = {
                "z": torch.stack(preactivations_z, dim=0),
                "r": torch.stack(preactivations_r, dim=0),
                "h_tilde": torch.stack(preactivations_h_tilde, dim=0),
            }
            return logits, hstack, extras

        return logits, hstack


def make_model(
    model_type: str,
    nin: int,
    nout: int,
    nhid: int,
    init: str,
    classif_type: str,
    rng: np.random.RandomState,
    dtype: torch.dtype,
    device: torch.device,
):

    model_type = model_type.lower()

    if model_type in {"rnn", "vanilla", "vanillarnn"}:
        return VanillaRNN(
            nin=nin,
            nout=nout,
            nhid=nhid,
            init=init,
            classif_type=classif_type,
            rng=rng,
            dtype=dtype,
            device=device,
        )

    if model_type in {"gru"}:
        return GRUModel(
            nin=nin,
            nout=nout,
            nhid=nhid,
            init=init,
            classif_type=classif_type,
            rng=rng,
            dtype=dtype,
            device=device,
        )

    raise ValueError(f"Unknown model_type={model_type}")
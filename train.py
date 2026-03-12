
from model import spectral_radius, _tanh_saturation_distance, _sigmoid_saturation_distance
from model import make_model
from tasks import make_task, to_torch
import argparse
import time
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))


def _sigmoid_sat_dist(v: torch.Tensor) -> torch.Tensor:
    return _sigmoid_saturation_distance(v)


def _tanh_sat_dist(v: torch.Tensor) -> torch.Tensor:
    return _tanh_saturation_distance(v)


def _hidden_sat_time(model, h: torch.Tensor) -> torch.Tensor:
    act = getattr(model, "act_name", "tanh")
    if act == "sigmoid":
        d = _sigmoid_sat_dist(h)
    else:
        d = _tanh_sat_dist(h)
    return d.mean(dim=(1, 2))


def compute_loss_and_error(task, model, x, y_onehot, return_extras: bool = False):
    classif = task.classifType

    try:
        out = model(x, return_extras=return_extras)
    except TypeError:
        out = model(x)

    extras = None
    if isinstance(out, tuple) and len(out) == 3:
        logits_or_y, h, extras = out
    else:
        logits_or_y, h = out

    if classif == "lastSoftmax":
        y_idx = y_onehot.argmax(dim=1)
        loss = F.cross_entropy(logits_or_y, y_idx)
        pred = logits_or_y.argmax(dim=1)
        err = (pred != y_idx).float().mean()
        return loss, err, logits_or_y, h, extras

    if classif == "softmax":
        y_idx = y_onehot.argmax(dim=1)
        loss = F.cross_entropy(logits_or_y, y_idx)
        T = x.shape[0]
        B = x.shape[1]
        pred_idx = logits_or_y.argmax(dim=1).reshape(T, B)
        true_idx = y_idx.reshape(T, B)
        if getattr(task, "report", "last") == "all":
            wrong_any = (pred_idx != true_idx).sum(dim=0) > 0
            err = wrong_any.float().mean()
        else:
            err = (pred_idx[-1] != true_idx[-1]).float().mean()
        return loss, err, logits_or_y, h, extras

    if classif == "lastLinear":
        y = logits_or_y
        loss = ((y_onehot - y) ** 2).mean(dim=0).sum()
        err_abs = float(getattr(task, "err_abs", 0.2))
        err = ((y_onehot - y).abs().sum(dim=1) > err_abs).float().mean()
        return loss, err, y, h, extras

    raise ValueError(f"Unknown classifType={classif}")


def omega_regularizer_and_gradW_hh(model, loss: torch.Tensor, h: torch.Tensor, bound: float):
    dH = torch.autograd.grad(loss, h, create_graph=True, retain_graph=True)[0]
    d = dH[1:]
    ht = h[1:]
    deriv = model.act_deriv_from_h(ht)
    tmp = d * deriv

    Tm1, B, H = tmp.shape
    tmp2 = (tmp.reshape(Tm1 * B, H) @ model.W_hh.t()).reshape(Tm1, B, H)
    tmp_x = (tmp2 ** 2).sum(dim=2)
    tmp_y = (d ** 2).sum(dim=2)

    cond = (tmp_y >= bound).float()
    n_elems = cond.mean(dim=1)
    ratio = tmp_x / (tmp_y + 1e-30)
    ratio = torch.where(cond > 0.0, ratio, torch.ones_like(ratio))
    reg = (ratio - 1.0) ** 2

    reg_time_mean = reg.mean(dim=1)
    omega = reg_time_mean.sum() / (n_elems.sum() + 1e-30)

    steps_in_past = n_elems.mean()

    gW_hh = torch.autograd.grad(
        omega, model.W_hh, retain_graph=True, create_graph=False)[0]
    return omega.detach(), gW_hh.detach(), steps_in_past.detach()


def grad_time_profile(task, model, x: torch.Tensor, y_onehot: torch.Tensor, collect_extras: bool = False):
    model.zero_grad(set_to_none=True)
    loss, err, _, h, extras = compute_loss_and_error(
        task, model, x, y_onehot, return_extras=collect_extras)
    dH = torch.autograd.grad(loss, h, create_graph=True, retain_graph=True)[0]
    dH = dH.detach()
    g_t = torch.norm(dH, dim=2).mean(dim=1)
    a_t = model.act_deriv_from_h(h).mean(dim=(1, 2))
    sat_t = _hidden_sat_time(model, h)

    z_sat_t, r_sat_t = None, None
    if collect_extras and isinstance(extras, dict):

        if "z" in extras:
            z_sat_t = _sigmoid_sat_dist(extras["z"]).mean(dim=(1, 2))
        if "r" in extras:
            r_sat_t = _sigmoid_sat_dist(extras["r"]).mean(dim=(1, 2))
    return (
        loss.detach(),
        err.detach(),
        g_t.detach(),
        a_t.detach(),
        sat_t.detach(),
        None if z_sat_t is None else z_sat_t.detach(),
        None if r_sat_t is None else r_sat_t.detach(),
    )


def global_grad_norm(params):
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += (p.grad.detach() ** 2).sum().item()
    return np.sqrt(total_norm)


def clip_rescale(params, cutoff: float):
    norm = global_grad_norm(params)
    if not np.isfinite(norm) or norm < 0 or norm > 1e10:
        return norm, True, False, norm

    clipped = False
    if norm > cutoff and cutoff > 0:
        scale = cutoff / (norm + 1e-30)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
        clipped = True

    norm_post = global_grad_norm(params)
    return norm, False, clipped, norm_post


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="mem",
                   choices=["torder", "torder3", "add", "mul", "mem", "perm"])
    p.add_argument("--model", type=str, default="rnn",
                   choices=["rnn", "gru"], help="Recurrent cell type.")
    p.add_argument("--init", type=str, default="basic_tanh",
                   choices=["sigmoid", "basic_tanh", "smart_tanh", "test"])
    p.add_argument("--nhid", type=int, default=50)
    p.add_argument("--seed", type=int, default=52)

    p.add_argument("--alpha", type=float, default=2.0,
                   help="Omega regularizer strength (applied to W_hh only).")
    p.add_argument("--lr", type=float, default=0.01)

    p.add_argument("--min_length", type=int, default=50)
    p.add_argument("--max_length", type=int, default=200)

    p.add_argument("--bs", type=int, default=20)
    p.add_argument("--ebs", type=int, default=10000)
    p.add_argument("--cbs", type=int, default=1000)

    p.add_argument("--checkFreq", type=int, default=20)

    p.add_argument("--collectDiags", action="store_true")
    p.add_argument("--diagBins", type=int, default=50)
    p.add_argument("--satThresh", type=float, default=0.05)
    p.add_argument("--diagGates", action="store_true")
    p.add_argument("--bound", type=float, default=1e-20)
    p.add_argument("--err_abs", type=float, default=0.2)
    p.add_argument("--valid_seed", type=int, default=12345)

    p.add_argument("--clipstyle", type=str, default="rescale",
                   choices=["rescale", "nothing"])
    p.add_argument("--cutoff", type=float, default=1.0)

    p.add_argument("--maxiters", type=int, default=int(20000))
    p.add_argument("--saveFreq", type=float, default=5.0)
    p.add_argument("--name", type=str, default="test_torch")

    p.add_argument("--memvalues", type=int, default=5)
    p.add_argument("--mempos", type=int, default=10)
    p.add_argument("--memall", action="store_true")

    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda"])
    return p.parse_args()


def _effective_max_seq_len(args, task_obj) -> int:
    base = int(args.max_length)
    t = args.task
    if t == "mem" and hasattr(task_obj, "n_pos"):
        return base + 2 * int(task_obj.n_pos)
    if t in ("add", "mul"):
        return base + int(np.ceil(base * 0.1)) + 2
    return base


def main():
    args = parse_args()
    device = args.device if (
        args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    rng = np.random.RandomState(args.seed)

    task_kwargs = {}
    if args.task == "mem":
        task_kwargs = {"n_values": args.memvalues,
                       "n_pos": args.mempos, "generate_all": args.memall}
    task = make_task(args.task, rng, **task_kwargs)
    task.err_abs = float(args.err_abs)

    model = make_model(
        model_type=args.model,
        nin=task.nin,
        nout=task.nout,
        nhid=args.nhid,
        init=args.init,
        classif_type=task.classifType,
        rng=rng,
        dtype=torch.float32,
        device=device,
    ).to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    rng_valid = np.random.RandomState(args.seed + args.valid_seed)
    task_valid = make_task(args.task, rng_valid, **task_kwargs)
    task_valid.err_abs = float(args.err_abs)
    n_eval_chunks = max(1, args.ebs // args.cbs)
    valid_batches = []
    for _ in range(n_eval_chunks):
        if args.max_length > args.min_length:
            vlen = args.min_length + \
                rng_valid.randint(args.max_length - args.min_length)
        else:
            vlen = args.min_length
        vx_np, vy_np = task_valid.generate(args.cbs, vlen)
        vx = to_torch(vx_np, device).float()
        vy = to_torch(vy_np, device).float()
        valid_batches.append((vx, vy))

    rng_diag = np.random.RandomState(args.seed + args.valid_seed + 999)
    task_diag = make_task(args.task, rng_diag, **task_kwargs)
    task_diag.err_abs = float(args.err_abs)
    diag_len = int(args.max_length)
    diag_bs = int(min(args.bs, args.cbs))
    dx_np, dy_np = task_diag.generate(diag_bs, diag_len)
    dx = to_torch(dx_np, device).float()
    dy = to_torch(dy_np, device).float()

    print("Starting to train (PyTorch).")
    best_score = 100.0
    cont = True
    solved = 0
    n = -1

    store_space = max(1, args.maxiters // args.checkFreq)
    store_train = np.full((args.maxiters,), -1.0, dtype=np.float32)
    store_valid = np.full((store_space,), -1.0, dtype=np.float32)
    store_norm = np.full((args.maxiters,), -1.0, dtype=np.float32)
    store_rho = np.full((store_space,), -1.0, dtype=np.float32)
    store_reg = np.full((args.maxiters,), -1.0, dtype=np.float32)
    store_steps = np.full((args.maxiters,), -1.0, dtype=np.float32)
    diag_storage_len = _effective_max_seq_len(args, task_diag)
    store_grad_time = np.full(
        (store_space, diag_storage_len), np.nan, dtype=np.float32)
    store_act_time = np.full(
        (store_space, diag_storage_len), np.nan, dtype=np.float32)
    store_sat_time = np.full(
        (store_space, diag_storage_len), np.nan, dtype=np.float32)
    store_gate_z_sat_time = np.full(
        (store_space, diag_storage_len), np.nan, dtype=np.float32)
    store_gate_r_sat_time = np.full(
        (store_space, diag_storage_len), np.nan, dtype=np.float32)
    store_diag_loss = np.full((store_space,), -1.0, dtype=np.float32)
    store_diag_err = np.full((store_space,), -1.0, dtype=np.float32)

    avg_cost = 0.0
    avg_norm = 0.0
    avg_norm_post = 0.0
    avg_reg = 0.0
    avg_steps = 0.0
    avg_len = 0.0

    last_save = time.time()

    lr = args.lr

    while lr > 1e-8 and cont and n < args.maxiters - 1:
        n += 1
        if args.max_length > args.min_length:
            length = args.min_length + \
                rng.randint(args.max_length - args.min_length)
        else:
            length = args.min_length

        x_np, y_np = task.generate(args.bs, length)
        x = to_torch(x_np, device).float()
        y = to_torch(y_np, device).float()

        optimizer.zero_grad(set_to_none=True)

        loss, err, _, h, _ = compute_loss_and_error(task, model, x, y)

        omega = torch.tensor(0.0, device=device)
        gW_hh_reg = None
        steps_in_past = torch.tensor(float("nan"), device=device)
        supports_omega = bool(getattr(model, "supports_omega", False))
        if args.alpha > 0 and supports_omega:
            omega, gW_hh_reg, steps_in_past = omega_regularizer_and_gradW_hh(
                model, loss, h, args.bound)

        loss.backward()

        if args.alpha > 0 and supports_omega and gW_hh_reg is not None:
            if getattr(model, "W_hh", None) is not None:
                if model.W_hh.grad is None:
                    model.W_hh.grad = torch.zeros_like(model.W_hh)
                model.W_hh.grad = model.W_hh.grad + \
                    float(args.alpha) * gW_hh_reg.to(device=device)

        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        if args.clipstyle == "rescale":
            norm, bad, clipped, norm_post = clip_rescale(
                list(model.parameters()), args.cutoff)
            if bad:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    Wrec = getattr(model, "recurrent_weight_for_rho", None)
                    if callable(Wrec):
                        W = Wrec()
                        if isinstance(W, torch.Tensor) and W.grad is not None:
                            W.grad = 0.02 * W.detach()
                    elif getattr(model, "W_hh", None) is not None and model.W_hh.grad is not None:
                        model.W_hh.grad = 0.02 * model.W_hh.detach()
                norm = global_grad_norm(list(model.parameters()))
        else:
            norm = global_grad_norm(list(model.parameters()))
            norm_post = 0

        optimizer.step()

        tr_cost = float(loss.detach().cpu())
        store_train[n] = tr_cost
        store_norm[n] = float(norm)
        store_reg[n] = float(omega.cpu().item()) if args.alpha > 0 else 0.0
        store_steps[n] = float(steps_in_past.cpu().item()
                               ) if args.alpha > 0 else 0.0

        avg_cost += tr_cost
        avg_norm += float(norm)
        avg_norm_post += float(norm_post)
        avg_reg += float(store_reg[n])
        avg_steps += float(store_steps[n])
        avg_len += float(length)

        if n % args.checkFreq == 0 and n > 0:
            avg_cost /= float(args.checkFreq)
            avg_norm /= float(args.checkFreq)
            avg_norm_post /= float(args.checkFreq)
            avg_reg /= float(args.checkFreq)
            avg_steps /= float(args.checkFreq)
            avg_len /= float(args.checkFreq)

            valid_cost = 0.0
            valid_err = 0.0
            valid_mae = 0.0
            valid_maxae = 0.0
            valid_err_005 = 0.0
            valid_err_010 = 0.0
            valid_err_abs = 0.0

            n_eval_chunks = len(valid_batches)
            model.eval()
            with torch.no_grad():
                for (vx, vy) in valid_batches:
                    vloss, verr, vout, _, _ = compute_loss_and_error(
                        task_valid, model, vx, vy)
                    valid_cost += float(vloss.detach().cpu())
                    valid_err += float(verr.detach().cpu())
                    if task_valid.classifType == "lastLinear":
                        abs_err = (vy - vout).abs()
                        valid_mae += float(abs_err.mean().detach().cpu())
                        valid_maxae = max(valid_maxae, float(
                            abs_err.max().detach().cpu()))
                        valid_err_005 += float((abs_err >
                                               0.05).float().mean().detach().cpu())
                        valid_err_010 += float((abs_err >
                                               0.10).float().mean().detach().cpu())
                        valid_err_abs += float(
                            (abs_err > float(task_valid.err_abs)).float().mean().detach().cpu())
            model.train()

            valid_cost /= float(n_eval_chunks)
            valid_err = (valid_err / float(n_eval_chunks)) * 100.0
            if task_valid.classifType == "lastLinear":
                valid_mae /= float(n_eval_chunks)
                valid_err_005 = (valid_err_005 / float(n_eval_chunks)) * 100.0
                valid_err_010 = (valid_err_010 / float(n_eval_chunks)) * 100.0
                valid_err_abs = (valid_err_abs / float(n_eval_chunks)) * 100.0

            try:
                Wrec = model.recurrent_weight_for_rho()
            except Exception:
                Wrec = getattr(model, "W_hh")
            Whh_np = Wrec.detach().cpu().numpy()
            rho = float(np.max(np.abs(np.linalg.eigvals(Whh_np))))

            pos = n // args.checkFreq
            if pos < store_space:
                store_valid[pos] = float(valid_err)
                store_rho[pos] = float(rho)

                if args.collectDiags:
                    model.eval()
                    dloss, derr, g_t, a_t, sat_t, zsat_t, rsat_t = grad_time_profile(
                        task_diag, model, dx, dy, collect_extras=args.diagGates
                    )
                    model.train()

                    Tdiag = int(g_t.shape[0])
                    Tstore = min(Tdiag, diag_storage_len)

                    store_grad_time[pos, :Tstore] = g_t[:Tstore].detach(
                    ).cpu().numpy().astype(np.float32)
                    store_act_time[pos, :Tstore] = a_t[:Tstore].detach(
                    ).cpu().numpy().astype(np.float32)
                    store_sat_time[pos, :Tstore] = sat_t[:Tstore].detach(
                    ).cpu().numpy().astype(np.float32)
                    if zsat_t is not None:
                        store_gate_z_sat_time[pos, :Tstore] = zsat_t[:Tstore].detach(
                        ).cpu().numpy().astype(np.float32)
                    if rsat_t is not None:
                        store_gate_r_sat_time[pos, :Tstore] = rsat_t[:Tstore].detach(
                        ).cpu().numpy().astype(np.float32)
                    store_diag_loss[pos] = float(dloss.detach().cpu())
                    store_diag_err[pos] = float(derr.detach().cpu())

                    eps = 1e-12
                    g_np = g_t[:Tstore].detach().cpu().numpy()
                    s_np = sat_t[:Tstore].detach().cpu().numpy()
                    g_l = np.log10(g_np + eps)
                    g_hist, g_edges = np.histogram(g_l, bins=args.diagBins)
                    s_hist, s_edges = np.histogram(
                        s_np, bins=args.diagBins, range=(0.0, 1.0))
                    sat_frac = float((s_np < args.satThresh).mean()) * 100.0
                    print(
                        f"  [diag] log10|dL/dh_t| hist bins={args.diagBins} range=({g_edges[0]:.2f},{g_edges[-1]:.2f}) "
                        f"mean={g_l.mean():.2f} p50={np.median(g_l):.2f} p05={np.quantile(g_l, 0.05):.2f} p95={np.quantile(g_l, 0.95):.2f}"
                    )
                    print(
                        f"  [diag] sat-dist hist bins={args.diagBins} sat<{args.satThresh:g}={sat_frac:.1f}% "
                        f"mean={s_np.mean():.3f} p10={np.quantile(s_np, 0.1):.3f} p90={np.quantile(s_np, 0.9):.3f}"
                    )
                    if zsat_t is not None and rsat_t is not None:
                        z_np = zsat_t[:Tstore].detach().cpu().numpy()
                        r_np = rsat_t[:Tstore].detach().cpu().numpy()
                        z_hist, z_edges = np.histogram(
                            z_np, bins=args.diagBins, range=(0.0, 1.0))
                        r_hist, r_edges = np.histogram(
                            r_np, bins=args.diagBins, range=(0.0, 1.0))
                        print(
                            f"  [diag] z-gate sat-dist hist bins={args.diagBins} "
                            f"mean={z_np.mean():.3f} p10={np.quantile(z_np, 0.1):.3f} p90={np.quantile(z_np, 0.9):.3f}"
                        )
                        print(
                            f"  [diag] r-gate sat-dist hist bins={args.diagBins} "
                            f"mean={r_np.mean():.3f} p10={np.quantile(r_np, 0.1):.3f} p90={np.quantile(r_np, 0.9):.3f}"
                        )

            extra = f", valid nll {valid_cost:05.3f}"
            if task_valid.classifType == "lastLinear":
                extra += (
                    f", valid mae {valid_mae:05.3f}"
                    f", err@0.05 {valid_err_005:06.2f}%"
                    f", err@0.10 {valid_err_010:06.2f}%"
                    f", err@{float(task_valid.err_abs):.2f} {valid_err_abs:06.2f}%"
                    f", maxae {valid_maxae:05.3f}"
                )
            print(
                f"Iter {n:07d}: "
                f"train nll {avg_cost:05.3f}, "
                f"valid error {valid_err:07.3f}%, "
                f"best valid error {best_score:07.3f}%, "
                f"avg grad norm {avg_norm:7.3f}, "
                f"avg grad norm (post clip) {avg_norm_post:7.3f}, "
                f"rho_Whh {rho:5.2f}, "
                f"Omega {avg_reg:5.2f}, "
                f"alpha {args.alpha:6.3f}, "
                f"steps in the past {avg_steps:05.3f}"
                + extra
            )

            if valid_err < best_score:
                best_score = valid_err

            if valid_err < 0.0001 and np.isfinite(valid_cost):
                cont = False
                solved = 1
                print("!!!!! STOPPING - Problem solved")

            if (time.time() - last_save) > args.saveFreq * 60.0:
                save_npz(
                    args.name + "_state.npz",
                    store_train, store_valid, store_norm, store_rho, store_reg, store_steps,
                    store_grad_time, store_act_time, store_sat_time,
                    store_gate_z_sat_time, store_gate_r_sat_time,
                    store_diag_loss, store_diag_err,
                    diag_len, diag_bs, args.checkFreq,
                    model,
                )
                last_save = time.time()

            avg_cost = avg_norm = avg_reg = avg_steps = avg_len = 0.0

    save_npz(
        args.name + "_final_state.npz",
        store_train, store_valid, store_norm, store_rho, store_reg, store_steps,
        store_grad_time, store_act_time, store_sat_time,
        store_gate_z_sat_time, store_gate_r_sat_time,
        store_diag_loss, store_diag_err,
        diag_len, diag_bs, args.checkFreq,
        model,
    )
    print(
        f"Done. solved={solved} steps={n} best_valid_error={best_score:.4f}%")


def save_npz(
    path,
    train_nll,
    valid_error,
    gradient_norm,
    rho_Whh,
    Omega,
    steps_in_past,
    grad_time,
    act_time,
    sat_time,
    gate_z_sat_time,
    gate_r_sat_time,
    diag_loss,
    diag_err,
    diag_len,
    diag_bs,
    checkFreq,
    model,
):
    payload = dict(
        train_nll=train_nll,
        valid_error=valid_error,
        gradient_norm=gradient_norm,
        rho_Whh=rho_Whh,
        Omega=Omega,
        steps_in_past=steps_in_past,
        grad_time=grad_time,
        act_time=act_time,
        sat_time=sat_time,
        gate_z_sat_time=gate_z_sat_time,
        gate_r_sat_time=gate_r_sat_time,
        diag_loss=diag_loss,
        diag_err=diag_err,
        diag_len=int(diag_len),
        diag_bs=int(diag_bs),
        checkFreq=int(checkFreq),
    )
    if hasattr(model, "numpy_state"):
        payload.update(model.numpy_state())
    else:
        for k, v in model.state_dict().items():
            if isinstance(v, torch.Tensor):
                payload[f"state__{k}"] = v.detach().cpu().numpy()

    np.savez(path, **payload)


if __name__ == "__main__":
    main()

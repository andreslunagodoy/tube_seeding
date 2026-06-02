"""Seed-selection CMA-ES for I(1,...,1, 0, -6, -6) on the maximal cut [1..8],
mirroring the (0,0,-12) recipe of paper §"Evolutionary strategies":

  - candidate space: bounding box B in (-a_9, -a_10, -a_11),
        a_9 in [0..6], a_10 in [0..12], a_11 in [0..12]
        ( = the (0,6,6) zig-zag tube reach (4, ~10, ~10) plus +2 each side ).
  - on max cut + no-dots, each candidate is the 11-tuple
        (1,1,1,1,1,1,1,1, -a_9, -a_10, -a_11).
  - small MLP (3 -> 16 -> 8 -> 1) scores each candidate by its
    (a_9, a_10, a_11) coords; CMA-ES tunes the MLP weights.
  - selection rule: include candidates with score > 0.
  - fitness:
        if reduction closes to masters: f = n_seeds
        else: f = LARGE + n_survivors  (graded penalty toward closure)
  - pretraining: MLP first mimics a tube heuristic
        score = halfwidth - dist((a_9, a_10, a_11), segment 0 -> (0,6,6)).

Parallelised across N workers via multiprocessing (spawn context, because
JuliaCall is not fork-safe).  Each worker loads pyfeyngym, recomputes
masters and pre-generates candidate IBP equations once at startup; then
the CMA-ES driver in the main process dispatches each population to
the pool via `pool.map(eval_one, xs_with_weights)`.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Tuple

import numpy as np

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

# ---------------------------------------------------------------------------
# Problem constants (visible in workers because they're imported at module load)
# ---------------------------------------------------------------------------
DP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IBP_FILE = DP_DIR + "/IBP_LI"
TRIV_FILE = DP_DIR + "/trivialsector"
MODULUS = 2**31 - 1
M_VALS = {"d": 23, "m1": 3, "m2": 5, "m3": 17, "m4": 23}
TOP_SECTOR = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
CUT = [1, 3, 4, 6, 7, 8]                  # 2-uncut cut (props 2 and 5 uncut)
N_INDICES = 11
TARGET = (1, 1, 1, 1, 1, 1, 1, 1, 0, -6, -6)

BOX_X, BOX_Y, BOX_Z = 2, 7, 7     # ISP box (tighter than 1-uncut to stay within RAM)
A2_MIN, A2_MAX = -2, 1            # uncut-prop range for prop 2
A5_MIN, A5_MAX = -2, 1            # uncut-prop range for prop 5
LARGE = 10_000.0

# Natural master basis for the (0,-6,-6) reduction on cut [1,3,4,6,7,8]
# (15 elements): the 9 max-cut masters + 6 sub-sector/dotted corners that
# the two relaxed propagators 2 and 5 open up.  Obtained by full-box
# reduction with no keep_on_rhs.
NATURAL_MASTERS = [
    (1, -1, 1, 1,  1, 1, 1, 1,  0,  0,  0),
    (1,  0, 1, 1,  0, 1, 1, 1,  0,  0,  0),
    (1,  0, 1, 1,  1, 1, 1, 1,  0,  0,  0),
    (1,  1, 1, 1, -1, 1, 1, 1,  0,  0,  0),
    (1,  1, 1, 1,  0, 1, 1, 1, -1,  0,  0),
    (1,  1, 1, 1,  0, 1, 1, 1,  0,  0,  0),
    (1,  1, 1, 1,  1, 1, 1, 1, -2,  0,  0),
    (1,  1, 1, 1,  1, 1, 1, 1, -1, -1,  0),
    (1,  1, 1, 1,  1, 1, 1, 1, -1,  0, -1),
    (1,  1, 1, 1,  1, 1, 1, 1, -1,  0,  0),
    (1,  1, 1, 1,  1, 1, 1, 1,  0, -2,  0),
    (1,  1, 1, 1,  1, 1, 1, 1,  0, -1, -1),
    (1,  1, 1, 1,  1, 1, 1, 1,  0, -1,  0),
    (1,  1, 1, 1,  1, 1, 1, 1,  0,  0, -2),
    (1,  1, 1, 1,  1, 1, 1, 1,  0,  0, -1),
]

# Per-worker state (populated by worker_init)
_state = {}


def all_candidates():
    """All candidates: (a_2, a_5, x, y, z) with x = -a_9, y = -a_10, z = -a_11,
    expressed as 11-tuples (1, a_2, 1, 1, a_5, 1, 1, 1, -x, -y, -z).  Includes
    the target (a_2=1, a_5=1, x=0, y=6, z=6) as a valid seed.  The corner
    integral (a_2=1, a_5=1, x=0, y=0, z=0) is omitted as it is the trivial
    top-sector master."""
    cands = []
    for a2 in range(A2_MIN, A2_MAX + 1):
        for a5 in range(A5_MIN, A5_MAX + 1):
            for x in range(BOX_X + 1):
                for y in range(BOX_Y + 1):
                    for z in range(BOX_Z + 1):
                        if (a2, a5, x, y, z) == (1, 1, 0, 0, 0):
                            continue
                        cands.append(((a2, a5, x, y, z),
                                      (1, a2, 1, 1, a5, 1, 1, 1, -x, -y, -z)))
    return cands


def tube_distance(a2, a5, x, y, z, end=(0, 6, 6)):
    """Distance from a 5-D point (a_2, a_5, x, y, z) to the segment from the
    origin in (-a_9,-a_10,-a_11) to `end`, plus penalties for each uncut
    propagator dipping away from a_i = 1 (the max-cut "no-dot" prescription).
    The tube heuristic thus prefers candidates near the (a_2=1, a_5=1) top
    sector and within the diagonal tube."""
    ex, ey, ez = end
    elen2 = ex * ex + ey * ey + ez * ez
    t = (x * ex + y * ey + z * ez) / elen2
    t = max(0.0, min(1.0, t))
    px, py, pz = t * ex, t * ey, t * ez
    d3 = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
    return d3 + abs(a2 - 1) + abs(a5 - 1)


# ---------------------------------------------------------------------------
# Worker initialization (called once per worker process)
# ---------------------------------------------------------------------------
def worker_init():
    import pyfeyngym as pfg
    import torch
    import torch.nn as nn

    eq_templates = pfg.gen_eq_templates(IBP_FILE, M_VALS)
    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=CUT, n_indices=N_INDICES)

    # Use the natural max-cut basis for the (0,-6,-6) target (hardcoded above).
    masters = list(NATURAL_MASTERS)
    masters_set = set(masters)

    # Candidate pool
    cands = all_candidates()
    coords = [c[0] for c in cands]
    seeds_full = [c[1] for c in cands]
    keep = [j for j, s in enumerate(seeds_full)
            if pfg.to_sector(s) not in trivial_sectors]
    coords = [coords[j] for j in keep]
    seeds_full = [seeds_full[j] for j in keep]

    # Pre-generate all candidate equations
    seed_op_eq_list, all_variables = pfg.gen_eqs(
        eq_templates, trivial_sectors, M_VALS, seeds_full)
    sorted_vars_full = pfg.sort_integrals_desc(all_variables)
    seed_to_idx = {s: j for j, s in enumerate(seeds_full)}
    candidate_to_eqs = [[] for _ in seeds_full]
    for (integral, op, eq) in seed_op_eq_list:
        j = seed_to_idx.get(tuple(integral))
        if j is not None:
            candidate_to_eqs[j].append(eq)

    # Network (same architecture as nn_optimize_strip.py)
    class ScoringNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, 16), nn.GELU(),
                nn.Linear(16, 8), nn.GELU(),
                nn.Linear(8, 1),
            )
        def forward(self, x): return self.net(x)

    network = ScoringNetwork()
    coords_t = torch.tensor(coords, dtype=torch.float32)

    _state.update(
        pfg=pfg, torch=torch, network=network,
        coords=coords, coords_t=coords_t, seeds_full=seeds_full,
        candidate_to_eqs=candidate_to_eqs, sorted_vars_full=sorted_vars_full,
        masters=masters, masters_set=masters_set,
    )


def assign_params(model, x: np.ndarray, torch_mod):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        w = torch_mod.from_numpy(x[idx:idx + n].astype(np.float32)).view_as(p)
        p.data.copy_(w); idx += n


def flatten_params(model) -> np.ndarray:
    return np.concatenate([p.data.view(-1).cpu().numpy() for p in model.parameters()])


def closure_eval(chosen_idx):
    """Run the modular solve on the selected subset and return (n_seeds, n_survivors)."""
    pfg = _state["pfg"]
    target = TARGET
    masters_set = _state["masters_set"]
    cand2eqs = _state["candidate_to_eqs"]
    eqs = []
    for j in chosen_idx:
        eqs.extend(cand2eqs[j])
    if not eqs:
        return 0, LARGE
    try:
        sol = pfg.solve_eqs_modulo(
            eqs, _state["sorted_vars_full"], MODULUS,
            keep_on_rhs=list(masters_set),
            complete_pivoting=False,
            needed_variables={target},
        )
    except Exception:
        return len(chosen_idx), LARGE
    red = sol.get(target)
    if red is None:
        return len(chosen_idx), LARGE
    surv = set(t[0] for t in red) - masters_set
    return len(chosen_idx), len(surv)


def eval_one(x):
    """Worker entry point: given an MLP-parameter vector, score all candidates,
    threshold at 0, run the closure check, return (fitness, n_seeds, n_survivors)."""
    network = _state["network"]
    torch_mod = _state["torch"]
    coords_t = _state["coords_t"]
    seeds_full = _state["seeds_full"]
    assign_params(network, np.asarray(x, dtype=np.float32), torch_mod)
    network.eval()
    with torch_mod.no_grad():
        sc = network(coords_t).squeeze(1).cpu().numpy()
    chosen = [j for j in range(len(seeds_full)) if sc[j] > 0]
    n_seeds, surv = closure_eval(chosen)
    f = float(n_seeds) if surv == 0 else LARGE + float(surv)
    return f, n_seeds, surv


# ---------------------------------------------------------------------------
# Pretraining (main process only — produces x0 for CMA-ES)
# ---------------------------------------------------------------------------
def pretrain_x0(coords, halfwidth=6.0, epochs=80, seed=42):
    """Pretrain MLP to a BINARY tube heuristic: +1 if within `halfwidth` of the
    segment 0 -> (0,6,6), else -1. Wider tube (halfwidth=6) labels enough
    candidates as +1 that the MLP doesn't collapse to all-negative outputs."""
    import torch
    import torch.nn as nn
    class ScoringNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, 16), nn.GELU(),
                nn.Linear(16, 8), nn.GELU(),
                nn.Linear(8, 1),
            )
        def forward(self, x): return self.net(x)
    torch.manual_seed(seed)
    network = ScoringNetwork()
    X = torch.tensor(coords, dtype=torch.float32)
    # coords are (a_4, x, y, z) tuples
    targets = [1.0 if tube_distance(*c) <= halfwidth else -1.0 for c in coords]
    n_pos = sum(1 for t in targets if t > 0)
    print(f"  pretrain labels: {n_pos}/{len(targets)} positive (halfwidth={halfwidth})",
          flush=True)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    network.train()
    opt = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(network(xb), yb)
            loss.backward(); opt.step()
    network.eval()
    # sanity-check: how many candidates does the pretrained MLP score > 0?
    with torch.no_grad():
        sc = network(X).squeeze(1).numpy()
    print(f"  pretrained MLP scores: min={sc.min():.3f} max={sc.max():.3f} "
          f"mean={sc.mean():.3f}  > 0: {(sc > 0).sum()}/{len(sc)}", flush=True)
    return flatten_params(network)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=2000)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--mutation-size", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-halfwidth", type=float, default=3.0)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--load-x0", type=str, default=None,
                        help="Path to a .npy file with a saved MLP parameter "
                             "vector to use as the CMA-ES starting point "
                             "(skips pretraining).")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "cmaes_066_uncut25")
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== CMA-ES seed selection for {TARGET} on cut {CUT} ===", flush=True)
    print(f"box: a_2∈[{A2_MIN},{A2_MAX}], a_5∈[{A5_MIN},{A5_MAX}], x∈[0,{BOX_X}], y∈[0,{BOX_Y}], z∈[0,{BOX_Z}]",
          flush=True)
    print(f"workers={args.workers}, popsize={args.popsize}", flush=True)

    # --- main-process candidate enumeration (just to compute x0 dimensions) ---
    cands = all_candidates()
    coords = [c[0] for c in cands]   # NB: this is the BOX candidates; trivial-sector
                                      # filter is reapplied inside workers, so coords
                                      # here may include a few extra that workers drop.
    print(f"  {len(coords)} candidates in box (before trivial-sector filter)", flush=True)

    # --- choose x0: either pretrain or load from disk ---
    if args.load_x0:
        print(f"\nLoading x0 from {args.load_x0}...", flush=True)
        x0 = np.load(args.load_x0)
        n_params = len(x0)
        print(f"  loaded x0: {n_params} parameters", flush=True)
    else:
        print(f"\nPretraining MLP (halfwidth={args.pretrain_halfwidth} around segment 0->(0,6,6))...",
              flush=True)
        x0 = pretrain_x0(coords, halfwidth=args.pretrain_halfwidth, seed=args.seed)
        n_params = len(x0)
        print(f"  pretrained x0: {n_params} parameters", flush=True)

    # --- spawn worker pool, init each worker ---
    print(f"\nSpawning {args.workers} workers (each loads pyfeyngym, ~30-60s)...",
          flush=True)
    t_init = time.time()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=worker_init)
    # Synchronize by sending a no-op so we know workers are up
    init_check = pool.map(eval_one, [x0] * args.workers)
    f_pre, n_pre, surv_pre = init_check[0]
    print(f"  workers up in {time.time()-t_init:.0f}s", flush=True)
    print(f"  pretrain x0 eval: f={f_pre:.1f} n_seeds={n_pre} survivors={surv_pre}",
          flush=True)

    if all(r[2] != 0 for r in init_check):
        # Try the FULL-box baseline to confirm the candidate pool can close at all.
        x_all = np.ones(n_params)              # not a great "all-positive" trick
        # Use a network that always returns +1 by setting all weights large positive.
        # Simpler: directly call closure_eval with all indices via a special worker entry.
        print("  pretrain doesn't close — running explicit full-box baseline check...",
              flush=True)
        f_all, n_all, surv_all = pool.apply(_full_baseline_check)
        print(f"  full-box baseline: n_seeds={n_all}  survivors={surv_all}",
              flush=True)
        if surv_all != 0:
            print("  *** Full box does NOT close (1 or more survivors). "
                  "Box is too small or sub-sector coverage missing. Stopping. ***",
                  flush=True)
            pool.close(); pool.join()
            return

    # --- CMA-ES with parallel population evaluation ---
    import cma
    sigma0 = args.mutation_size * (float(np.mean(np.abs(x0))) + 1.0)
    # Disable the flat-fitness early-stop (default 1 generation, too tight) and
    # loosen tolfun / tolx so we don't bail before exploring.
    options = {
        "maxfevals": args.max_evals, "seed": args.seed,
        "verb_log": 0, "verb_disp": 0, "popsize": args.popsize,
        "tolflatfitness": 200,
        "tolfun": 1e-3, "tolfunhist": 1e-3, "tolx": 1e-4,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    best = {"f": float("inf"), "n_seeds": None, "survivors": None, "x": None}
    eval_count = 0
    gen = 0
    t_loop = time.time()
    print(f"\nCMA-ES starting (sigma0={sigma0:.4f}, popsize={args.popsize})...",
          flush=True)

    while not es.stop() and eval_count < args.max_evals:
        xs = es.ask()
        results = pool.map(eval_one, xs)
        fs = [r[0] for r in results]
        es.tell(xs, fs)
        eval_count += len(xs); gen += 1
        # Track best
        for r, x in zip(results, xs):
            if r[0] < best["f"]:
                best.update({"f": r[0], "n_seeds": r[1], "survivors": r[2],
                              "x": np.array(x, copy=True)})
                print(f"  [gen {gen:3d}, eval {eval_count:5d}, "
                      f"t={time.time()-t_loop:6.0f}s] NEW BEST "
                      f"f={r[0]:.1f}  n_seeds={r[1]}  survivors={r[2]}",
                      flush=True)
                # Checkpoint immediately so a kill doesn't lose progress
                try:
                    np.save(os.path.join(save_dir, "best_x.npy"), best["x"])
                    with open(os.path.join(save_dir, "result.json"), "w") as fp:
                        json.dump({"target": list(TARGET), "cut": CUT,
                                   "box": [A2_MIN, A2_MAX, A5_MIN, A5_MAX, BOX_X, BOX_Y, BOX_Z],
                                   "checkpoint": True,
                                   "current_best_n_seeds": best["n_seeds"],
                                   "current_best_survivors": best["survivors"],
                                   "evals_so_far": eval_count,
                                   "elapsed_s": round(time.time() - t_loop, 1)},
                                   fp, indent=2)
                except Exception:
                    pass
        # heartbeat every 5 gens
        if gen % 5 == 0:
            min_f = min(fs); med_f = float(np.median(fs))
            print(f"  [gen {gen:3d}, eval {eval_count:5d}, "
                  f"t={time.time()-t_loop:6.0f}s] gen-min f={min_f:.1f} "
                  f"med f={med_f:.1f} best f={best['f']:.1f}", flush=True)

    print(f"\nCMA-ES finished: {eval_count} evals, {time.time()-t_loop:.0f}s",
          flush=True)
    print(f"  best: n_seeds={best['n_seeds']}, survivors={best['survivors']}",
          flush=True)

    # --- save best network + final readback (also from a worker) ---
    if best["x"] is not None:
        f_final, n_final, surv_final = pool.apply(eval_one, (best["x"],))
        # Pull chosen seed list from a worker
        chosen_seeds = pool.apply(_extract_chosen_seeds, (best["x"],))
    else:
        f_final = n_final = surv_final = None
        chosen_seeds = []

    out = {
        "target": list(TARGET), "cut": CUT, "box": [A2_MIN, A2_MAX, A5_MIN, A5_MAX, BOX_X, BOX_Y, BOX_Z],
        "n_candidates": len(coords),
        "n_evals": eval_count,
        "wallclock_s": round(time.time() - t_loop, 1),
        "pretrain_n_seeds": n_pre, "pretrain_survivors": surv_pre,
        "final_n_seeds": n_final, "final_survivors": surv_final,
        "final_seeds_xyz": chosen_seeds,
    }
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(save_dir, "best_x.npy"), best["x"])
    print(f"\nSaved {save_dir}/result.json  (+ best_x.npy)", flush=True)

    pool.close(); pool.join()


def _full_baseline_check():
    """Worker side: confirm full-box closure (all candidates selected)."""
    n = len(_state["seeds_full"])
    return (LARGE, *closure_eval(list(range(n))))


def _extract_chosen_seeds(x):
    """Worker side: given MLP weights, return the list of chosen (x,y,z) coords."""
    network = _state["network"]
    torch_mod = _state["torch"]
    coords_t = _state["coords_t"]
    coords = _state["coords"]
    assign_params(network, np.asarray(x, dtype=np.float32), torch_mod)
    network.eval()
    with torch_mod.no_grad():
        sc = network(coords_t).squeeze(1).cpu().numpy()
    return [list(coords[j]) for j in range(len(coords)) if sc[j] > 0]


if __name__ == "__main__":
    main()

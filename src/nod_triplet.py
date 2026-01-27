from pathlib import Path
import json
import numpy as np
import pandas as pd

# ----------------------------
# HARD-CODE PATHS HERE
# ----------------------------
GROUP_RDM_DIR = Path("T:\\multimodal_brain_inspired\\BI\\triplet")         # contains early_group_rdm_vector.npy, etc.
OUT_BASE_DIR  = Path("T:\\multimodal_brain_inspired\\BI\\triplet\\csv")   # where triplet CSVs will be written
FILENAMES_TXT = Path("C:\\Users\\BrainInspired\\Documents\\GitHub\\NaturalObjectDataset-Processing\\Nick_RDMs\\outputs\\sub-02\\filenames.txt")
# ----------------------------

# Which group RDMs to generate triplets for:
GROUPS = ["early", "mid", "late", "all"]  # or ["all"] to start simple

# Triplet mining hyperparams
TRIPLETS_PER_ANCHOR = 50   # total triplets per anchor category (e.g., 1000 anchors -> 50k triplets)
POS_FRAC = 0.03            # positives sampled from closest 3% (lowest dissimilarity)
NEG_TAIL_FRAC = 0.05       # negatives sampled from farthest 5% (highest dissimilarity)
SEED = 42

# Split fractions over TRIPLETS (not anchors)
TRAIN_FRAC = 0.90
VAL_FRAC   = 0.05
TEST_FRAC  = 0.05

def load_idx_to_cat(filenames_path: Path) -> list[str]:
    lines = [ln.strip() for ln in filenames_path.read_text().splitlines() if ln.strip()]
    cats = []
    for ln in lines:
        if "\\n" in ln:
            cat = ln.split("\\n", 1)[0]
        else:
            # fallback: if the file is in a different format, take first token
            cat = ln.split()[0]
        cats.append(cat)
    return cats

def infer_n_from_vec_len(L: int) -> int:
    # Solve L = N*(N-1)/2
    disc = 1 + 8 * L
    n = int((1 + int(np.sqrt(disc))) // 2)
    if n * (n - 1) // 2 != L:
        raise ValueError(f"Vector length {L} is not a valid N*(N-1)/2.")
    return n


def rdm_vector_to_square(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    L = int(vec.shape[0])
    N = infer_n_from_vec_len(L)

    rdm = np.zeros((N, N), dtype=np.float32)
    iu = np.triu_indices(N, k=1)  # upper triangle indices
    rdm[iu] = vec.astype(np.float32)
    rdm[(iu[1], iu[0])] = rdm[iu]  # mirror to lower triangle
    np.fill_diagonal(rdm, 0.0)
    return rdm


def mine_triplets_from_rdm(rdm: np.ndarray, triplets_per_anchor: int,
                           pos_frac: float, neg_tail_frac: float,
                           rng: np.random.Generator) -> np.ndarray:
    """
    rdm: NxN dissimilarity matrix (smaller = more similar)
    returns: array of shape [num_triplets, 3] with columns (anchor, pos, neg)
    """
    N = rdm.shape[0]
    if rdm.shape[1] != N:
        raise ValueError("RDM must be square.")
    if N < 3:
        raise ValueError("Need at least 3 conditions to form triplets.")

    # Pools sizes
    pos_k = max(1, int(round(pos_frac * (N - 1))))
    neg_k = max(1, int(round(neg_tail_frac * (N - 1))))

    triplets = []

    for a in range(N):
        d = rdm[a].copy()
        d[a] = np.inf  # exclude self

        # Sort by dissimilarity ascending: closest first
        order = np.argsort(d)  # length N, includes a somewhere but d[a]=inf pushes it to end

        pos_pool = order[:pos_k]                 # low dissimilarity
        neg_pool = order[-neg_k:]                # high dissimilarity

        # Safety: ensure pools don't overlap / contain anchor
        pos_pool = pos_pool[pos_pool != a]
        neg_pool = neg_pool[(neg_pool != a) & (~np.isin(neg_pool, pos_pool))]

        if len(pos_pool) == 0 or len(neg_pool) == 0:
            continue

        # Sample triplets for this anchor
        for _ in range(triplets_per_anchor):
            p = int(rng.choice(pos_pool))
            n = int(rng.choice(neg_pool))
            triplets.append((a, p, n))

    if not triplets:
        raise RuntimeError("No triplets mined. Check your RDM or parameters.")
    return np.array(triplets, dtype=np.int32)


def split_and_save_triplets_as_categories(triplets: np.ndarray, idx_to_cat: list[str], out_dir: Path, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # shuffle
    rng = np.random.default_rng(seed)
    triplets = triplets[rng.permutation(len(triplets))]

    n_total = len(triplets)
    n_train = int(round(TRAIN_FRAC * n_total))
    n_val   = int(round(VAL_FRAC * n_total))

    train = triplets[:n_train]
    val   = triplets[n_train:n_train + n_val]
    test  = triplets[n_train + n_val:]

    def to_cat_df(arr: np.ndarray) -> pd.DataFrame:
        a = [idx_to_cat[i] for i in arr[:, 0]]
        p = [idx_to_cat[i] for i in arr[:, 1]]
        n = [idx_to_cat[i] for i in arr[:, 2]]
        return pd.DataFrame({"anchor_cat": a, "positive_cat": p, "negative_cat": n})

    to_cat_df(train).to_csv(out_dir / "train_triplets.csv", index=False)
    to_cat_df(val).to_csv(out_dir / "val_triplets.csv", index=False)
    to_cat_df(test).to_csv(out_dir / "test_triplets.csv", index=False)

    meta = {
        "n_total": n_total,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": seed,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

def main():
    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for gname in GROUPS:
        vec_path = GROUP_RDM_DIR / f"{gname}_group_rdm_vector.npy"
        if not vec_path.exists():
            print(f"[SKIP] Missing: {vec_path}")
            continue

        vec = np.load(vec_path)
        rdm = rdm_vector_to_square(vec)

        rng = np.random.default_rng(SEED)
        triplets = mine_triplets_from_rdm(
            rdm=rdm,
            triplets_per_anchor=TRIPLETS_PER_ANCHOR,
            pos_frac=POS_FRAC,
            neg_tail_frac=NEG_TAIL_FRAC,
            rng=rng,
        )

        out_dir = OUT_BASE_DIR / gname
        idx_to_cat = load_idx_to_cat(FILENAMES_TXT)
        split_and_save_triplets_as_categories(triplets, idx_to_cat, out_dir, seed=SEED)

    print("\nDone.")


if __name__ == "__main__":
    main()

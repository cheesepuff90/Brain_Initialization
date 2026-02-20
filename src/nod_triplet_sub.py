from pathlib import Path
import json
import numpy as np
import pandas as pd
import importlib.util


# Choose which ROIs to average
ROIS = ["V1", "V2", "V3"] 

# Paths
OUT_DIR = Path(r"T:\multimodal_brain_inspired\BI\triplet\csv_imagenet_per_subject")
SUBJECT_DIR = Path(r"C:\Users\BrainInspired\Documents\GitHub\NaturalObjectDataset-Processing\Nick_RDMs\outputs")

SUBJECTS = list(range(1, 31))

# Triplet mining hyperparams
TRIPLETS_PER_ANCHOR = 5
POS_FRAC = 0.03
NEG_TAIL_FRAC = 0.05
SEED = 42

# Split fractions
TRAIN_FRAC = 0.90
VAL_FRAC   = 0.05
TEST_FRAC  = 0.05


# HELPERS

def load_image_list(filenames_path: Path) -> list[str]:
    """
    Returns list of length N where index i corresponds to row/col i in the RDM.
    Keeps the first token per line (robust to extra columns).
    """
    lines = [ln.strip() for ln in filenames_path.read_text().splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if "\\n" in ln:
            ln = ln.split("\\n", 1)[0].strip()
        # keep first token if extra stuff exists
        tok = ln.split()[0]
        out.append(tok)
    return out


def infer_n_from_vec_len(L: int) -> int:
    # Solve L = N*(N-1)/2
    disc = 1 + 8 * L
    n = int((1 + int(np.sqrt(disc))) // 2)
    if n * (n - 1) // 2 != L:
        raise ValueError(f"Vector length {L} is not a valid N*(N-1)/2.")
    return n


def rdm_vector_to_square(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec).astype(np.float32)
    if vec.ndim != 1:
        raise ValueError("rdm_vector_to_square expects a 1D vector.")
    L = int(vec.shape[0])
    N = infer_n_from_vec_len(L)

    rdm = np.zeros((N, N), dtype=np.float32)
    iu = np.triu_indices(N, k=1)
    rdm[iu] = vec
    rdm[(iu[1], iu[0])] = rdm[iu]
    np.fill_diagonal(rdm, 0.0)
    return rdm


def load_rdm(path_base: Path) -> np.ndarray:
    """
    Loads an RDM saved as .npy (vector or square)
    """
    npy = path_base.with_suffix(".npy")
    if npy.exists():
        arr = np.load(npy)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return rdm_vector_to_square(arr)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr.astype(np.float32)
        raise ValueError(f"Unexpected shape in {npy}: {arr.shape}")
    
    raise FileNotFoundError(f"Missing RDM file: {npy} (or {py})")


def mine_triplets_from_rdm(
    rdm: np.ndarray,
    triplets_per_anchor: int,
    pos_frac: float,
    neg_tail_frac: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    rdm: NxN dissimilarity matrix (smaller = more similar)
    returns: [num_triplets, 3] of (anchor, pos, neg)
    """
    N = rdm.shape[0]
    if rdm.shape[1] != N:
        raise ValueError("RDM must be square.")
    if N < 3:
        raise ValueError("Need at least 3 images to form triplets.")

    pos_k = max(1, int(round(pos_frac * (N - 1))))
    neg_k = max(1, int(round(neg_tail_frac * (N - 1))))

    triplets = []

    for a in range(N):
        d = rdm[a].copy()
        d[a] = np.inf  # exclude self

        order = np.argsort(d)  # closest first

        pos_pool = order[:pos_k]
        neg_pool = order[-neg_k:]

        # remove anchor + prevent overlap
        pos_pool = pos_pool[pos_pool != a]
        neg_pool = neg_pool[(neg_pool != a) & (~np.isin(neg_pool, pos_pool))]

        if len(pos_pool) == 0 or len(neg_pool) == 0:
            continue

        for _ in range(triplets_per_anchor):
            p = int(rng.choice(pos_pool))
            n = int(rng.choice(neg_pool))
            triplets.append((a, p, n))

    if not triplets:
        raise RuntimeError("No triplets mined. Check RDM or parameters.")
    return np.array(triplets, dtype=np.int32)


def split_and_save_triplets_as_images(
    triplets: np.ndarray,
    idx_to_img: list[str],
    out_dir: Path,
    seed: int,
    meta_extra: dict
):
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    triplets = triplets[rng.permutation(len(triplets))]

    n_total = len(triplets)
    n_train = int(round(TRAIN_FRAC * n_total))
    n_val   = int(round(VAL_FRAC * n_total))

    train = triplets[:n_train]
    val   = triplets[n_train:n_train + n_val]
    test  = triplets[n_train + n_val:]

    def to_img_df(arr: np.ndarray) -> pd.DataFrame:
        a = [idx_to_img[i] for i in arr[:, 0]]
        p = [idx_to_img[i] for i in arr[:, 1]]
        n = [idx_to_img[i] for i in arr[:, 2]]
        return pd.DataFrame({"anchor_img": a, "positive_img": p, "negative_img": n})

    to_img_df(train).to_csv(out_dir / "train_triplets.csv", index=False)
    to_img_df(val).to_csv(out_dir / "val_triplets.csv", index=False)
    to_img_df(test).to_csv(out_dir / "test_triplets.csv", index=False)

    meta = {
        "n_total": n_total,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": seed,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        **meta_extra
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roi_tag = "avg_" + "-".join(ROIS)

    for sub in SUBJECTS:
        sub_dir = SUBJECT_DIR / f"sub-{sub}"
        filenames_path = sub_dir / "filenames.txt"
        if not filenames_path.exists():
            print(f"[SKIP] Missing filenames: {filenames_path}")
            continue

        idx_to_img = load_image_list(filenames_path)

        # Load and average RDMs across selected ROIs
        rdms = []
        for roi in ROIS:
            base = sub_dir / f"{roi}_rdm_vector"  # expects .npy
            try:
                rdm_roi = load_rdm(base)
                rdms.append(rdm_roi)
            except FileNotFoundError:
                print(f"[SKIP] Missing ROI file for sub-{sub}: {base}.npy/.py")
            except Exception as e:
                print(f"[SKIP] Failed loading {base}: {e}")

        if len(rdms) == 0:
            print(f"[SKIP] No ROI RDMs loaded for sub-{sub}.")
            continue

        # shape checks
        N = rdms[0].shape[0]
        if any(r.shape != (N, N) for r in rdms):
            print(f"[SKIP] Shape mismatch among ROI RDMs for sub-{sub}.")
            continue
        if len(idx_to_img) != N:
            print(f"[SKIP] filenames.txt length {len(idx_to_img)} != RDM N {N} for sub-{sub}.")
            continue

        rdm_avg = np.mean(np.stack(rdms, axis=0), axis=0).astype(np.float32)

        rng = np.random.default_rng(SEED)
        triplets = mine_triplets_from_rdm(
            rdm=rdm_avg,
            triplets_per_anchor=TRIPLETS_PER_ANCHOR,
            pos_frac=POS_FRAC,
            neg_tail_frac=NEG_TAIL_FRAC,
            rng=rng,
        )

        out_sub_dir = OUT_DIR / f"sub-{sub}" / roi_tag
        meta_extra = {
            "subject": f"sub-{sub}",
            "rois": ROIS,
            "n_rois_used": len(rdms),
            "pos_frac": POS_FRAC,
            "neg_tail_frac": NEG_TAIL_FRAC,
            "triplets_per_anchor": TRIPLETS_PER_ANCHOR,
            "N_images": N,
        }

        split_and_save_triplets_as_images(
            triplets=triplets,
            idx_to_img=idx_to_img,
            out_dir=out_sub_dir,
            seed=SEED,
            meta_extra=meta_extra,
        )

        print(f"[OK] sub-{sub}: saved {len(triplets)} triplets -> {out_sub_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

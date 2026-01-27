from pathlib import Path
import numpy as np

OUTPUTS_ROOT = Path("C:\\Users\\BrainInspired\\Documents\\GitHub\\NaturalObjectDataset-Processing\\Nick_RDMs\\outputs")
OUT_DIR      = Path("T:\\multimodal_brain_inspired\\BI\\triplet")

ROI_GROUPS = {
    "early": ["V1", "V2", "V3"],
    "mid":   ["V4", "V8", "LO1", "LO2", "LO3"],
    "late":  ["PIT", "VVC", "FFC", "VMV1", "VMV2", "VMV3"],
}
ROI_GROUPS["all"] = ROI_GROUPS["early"] + ROI_GROUPS["mid"] + ROI_GROUPS["late"]


def list_subject_dirs(outputs_root: Path):
    return sorted([p for p in outputs_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])


def read_filenames(subject_dir: Path):
    fpath = subject_dir / "filenames.txt"
    with open(fpath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    subj_dirs = list_subject_dirs(OUTPUTS_ROOT)
    if not subj_dirs:
        raise FileNotFoundError(f"No sub-XX folders found under: {OUTPUTS_ROOT}")

    # Canonical ordering (assumed consistent across subjects)
    filenames = read_filenames(subj_dirs[0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save filenames once (recommended so indices stay interpretable)
    with open(OUT_DIR / "filenames.txt", "w") as f:
        f.write("\n".join(filenames) + "\n")

    # Aggregate vectors directly (no square reconstruction, no z-score)
    for group_name, rois in ROI_GROUPS.items():
        acc = None
        count = 0

        for sd in subj_dirs:
            for roi in rois:
                vec_path = sd / f"{roi}_roi_rdm_vector.npy"
                if not vec_path.exists():
                    continue

                v = np.load(vec_path).astype(np.float32)
                if acc is None:
                    acc = np.zeros_like(v)
                elif v.shape != acc.shape:
                    raise ValueError(f"Vector length mismatch: {vec_path} has {v.shape}, expected {acc.shape}")

                acc += v
                count += 1

        if count == 0:
            raise RuntimeError(f"No vectors found for group '{group_name}'. Check ROI names/paths.")

        group_vec = acc / float(count)
        np.save(OUT_DIR / f"{group_name}_group_rdm_vector.npy", group_vec)
        print(f"[OK] {group_name}: averaged {count} vectors -> {OUT_DIR / (group_name + '_group_rdm_vector.npy')}")

    print(f"\nDone. Wrote group vectors + filenames to: {OUT_DIR}")


if __name__ == "__main__":
    main()

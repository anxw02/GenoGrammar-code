#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PREP = REPO / "pipeline" / "preprocessing" / "prepare_formal_workspace.py"
S02 = REPO / "pipeline" / "preprocessing" / "stage02_gene_order.py"
S03 = REPO / "pipeline" / "preprocessing" / "stage03_protein_families.sh"
S06A = REPO / "pipeline" / "preprocessing" / "extract_family_representatives.py"
S06B = REPO / "pipeline" / "esm" / "embed_family_representatives_esm2.py"
EXPECTED_NPY_SHA = "7e0d42cff60e4b8826e2ca189b82b1c63a926cbc2f9581aad31231d472b6b02f"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def run(cmd, env):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env)

def main():
    ap = argparse.ArgumentParser(description="Rebuild the formal GenoGramma family embeddings from the frozen raw-genome universe.")
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    work = Path(args.work_root).resolve()

    prodigal = shutil.which("prodigal")
    mmseqs = shutil.which("mmseqs")
    missing = [n for n,v in [("prodigal",prodigal),("mmseqs",mmseqs)] if not v]
    try:
        import esm  # noqa
    except Exception:
        missing.append("fair-esm")
    if missing and not args.plan_only:
        raise SystemExit("[FAIL] missing dependencies: " + ", ".join(missing) + "\nCreate the conda environment from environment.yml and activate it.")
    if missing and args.plan_only:
        print("[INFO] dependencies not currently active:", ", ".join(missing))

    env = os.environ.copy()
    env["GENOGRAMMA_WORK_ROOT"] = str(work)
    env["GENOGRAMMA_PRODIGAL"] = prodigal
    env["GENOGRAMMA_MMSEQS"] = mmseqs
    env["GENOGRAMMA_PYTHON"] = sys.executable
    env["TORCH_HOME"] = str(REPO / "model_assets" / "torch_cache")

    steps = [
        ("workspace", [sys.executable, PREP, "--work-root", work], work / "01_1_ani_robustness" / "06_FINAL_robust_ANI_blocks.csv"),
        ("stage02", [sys.executable, S02], work / "02_gene_order" / "12_stage02_summary.json"),
        ("stage03", ["bash", S03], work / "03_protein_families" / "13_STAGE03_summary.json"),
        ("stage06a", [sys.executable, S06A], work / "06_family_esm2" / "02_family_representative_index.csv"),
        ("stage06b", [sys.executable, S06B], work / "06_family_esm2" / "03_ESM2_t12_35M_embeddings.f16.mmap"),
    ]
    print("Formal raw-to-family plan:")
    for name, cmd, marker in steps:
        print(f"  {name:10s} -> {marker}")
    if args.plan_only:
        print("[PASS] plan only; no preprocessing executed")
        return

    for name, cmd, marker in steps:
        if args.resume and marker.exists() and marker.stat().st_size > 0:
            print(f"[CACHE] {name}: {marker}")
            continue
        run(cmd, env)

    mmap = work / "06_family_esm2" / "03_ESM2_t12_35M_embeddings.f16.mmap"
    npy = work / "06_family_esm2" / "00_family_embeddings_276517x480.npy"
    x = np.memmap(mmap, mode="r", dtype=np.float16, shape=(276517,480))
    if not np.isfinite(x).all():
        raise SystemExit("[FAIL] non-finite ESM embedding values")
    np.save(npy, np.asarray(x))
    got = sha256(npy)
    print("[INFO] rebuilt family embedding SHA256 =", got)
    if got != EXPECTED_NPY_SHA:
        raise SystemExit("[FAIL] rebuilt family embedding does not match the frozen formal asset")
    print("[FINAL PASS] raw-genome -> family ESM embeddings reproduced byte-identically")
    print("family_embeddings =", npy)

if __name__ == "__main__":
    main()

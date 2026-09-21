#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, shutil
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "model_assets" / "formal_upstream"
RAW = REPO / "raw_data" / "genomes"

def remap_csv(src: Path, dst: Path, work: Path) -> int:
    df = pd.read_csv(src)
    raw_files = list(RAW.rglob("*.fa.gz"))
    by_name = {}
    for p in raw_files:
        by_name.setdefault(p.name, []).append(p)
    changed = 0
    for col in df.columns:
        if "path" not in col.lower():
            continue
        vals = []
        for v in df[col].tolist():
            if not isinstance(v, str) or not v:
                vals.append(v); continue
            nv = v
            marker = "/genomes/"
            if marker in v:
                rel = v.split(marker, 1)[1]
                cand = work / "genomes" / rel
                if cand.exists():
                    nv = str(cand)
                else:
                    hits = by_name.get(Path(v).name, [])
                    if len(hits) == 1:
                        nv = str(hits[0])
            elif Path(v).is_absolute() and not Path(v).exists():
                hits = by_name.get(Path(v).name, [])
                if len(hits) == 1:
                    nv = str(hits[0])
            changed += int(nv != v)
            vals.append(nv)
        df[col] = vals
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    args = ap.parse_args()
    work = Path(args.work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)

    gdst = work / "genomes"
    if not gdst.exists():
        gdst.symlink_to(RAW, target_is_directory=True)

    robust_src = ASSETS / "06_FINAL_robust_ANI_blocks.csv"
    primary_src = ASSETS / "07_PRIMARY_FINAL_robust_ANI_blocks.csv"
    qc_src = ASSETS / "10_STAGE02_1_QC_DECISIONS.csv"
    for p in (robust_src, primary_src, qc_src):
        if not p.is_file():
            raise SystemExit(f"[FAIL] missing frozen asset: {p}")

    robust_dst = work / "01_1_ani_robustness" / robust_src.name
    primary_dst = work / "01_1_ani_robustness" / primary_src.name
    qc_dst = work / "02_1_gene_call_qc" / qc_src.name

    c1 = remap_csv(robust_src, robust_dst, work)
    c2 = remap_csv(primary_src, primary_dst, work)
    qc_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(qc_src, qc_dst)

    robust = pd.read_csv(robust_dst)
    if len(robust) != 994:
        raise SystemExit(f"[FAIL] formal robust universe must be 994 rows, got {len(robust)}")
    if "strain_entity_id" in robust and robust["strain_entity_id"].duplicated().any():
        raise SystemExit("[FAIL] duplicate strain_entity_id in robust universe")

    path_cols = [c for c in robust.columns if "path" in c.lower()]
    missing = []
    for c in path_cols:
        for v in robust[c].dropna().astype(str):
            if v and Path(v).is_absolute() and not Path(v).exists():
                missing.append((c, v))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        raise SystemExit(f"[FAIL] unresolved genome paths, preview={missing[:10]}")

    print("[PASS] formal workspace prepared")
    print("work_root =", work)
    print("formal_genomes =", len(robust))
    print("raw_archive_fasta_gz =", len(list(RAW.rglob('*.fa.gz'))))
    print("remapped_paths =", c1 + c2)

if __name__ == "__main__":
    main()

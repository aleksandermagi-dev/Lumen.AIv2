
#!/usr/bin/env python3
"""
Great Attractor Bulk-Flow • Local Analysis Tool
------------------------------------------------
Usage:
  python run_ga_flow.py --csv path/to/your_cone.csv --h0 70 --out out_dir --boot 1000

Input CSV must have columns (case-insensitive):
  ra_deg, dec_deg, cz_kms, dist_mpc, dist_err_mpc

This script:
  1) Computes peculiar velocities: v_pec = cz_kms - H0*dist_mpc
  2) Fits a 3D bulk-flow vector via weighted least squares
  3) Bootstraps the fit to estimate a 1σ uncertainty circle on sky
  4) Saves:
     - apex_map.png (mobile-friendly)
     - apex_bootstrap.png (cloud of bootstrap apices)
     - results.txt (numbers)
     - GA_OnePager.pdf (one-page PDF summary)
"""

import argparse, os, math, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def radec_to_unit(ra_deg, dec_deg):
    ra = math.radians(ra_deg); dec = math.radians(dec_deg)
    return np.array([math.cos(dec)*math.cos(ra), math.cos(dec)*math.sin(ra), math.sin(dec)])

def unit_to_radec(v):
    x,y,z = v
    n = math.sqrt(x*x+y*y+z*z)
    if n == 0: return (float('nan'), float('nan'))
    x/=n; y/=n; z/=n
    dec = math.degrees(math.asin(z))
    ra = (math.degrees(math.atan2(y,x)) + 360.0) % 360.0
    return ra, dec

def angsep_radec(ra1, dec1, ra2, dec2):
    a = math.radians(90-dec1); b = math.radians(90-dec2)
    C = math.radians((ra2 - ra1) % 360.0)
    cosc = math.cos(a)*math.cos(b) + math.sin(a)*math.sin(b)*math.cos(C)
    cosc = max(-1.0, min(1.0, cosc))
    return math.degrees(math.acos(cosc))

def fit_bulk(df, H0):
    c_kms = 299792.458
    # required cols (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    for need in ["ra_deg","dec_deg","cz_kms","dist_mpc","dist_err_mpc"]:
        if need not in cols:
            raise ValueError(f"Missing column: {need}")
    ra = df[cols["ra_deg"]].values.astype(float)
    dec = df[cols["dec_deg"]].values.astype(float)
    cz  = df[cols["cz_kms"]].values.astype(float)
    dmpc= df[cols["dist_mpc"]].values.astype(float)
    derr= df[cols["dist_err_mpc"]].values.astype(float)

    vobs = cz
    vpec = vobs - H0*dmpc
    sigma = np.sqrt(50.0**2 + (H0*derr)**2)  # redshift floor + distance error contribution
    wts = 1.0/(sigma**2)

    A = np.array([radec_to_unit(r, d) for r,d in zip(ra, dec)])
    W = np.diag(wts)
    ATA = A.T @ W @ A
    ATv = A.T @ W @ vpec
    B = np.linalg.solve(ATA, ATv)
    amp = float(np.linalg.norm(B))
    apex_ra, apex_dec = unit_to_radec(B)
    return {"B":B, "amp":amp, "apex_ra":apex_ra, "apex_dec":apex_dec, "vpec":vpec, "sigma":sigma}

def bootstrap_apex(df, H0, nboot=500, seed=42):
    rng = np.random.default_rng(seed)
    apices = []
    n = len(df)
    for _ in range(nboot):
        idx = rng.integers(0, n, size=n)
        dfi = df.iloc[idx].reset_index(drop=True)
        try:
            r = fit_bulk(dfi, H0)
            apices.append([r["apex_ra"], r["apex_dec"]])
        except np.linalg.LinAlgError:
            continue
    return np.array(apices)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with columns ra_deg, dec_deg, cz_kms, dist_mpc, dist_err_mpc")
    ap.add_argument("--h0", type=float, default=70.0, help="Hubble constant (km/s/Mpc)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--boot", type=int, default=500, help="Bootstrap iterations")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    res = fit_bulk(df, args.h0)
    # Great Attractor/cluster markers
    GA_RA, GA_DEC = 200.0, -49.0
    norma_ra, norma_dec = 243.592, -60.872
    cent_ra, cent_dec   = 192.216, -41.306
    hydra_ra, hydra_dec = 159.174, -27.524

    # Distances to apex
    ang_cent = angsep_radec(res["apex_ra"], res["apex_dec"], cent_ra, cent_dec)
    ang_norma= angsep_radec(res["apex_ra"], res["apex_dec"], norma_ra, norma_dec)
    ang_hydra= angsep_radec(res["apex_ra"], res["apex_dec"], hydra_ra, hydra_dec)
    ang_ga   = angsep_radec(res["apex_ra"], res["apex_dec"], GA_RA, GA_DEC)

    # Bootstrap for uncertainty
    apx = bootstrap_apex(df, args.h0, nboot=args.boot)
    # Compute 1-sigma equivalent circle radius (angular) from bootstrap covariance
    if len(apx) > 20:
        # compute angular stddev radius about the mean apex (rough)
        mean_ra = np.mean(apx[:,0]); mean_dec = np.mean(apx[:,1])
        angs = [angsep_radec(a[0], a[1], mean_ra, mean_dec) for a in apx]
        sigma_deg = np.percentile(angs, 68)  # 68th percentile
    else:
        sigma_deg = 10.0  # fallback

    # Save results text
    with open(os.path.join(args.out, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Bulk-flow amplitude: {res['amp']:.1f} km/s\n")
        f.write(f"Apex (RA, Dec): ({res['apex_ra']:.2f}°, {res['apex_dec']:.2f}°)\n")
        f.write(f"Angle to Centaurus: {ang_cent:.2f}°\n")
        f.write(f"Angle to Norma: {ang_norma:.2f}°\n")
        f.write(f"Angle to Hydra: {ang_hydra:.2f}°\n")
        f.write(f"Angle to GA dir (~200°,-49°): {ang_ga:.2f}°\n")
        f.write(f"Bootstrap 1σ radius (approx): {sigma_deg:.2f}°\n")

    # Mobile-friendly map
    plt.figure(figsize=(6,8))
    plt.scatter(norma_ra, norma_dec, s=160, label="Norma (A3627)")
    plt.scatter(cent_ra, cent_dec, s=160, label="Centaurus (A3526)")
    plt.scatter(hydra_ra, hydra_dec, s=160, label="Hydra (A1060)")
    plt.scatter(res["apex_ra"], res["apex_dec"], marker="*", s=300, label="Bulk-flow apex")
    ax = plt.gca()
    ax.invert_xaxis()
    circ = plt.Circle((res["apex_ra"], res["apex_dec"]), sigma_deg, fill=False, linestyle="--")
    ax.add_patch(circ)
    plt.title("Bulk-flow Apex vs Great Attractor Clusters")
    plt.xlabel("RA (deg)"); plt.ylabel("Dec (deg)")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True, alpha=0.3)
    map_path = os.path.join(args.out, "apex_map.png")
    plt.tight_layout(); plt.savefig(map_path, dpi=220, bbox_inches="tight"); plt.close()

    # Bootstrap scatter (optional visualization)
    if len(apx) > 0:
        plt.figure(figsize=(6,8))
        plt.scatter(apx[:,0], apx[:,1], s=10, alpha=0.5)
        plt.scatter(res["apex_ra"], res["apex_dec"], marker="*", s=250)
        plt.gca().invert_xaxis()
        plt.title("Bootstrap Apex Cloud")
        plt.xlabel("RA (deg)"); plt.ylabel("Dec (deg)")
        plt.grid(True, alpha=0.3)
        boot_path = os.path.join(args.out, "apex_bootstrap.png")
        plt.tight_layout(); plt.savefig(boot_path, dpi=220, bbox_inches="tight"); plt.close()

    # One-page PDF using reportlab if available; if not, skip gracefully
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        pdf_path = os.path.join(args.out, "GA_OnePager.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("<b>Great Attractor – Bulk Flow Result</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Image(map_path, width=350, height=460))
        story.append(Spacer(1, 12))
        cap = (f"Bulk-flow apex = ({res['apex_ra']:.2f}°, {res['apex_dec']:.2f}°), "
               f"1σ ≈ {sigma_deg:.1f}°. "
               f"Angles: Centaurus {ang_cent:.1f}°, Norma {ang_norma:.1f}°, Hydra {ang_hydra:.1f}°, GA {ang_ga:.1f}°.")
        story.append(Paragraph(cap, styles["BodyText"]))
        doc.build(story)
    except Exception as e:
        # Not fatal
        with open(os.path.join(args.out, "PDF_ERROR.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

    print("Done.\nOutputs in:", args.out)

if __name__ == "__main__":
    main()

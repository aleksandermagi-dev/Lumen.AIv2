# --- HST/COS Si IV finder (ASCII-safe) ---
# Loads an x1d FITS file, flattens all segments,
# prints min dip, wavelength, and velocity for Si IV,
# and plots a zoomed window with velocity markers.

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# --------- user input ---------
FILENAME = r"C:.fits"  # <-- Data Folder Location
# ------------------------------

C_KMS = 299792.458

def smooth(y, win=7):
    win = max(1, int(win))
    if y.size < 2 or win <= 1:
        return y.astype(float)
    k = np.ones(win) / win
    return np.convolve(y.astype(float), k, mode="same")

def load_spectrum(path):
    with fits.open(path) as hdul:
        tab = hdul[1].data
        f_all = np.array(tab["FLUX"], dtype=float).ravel()
        w_all = np.array(tab["WAVELENGTH"], dtype=float).ravel()
    m = np.isfinite(w_all) & np.isfinite(f_all)
    w = w_all[m]; f = f_all[m]
    print(f"Coverage: {w.min():.1f}-{w.max():.1f} A (N={w.size})")
    return w, f

def zoom(w, f, rest, width=3.0, title=None, smooth_win=7):
    half = width / 2.0
    sel = (w > rest - half) & (w < rest + half)
    if sel.sum() == 0:
        print(f"No data around {rest:.3f} A")
        return None

    wseg = w[sel]
    fraw = f[sel]
    fseg = smooth(fraw, smooth_win) if fraw.size > 2 else fraw

# --- robust continuum from outer quartiles (bright side) ---
    k = max(1, int(0.25 * wseg.size))
    edges = np.concatenate([fseg[:k], fseg[-k:]])
    cont = float(np.nanpercentile(edges, 90))
    if not np.isfinite(cont) or cont <= 0.0:
        cont = float(np.nanpercentile(fseg, 90))
        if not np.isfinite(cont) or cont <= 0.0:
            cont = 1.0 # last resort to avoid divide-by-zero

# --- find minimum and compute depth (clamped 0..1) ---
    i = int(np.nanargmin(fseg))
    fmin = float(fseg[i])
    lam_min = float(wseg[i])

    depth = 1.0 - (fmin / cont)
    if depth < 0.0: depth = 0.0
    if depth > 1.0: depth = 1.0

    v_kms = C_KMS * (lam_min / rest - 1.0)
    name = title if title else f"rest {rest:.1f} A"
    print(f"{name}: min dip = {depth*100:.2f}% at lam = {lam_min:.3f} A v = {v_kms:+.0f} km/s")
    return lam_min, depth, v_kms

# Key UV rest wavelengths (A)
LINES = [
    (1215.670, "LyA"),
    (1206.500, "Si III"),
    (1393.760, "Si IV 1393"),
    (1402.773, "Si IV 1402"),
    (1548.204, "C IV 1548"),
    (1550.781, "C IV 1550"),
    (1238.821, "N V 1238"),
    (1242.804, "N V 1242"),
    (1250.584, "S II 1250"),
    (1253.811, "S II 1253"),
    (1259.519, "S II 1259"),
    (1260.422, "Si II 1260"),
    (1302.168, "O I 1302"),
    (1304.858, "O I 1304"),
    (1306.029, "O I 1306"),
    (1334.532, "C II 1334"),
    (1335.708, "C II 1335"),
    (1608.451, "Fe II 1608"),
    (1670.788, "Al II 1670"),
    (1031.926, "O VI 1031"),
    (1037.617, "O VI 1037"),
]

def plot_si_iv_window(w, f):
    lo, hi = 1391.0, 1404.0
    m = (w >= lo) & (w <= hi)
    if m.sum() == 0:
        print("No data in 1391-1404 A")
        return

    ws = w[m]
    fr = f[m]
    fs = smooth(fr, 9)

    plt.figure(figsize=(10,4))
    plt.plot(ws, fr, lw=0.5, label="raw")
    plt.plot(ws, fs, lw=1.2, label="smoothed")
    plt.xlim(lo, hi)
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux")
    plt.title("Si IV window")

    r1393, r1402 = 1393.760, 1402.773

    def vline_at(rest, v_kms, text):
        lam = rest * (1.0 + v_kms / C_KMS)
        plt.axvline(lam, ls="--", alpha=0.7)
        top = plt.ylim()[1]
        plt.text(lam, top*0.92, text, rotation=90, ha="right", va="top", fontsize=8)

    for v, tag in [(0, "MW 0"), (-219, "cand -219"), (-300, "M31 -300")]:
        vline_at(r1393, v, f"1393 {tag}")
        vline_at(r1402, v, f"1402 {tag}")

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1) Load spectrum
    w0, f0 = load_spectrum(FILENAME)

# 2) Quick overview plot (close it to continue)
    plt.figure(figsize=(11,4))
    plt.plot(w0, f0, lw=0.4, label="raw")
    plt.plot(w0, smooth(f0, 11), lw=1.0, label="smoothed")
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux")
    plt.title("HST Spectrum")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) OPTIONAL: broad scan (commented out to keep output short)
    # for rest, name in LINES:
    # if "Si IV" in name:
    # width = 12.0
    # elif "C IV" in name:
    # width = 10.0
    # else:
    # width = 8.0
    # zoom(w0, f0, rest, width=width, title=name)

    # 4) Targeted Si IV checks at three velocities

    print("\nChecking v = -219 km/s (candidate):")
    # expected approx centers: 1392.742 A and 1401.748 A
    zoom(w0, f0, 1393.760, width=0.8, title="Si IV 1393 @ -219", smooth_win=5)
    zoom(w0, f0, 1402.773, width=0.8, title="Si IV 1402 @ -219", smooth_win=5)

    print("\nChecking M31 systemic (v = -300 km/s):")
    # expected approx centers: 1392.365 A and 1401.369 A
    zoom(w0, f0, 1393.760, width=1.0, title="Si IV 1393 @ -300", smooth_win=5)
    zoom(w0, f0, 1402.773, width=1.0, title="Si IV 1402 @ -300", smooth_win=5)

    print("\nChecking Milky Way foreground (v ~ 0 km/s):")
    zoom(w0, f0, 1393.760, width=1.0, title="Si IV 1393 @ 0", smooth_win=5)
    zoom(w0, f0, 1402.773, width=1.0, title="Si IV 1402 @ 0", smooth_win=5)

    # 5) Zoomed plot with velocity markers
    plot_si_iv_window(w0, f0)

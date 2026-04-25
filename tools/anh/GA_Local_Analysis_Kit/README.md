
Great Attractor – Local Bulk-Flow Analysis Kit
==============================================

What this does
--------------
Given a small cone of galaxies near the Great Attractor (RA 150–240°, Dec -70° to -20°),
this kit computes the bulk-flow apex and a 1σ uncertainty, then shows where it lands
relative to Centaurus (A3526), Norma (A3627), Hydra (A1060), and the GA direction.

What you need
-------------
- Python 3.9+
- pip install numpy pandas matplotlib (optional: reportlab)

How to get the CSV (Cosmicflows-4 via EDD)
------------------------------------------
1) Go to the Extragalactic Distance Database (EDD) Cosmicflows-4 interface.
2) Query a cone around the GA region (e.g., RA 150–240°, Dec -70° to -20°).
3) Include columns: RA (deg), Dec (deg), cz (km/s) **in CMB frame if available**, Distance (Mpc), Distance error (Mpc).
4) Export to CSV and rename columns EXACTLY to:
     ra_deg, dec_deg, cz_kms, dist_mpc, dist_err_mpc
5) Save your file, e.g., cf4_ga_cone.csv

How to run
----------
python run_ga_flow.py --csv cf4_ga_cone.csv --h0 70 --out results --boot 1000

What you get
------------
- results/apex_map.png        : phone-friendly map with apex + 1σ circle and cluster markers
- results/apex_bootstrap.png  : bootstrap cloud of apex directions
- results/results.txt         : numeric summary (angles to Centaurus/Norma/Hydra/GA)
- results/GA_OnePager.pdf     : one-page PDF summary (if reportlab is installed)

Pass/Fail intuition
-------------------
- If Centaurus (A3526) lies inside or on the 1σ circle AND
  the angle to GA/centaurus is small (<~10°),
  and this remains true under bootstrap (and later jackknife),
  then you've got **independent confirmation** of the GA node alignment.

Notes
-----
- You can adjust --h0 (e.g., 67–73) to test sensitivity.
- For a quick dry run, edit templates/ga_cone_template.csv with a couple rows and run it.

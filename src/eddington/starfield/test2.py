import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from astroquery.vizier import Vizier
import numpy as np
import matplotlib.pyplot as plt

# Choose the DR3 Gaia catalog
GAIA_DR3 = "I/355/gaiadr3"

# VizieR names -> ESA archive names
COLS = {
    "Source":  "source_id",
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "Gmag":    "phot_g_mean_mag",
    "BP-RP":   "bp_rp",
    "Plx":     "parallax",
    "pmRA":    "pmra",
    "pmDE":    "pmdec",
    "RUWE":    "ruwe",
}

# Coordinate of interest: where the Sun will be during the 2027 eclipse
coord = SkyCoord(ra='08h48m', dec=17.9, unit=(u.hourangle, u.degree), frame='icrs')

v = Vizier(
    columns=list(COLS),
    column_filters={"Gmag": "<18"},  # optional: drop faint stars server-side
    row_limit=-1,                    # default is 50!
)
v.TIMEOUT = 300

result = v.query_region(coord, width=2.0 * u.deg, height=2.0 * u.deg, catalog=GAIA_DR3)
r = result[0]  # query_region returns a TableList, one table per catalog

r.rename_columns(list(COLS), list(COLS.values()))
print(f"Length of results {len(r)}")
# ---- tangent-plane projection centred on the field (1 "pixel" = 1 arcsec) ----
wcs = WCS(naxis=2)
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
wcs.wcs.crval = [coord.ra.deg, coord.dec.deg]
wcs.wcs.crpix = [1, 1]                      # centre lands on pixel (0, 0) with origin=0
wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]       # negative: East on the left, as on the sky

ra  = np.asarray(r["ra"], dtype=float)
dec = np.asarray(r["dec"], dtype=float)
G   = np.ma.filled(r["phot_g_mean_mag"], np.nan).astype(float)
ok  = np.isfinite(G)
ra, dec, G = ra[ok], dec[ok], G[ok]
x, y = wcs.all_world2pix(ra, dec, 0)

# ---- brightness -> marker area: area ∝ sqrt(flux) squeezes ~10 mag into a readable range ----
G_REF = np.min(G)
def mag_to_size(m):
    return 2 + 160 * np.sqrt(10 ** (-0.4 * (m - G_REF)))

order = np.argsort(-G)                       # faint first, bright on top

# ---- figure ----
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection=wcs)
ax.scatter(x[order], y[order], s=mag_to_size(G[order]),
           c="#1f2328", edgecolors="white", linewidths=0.6, zorder=3)

# ---- axes: proper RA (hh:mm:ss) and Dec (dd:mm) ticks ----
lon, lat = ax.coords[0], ax.coords[1]
lon.set_major_formatter("hh:mm")
lat.set_major_formatter("dd:mm")
lon.set_ticks(spacing=30 * u.arcmin)         # 30' = 2 min of RA at this scale
lat.set_ticks(spacing=30 * u.arcmin)
lon.display_minor_ticks(True); lon.set_minor_frequency(4)
lat.display_minor_ticks(True); lat.set_minor_frequency(3)
lon.set_axislabel("Right Ascension (ICRS)")
lat.set_axislabel("Declination (ICRS)")

# ---- major grid ----
ax.coords.grid(color="#8c959f", alpha=0.7, linestyle="-", linewidth=0.6)

# ---- minor grid: WCSAxes only grids major ticks, so overlay a denser, fainter grid ----
minor = ax.get_coords_overlay("icrs")
minor[0].set_ticks(spacing=7.5 * u.arcmin)
minor[1].set_ticks(spacing=10 * u.arcmin)
minor.grid(color="#8c959f", alpha=0.25, linestyle=":", linewidth=0.5)
for c in (minor[0], minor[1]):
    c.set_ticks_visible(False)
    c.set_ticklabel_visible(False)
    c.set_axislabel("")

# ---- optional but useful: the Sun and 2/5/10 R☉ rings ----
R_SUN = 0.2627 * u.deg                       # angular radius on 2 Aug 2027
for n in (1, 2, 5, 10):
    ax.add_patch(SphericalCircle((coord.ra, coord.dec), n * R_SUN,
                 transform=ax.get_transform("icrs"),
                 edgecolor="#bc4c00", facecolor="none",
                 linewidth=1.5 if n == 1 else 0.8,
                 linestyle="-" if n == 1 else "--", zorder=2))

# ---- field limits + true aspect ----
HALF = 1.6 * 3600                            # half-width in arcsec
ax.set_xlim(-HALF, HALF)
ax.set_ylim(-HALF, HALF)
ax.set_aspect("equal")

# ---- magnitude key ----
for m in (4, 6, 8, 10):
    ax.scatter([], [], s=mag_to_size(m), c="#1f2328", edgecolors="white", label=f"G = {m}")
ax.legend(title="Gaia G mag", loc="upper right", framealpha=0.9, labelspacing=1.2)

ax.set_title("Star field around the Sun, 2 Aug 2027")
plt.show()
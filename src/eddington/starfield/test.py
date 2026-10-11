import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"    # Select Data Release 3, default
Gaia.ROW_LIMIT = -1                             # fetch all unlimited
INTEREST_COLS = ["source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe"]

coord = SkyCoord(ra='08h48m', dec=17.9, unit=(u.hourangle, u.degree), frame='icrs')
width = u.Quantity(3.0, u.deg)
height = u.Quantity(3.0, u.deg)

r = Gaia.query_object_async(coordinate=coord, width=width, height=height, columns=INTEREST_COLS, verbose=True)

print(r)

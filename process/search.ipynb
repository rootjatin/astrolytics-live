
	
cells	
0	
cell_type	"markdown"
id	"b416bfd9"
metadata	{}
source	
0	"# 00 — Search and download astronomy data\n"
1	"\n"
2	"Use this notebook to search MAST for a target, inspect matching observations, and download one or more FITS products.\n"
3	"\n"
4	"This follows the same pattern as your Mars/JWST notebook: query observations, filter by instrument, rank products, and download locally. \ue200filecite\ue202turn0file0\ue201\n"
1	
cell_type	"code"
execution_count	null
id	"36f26832"
metadata	{}
outputs	[]
source	
0	"from pathlib import Path\n"
1	"import numpy as np\n"
2	"from astropy.table import Table\n"
3	"from astro_utils import get_obs_table, pick_products_for_obs, download_one\n"
2	
cell_type	"code"
execution_count	null
id	"3b6d0ed9"
metadata	{}
outputs	[]
source	
0	'TARGET_NAME = "Mars"\n'
1	'INSTRUMENT = "NIRCAM"      # try "MIRI", "NIRSPEC", "WFC3", etc.\n'
2	'RADIUS = "0.3 deg"\n'
3	"PROGRAM_ID = 1415          # set to None to search more broadly\n"
4	'COLLECTION = "JWST"        # MAST obs_collection\n'
5	'DOWNLOAD_DIR = Path("data")\n'
6	"\n"
7	"obs = get_obs_table(\n"
8	"    target=TARGET_NAME,\n"
9	"    instrument=INSTRUMENT,\n"
10	"    radius=RADIUS,\n"
11	"    program_id=PROGRAM_ID,\n"
12	"    collection=COLLECTION,\n"
13	")\n"
14	'print(f"Found {len(obs)} matching observations")\n'
15	"obs[:10]\n"
3	
cell_type	"code"
execution_count	null
id	"1e08e73d"
metadata	{}
outputs	[]
source	
0	"# Show useful columns if present\n"
1	'wanted = [c for c in ["obsid", "target_name", "instrument_name", "t_exptime", "filters", "obs_collection"] if c in obs.colnames]\n'
2	"obs[wanted][:10]\n"
4	
cell_type	"code"
execution_count	null
id	"55fce183"
metadata	{}
outputs	[]
source	
0	"# Choose one observation row and inspect products\n"
1	"row_index = 0\n"
2	"row = obs[row_index]\n"
3	"\n"
4	"prods = pick_products_for_obs(row)\n"
5	'print(f"Found {0 if prods is None else len(prods)} candidate products")\n'
6	"prods[:10] if prods is not None else None\n"
5	
cell_type	"code"
execution_count	null
id	"ada36b98"
metadata	{}
outputs	[]
source	
0	"# Download a few files\n"
1	"max_to_download = 2\n"
2	"downloaded = []\n"
3	"\n"
4	"if prods is not None:\n"
5	"    for pr in prods[:max_to_download]:\n"
6	"        path = download_one(pr, DOWNLOAD_DIR)\n"
7	"        downloaded.append(path)\n"
8	'        print("Downloaded:", path)\n'
9	"\n"
10	"downloaded\n"
6	
cell_type	"markdown"
id	"105edd29"
metadata	{}
source	
0	"### Adapt this notebook\n"
1	"- Change `TARGET_NAME` to your source\n"
2	"- Set `PROGRAM_ID=None` for a wider search\n"
3	"- Replace `INSTRUMENT` with the detector or instrument you need\n"
4	"- Increase `max_to_download` if you want multiple products\n"
metadata	
kernelspec	
display_name	"Python 3"
language	"python"
name	"python3"
nbformat	4
nbformat_minor	5

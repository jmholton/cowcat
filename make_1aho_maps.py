"""Convert gt48.mtz to 2fofc/fofc/fc CCP4 maps."""
import gemmi, sys
from pathlib import Path

mtz_path = Path('1aho/gt48.mtz')
outdir = Path('1aho_maps')
outdir.mkdir(exist_ok=True)

mtz = gemmi.read_mtz_file(str(mtz_path))

for f_col, phi_col, name in [
    ('FWT',    'PHWT',    '2fofc.map'),
    ('DELFWT', 'PHDELWT', 'fofc.map'),
    ('FC_ALL', 'PHIC_ALL','fc.map'),
]:
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=3.0)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    out = outdir / name
    ccp4.write_ccp4_map(str(out))
    print(f'Written {out}  shape={grid.point_count}')

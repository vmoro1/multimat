from mp_api.client import MPRester
from emmet.core.summary import HasProps
import h5py
import numpy as np

h5_file = "mp_test.h5"
MP_API_KEY="<your_api_key>"

# Electronic structure
for props in ['electronic_structure']:
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.summary.search(has_props = [getattr(HasProps,props)], fields=["structure","material_id", "density", "efermi", "band_gap", "cbm", "vbm"])
        for doc0 in docs:
            stru = str(doc0.structure)
            mpid = doc0.material_id
            density = doc0.density
            efermi = doc0.efermi
            band_gap = doc0.band_gap
            cbm = doc0.cbm
            vbm = doc0.vbm
            with h5py.File(h5_file, "a") as f:
                f.create_dataset(str(mpid) + "/structure", data=stru)
                f.create_dataset(str(mpid) + "/density", dtype='f', data=density)
                f.create_dataset(str(mpid) + "/efermi", dtype='f', data=efermi)
                f.create_dataset(str(mpid) + "/band_gap", dtype='f', data=band_gap)
                f.create_dataset(str(mpid) + "/cbm", dtype='f', data=cbm)
                f.create_dataset(str(mpid) + "/vbm", dtype='f', data=vbm)

    print(props, len(docs))

# Dielectric
for props in ['dielectric']:
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.summary.search(has_props = [getattr(HasProps,props)], fields=["material_id"])
        diel_mpids = [doc.material_id for doc in docs]
    print(props, len(diel_mpids))

for mpid in diel_mpids:
    with MPRester(MP_API_KEY) as mpr:
        diel = mpr.dielectric.get_data_by_id(mpid)
        total_diel = np.array(vars(diel)['total'])
        ionic_diel = np.array(vars(diel)['ionic'])
        elec_diel = np.array(vars(diel)['electronic'])
        with h5py.File(h5_file, "a") as f:
            f.create_dataset(str(mpid) + "/dielectric/total", dtype = 'f', data=total_diel)
            f.create_dataset(str(mpid) + "/dielectric/ionic", dtype = 'f', data=ionic_diel)
            f.create_dataset(str(mpid) + "/dielectric/electronic", dtype = 'f', data=elec_diel)


# DOS
for props in ['dos']:
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.summary.search(has_props = [getattr(HasProps,props)], fields=["material_id"])
        dos_mpids = [doc.material_id for doc in docs]
    print(props, len(dos_mpids))

for mpid in dos_mpids:
    with MPRester(MP_API_KEY) as mpr:
        dos0 = mpr.get_dos_by_material_id(mpid)
        energy = vars(dos0)['energies'] # array
        dos_up = None
        dos_down = None
        for k,v in vars(dos0)['densities'].items():
            if str(k) == '1':
                dos_up = v # array
            elif str(k) == '-1':
                dos_down = v
        with h5py.File(h5_file, "a") as f:
            f.create_dataset(str(mpid) + "/dos/energies", dtype='f', data=energy)
            if dos_up is not None:
                f.create_dataset(str(mpid) + "/dos/dos_up", dtype='f', data=dos_up)
            if dos_down is not None:
                f.create_dataset(str(mpid) + "/dos/dos_down", dtype='f', data=dos_down)

# charge density
for props in ['charge_density']:
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.summary.search(has_props = [getattr(HasProps,props)], fields=["material_id"])
        chgden_mpids = [doc.material_id for doc in docs]
    print(props, len(chgden_mpids))

for mpid in chgden_mpids:
    with MPRester(MP_API_KEY) as mpr:
        chgcar = mpr.get_charge_density_from_material_id(mpid)
        chg_den_total = vars(chgcar)['data']['total'] # array
        chg_den_diff = vars(chgcar)['data']['diff']   # array
        chg_den_xpoints = vars(chgcar)['xpoints'] # array
        chg_den_ypoints = vars(chgcar)['ypoints'] # array
        chg_den_zpoints = vars(chgcar)['zpoints'] # array
        with h5py.File(h5_file, "a") as f:
            f.create_dataset(str(mpid) + "/charge_density/total", dtype='f', data=chg_den_total)
            f.create_dataset(str(mpid) + "/charge_density/diff", dtype='f', data=chg_den_diff)
            f.create_dataset(str(mpid) + "/charge_density/xpoints", dtype='f', data=chg_den_xpoints)
            f.create_dataset(str(mpid) + "/charge_density/ypoints", dtype='f', data=chg_den_ypoints)
            f.create_dataset(str(mpid) + "/charge_density/zpoints", dtype='f', data=chg_den_zpoints)


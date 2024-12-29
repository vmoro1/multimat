# generate dictionary type .pt files for dos and bandgap. Dictionaries is on form {'mp-id': dos} and {'mp-id': bandgap} and are 
# saved as .pt files

import h5py
import numpy as np
import torch


def generate_dos_bandgap(path_to_h5='./MP_dos_bandgap_efermi_v2.h5', shift_by_efermi=True):

    # load h5 file
    f = h5py.File(path_to_h5,'r')

    # extract data from h5 file
    material_ids = [material_id for material_id in f.keys()]                        # material ids
    dos = [f[material_id + '/dos/dos_up'][()] for material_id in f.keys()]          # density of states
    energies = [f[material_id + '/dos/energies'][()] for material_id in f.keys()]   # energies
    efermi = [f[material_id + '/efermi'][()] for material_id in f.keys()]           # fermi energy

    # shift energies by fermi energy
    if shift_by_efermi:
        for i in range(len(energies)):
            energies[i] = energies[i] - efermi[i]

    # replace nan values with the average of it's closest neighbors that aren't nan themselves
    energies = replace_nans(energies)
    dos = replace_nans(dos)

    # down/upsample energies and dos
    # we know that the only possible lengths are 301, 601 and 2001
    desired_dim = 601
    dos_dict = {}
    for i in range(len(energies)):
        # determine if downsampling upsampling or neither is required
        if len(energies[i]) == desired_dim:
            # convert to torch tensors and unsqueeze
            energies_torch = torch.tensor(energies[i]).unsqueeze(0)
            dos_torch = torch.tensor(dos[i]).unsqueeze(0)
            dos_cat = torch.cat((energies_torch, dos_torch), dim=0).float()
            dos_dict[material_ids[i]] = dos_cat
        elif len(energies[i]) > desired_dim:
            # downsample
            energy_downsampled = energies[i][::3]
            energy_downsampled = [item for i, item in enumerate(energy_downsampled) if (i+1) % 10 != 0]
            dos_downsampled = dos[i][::3]
            dos_downsampled = [item for i, item in enumerate(dos_downsampled) if (i+1) % 10 != 0]
            
            # convert to torch tensors and unsqueeze
            energies_torch = torch.tensor(energy_downsampled).unsqueeze(0)
            dos_torch = torch.tensor(dos_downsampled).unsqueeze(0)
            dos_cat = torch.cat((energies_torch, dos_torch), dim=0).float()
            dos_dict[material_ids[i]] = dos_cat
        else:
            # upsample
            # by factor of two - 1 element
            energies_upsampled = upsample_array(energies[i], 2)
            dos_upsampled = upsample_array(dos[i], 2)

            # convert to torch tensors and unsqueeze
            energies_torch = torch.tensor(energies_upsampled).unsqueeze(0)
            dos_torch = torch.tensor(dos_upsampled).unsqueeze(0)
            dos_cat = torch.cat((energies_torch, dos_torch), dim=0).float() 
            dos_dict[material_ids[i]] = dos_cat

    # save
    torch.save(dos_dict, './dos.pt')

    # extract bandgaps
    bandgap_dict = {}
    for material_id in material_ids:
        bandgap_id = material_id + '/bandgap'
        bandgap = f[bandgap_id][()]
        bandgap = torch.tensor(bandgap).float()
        
        bandgap_dict[material_id] = bandgap
            

def replace_nans(x):
    """Replace nan values with the average of its closest neighbors that aren't nan themselves. x is list of numpy arrays"""
    for i in range(len(x)):
        # if any element in x[i]
        if np.isnan(x[i]).any():
            for j in range(len(x[i])):
                if np.isnan(x[i][j]):
                    # find closest non-nan value
                    k = 1
                    while np.isnan(x[i][j-k]) and np.isnan(x[i][j+k]):
                        k += 1
                    if np.isnan(x[i][j-k]):
                        x[i][j] = x[i][j+k]
                    elif np.isnan(x[i][j+k]):
                        x[i][j] = x[i][j-k]
                    else:
                        x[i][j] = (x[i][j-k] + x[i][j+k]) / 2
    return x


def upsample_array(input_array, factor):
    x = np.arange(len(input_array))
    upsampled_x = np.linspace(0, len(input_array) - 1, len(input_array) * factor -1 )
    upsampled_array = np.interp(upsampled_x, x, input_array)
    return upsampled_array


if __name__ == '__main__':
    # generate .pt files
    generate_dos_bandgap()
    print('done')
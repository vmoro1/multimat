# create .pt file containing dielectric tensors. The tensors will be flattened to 1D tensors. The .pt file being loaded is alreadya a 
# dictionary but contains unnecessary values

import torch


def generate_dielectric_tensors(path_to_dielectric_tensors_file='./dielectric.pt'):

    dielectric = torch.load(path_to_dielectric_tensors_file)      # this is already a dict but it contains unnecessary values
    material_ids = list(dielectric.keys())

    for material_id in material_ids:
        ionic_tensor = dielectric[material_id]['ionic']
        electronic_tensor = dielectric[material_id]['electronic']

        # convert to torch tensors, flatten and concatenate
        ionic_tensor = torch.tensor(ionic_tensor).flatten()
        electronic_tensor = torch.tensor(electronic_tensor).flatten()
        dielectric_tensor = torch.cat((ionic_tensor, electronic_tensor))

        # replace old tensor with new flattened tensor
        dielectric[material_id] = dielectric_tensor.float()

    # save
    torch.save(dielectric, './dielectric.pt')


if __name__ == '__main__':
    generate_dielectric_tensors()
    print('done')

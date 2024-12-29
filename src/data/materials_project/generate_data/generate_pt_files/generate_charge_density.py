# generate dictionary type .pt file for charge densities. Dictoinary is on form {material_id: charge_density}

import torch


def generate_charge_density(path_to_charge_density_file='./charge_density_n32_17000.pt'):
    
    charge_densities = torch.load(path_to_charge_density_file)      # this is already a dict but it contains unnecessary values
    
    # only include charge density in dict, convert to torch tensor and unsqueeze
    charge_densities_dict = {material_id: torch.tensor(charge_densities[material_id]['total']).unsqueeze(0).float() for material_id in charge_densities.keys()}

    # save
    torch.save(charge_densities_dict, './charge_density.pt')


if __name__ == '__main__':
    # generate charge density .pt file
    generate_charge_density()
    print('done')
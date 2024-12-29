# Dataset for different modalities from materials project. To run this script as a demonstration, the path in "sys.path.append" must 
# be changed according to the users path. However, no changes are needed if it's only imported to another script, e.g to clip.py

import sys
import pickle
import numpy as np
import random

import torch
from src.utils.utils import collate

import pickle
import numpy as np

import os

def sort_key(material_id: str):
    """Sort material ids by there id and not lexographically by the entire string"""
    return int(material_id.split('-')[1])

class MatDataset(torch.utils.data.Dataset):
    """
    Dataset that can load and return a combination of a selection of modalities.
    The instance of the class takes a list of modalities and loads dictionaries from their paths
    Then it finds the intersection of the keys of each dictionary, which forms the indexing for __getitem__
    """
    def __init__(
            self,
            modalities: list = [],
            data_path: str = './data/',
            non_normalize_targets: bool = False,
            scalars_to_use: list = [],
            crystal_file: str = 'crystal_potnet.pt',
            file_to_keys: str = None,
            file_to_modalities_dicts: str = None,
            mask_non_intersect=True
            ):
        """
        :param modalities: list of modalities to load
        :param data_path: path to data directory
        :param normalize_targets: whether to normalize targets or not (mean 0, std 1 normalization)
        :param scalars_to_use: list of scalars to use for the scalar modality
        :param crystal_file: name of the crystal file to use (useful since different crystal fils are used for different models,
        e.g Matformer vs. PotNet)
        """
        self.mask_non_intersect = mask_non_intersect
        self.non_normalize_targets = non_normalize_targets
        self.modalities = modalities
        self.scalars_to_use = scalars_to_use
        
        self.path_to_crystal = os.path.join(data_path, crystal_file)
        self.path_to_dielectric = os.path.join(data_path,'dielectric.pt')
        self.path_to_efermi = os.path.join(data_path, 'efermi.pt')
        self.path_to_eform = os.path.join(data_path, 'eform.pt')
        if mask_non_intersect:
            self.path_to_dos = os.path.join(data_path, 'dos_allmpids.pt')
        else:
            self.path_to_dos = os.path.join(data_path, 'dos.pt')
        self.path_to_bandgap = os.path.join(data_path, 'bandgap.pt')
        self.path_to_is_metal = os.path.join(data_path, 'is_metal.pt')
        self.path_to_bands = os.path.join(data_path, 'bands.pt')
        if mask_non_intersect:
            self.path_to_charge_density_tensors = os.path.join(data_path, 'charge_density_allmpids', 'tensors')
            self.path_to_charge_density_materials = os.path.join(data_path, 'charge_density_allmpids', 'material_ids.pt')
        else:
            self.path_to_charge_density_tensors = os.path.join(data_path, 'charge_density', 'tensors')
            self.path_to_charge_density_materials = os.path.join(data_path, 'charge_density', 'material_ids.pt')
        self.path_to_dielectric = os.path.join(data_path, 'dielectric.pt')
        self.path_to_dielectric_eig = os.path.join(data_path, 'dielectric_eig.pt')
        self.path_to_bulk_modulus = os.path.join(data_path, 'bulk_modulus.pt')
        self.path_to_shear_modulus = os.path.join(data_path, 'shear_modulus.pt')
        self.path_to_elastic_tensor = os.path.join(data_path, 'elastic_tensor.pt')
        self.path_to_compliance_tensor = os.path.join(data_path, 'compliance_tensor.pt')
        if mask_non_intersect:
            self.path_to_text = os.path.join(data_path, 'mp_matbert_text_embeds_allmpids.pt')
        else:
            self.path_to_text = os.path.join(data_path, 'mp_matbert_text_embeds.pt')

        # modality to path
        self.modality_to_path = {
            'crystal': self.path_to_crystal,
            'efermi': self.path_to_efermi,
            'eform': self.path_to_eform,
            'dos': self.path_to_dos,
            'bandgap': self.path_to_bandgap,
            'is_metal': self.path_to_is_metal,
            'bands': self.path_to_bands,
            'dielectric': self.path_to_dielectric,
            'charge_density': (self.path_to_charge_density_tensors, self.path_to_charge_density_materials),
            'dielectric_eig': self.path_to_dielectric_eig,
            'bulk_modulus': self.path_to_bulk_modulus,
            'shear_modulus': self.path_to_shear_modulus,
            'elastic_tensor': self.path_to_elastic_tensor,
            'compliance_tensor': self.path_to_compliance_tensor,
            'text': self.path_to_text
        }

        if self.mask_non_intersect:
            crystal_dict = torch.load(self.path_to_crystal)
            self.keys = crystal_dict.keys()
            self.keys = list(self.keys)
            self.keys.sort(key=sort_key)
            self.modalities_dicts = {}
            for modality in self.modalities:
                if modality != 'charge_density':
                    self.modalities_dicts[modality] = torch.load(self.modality_to_path[modality])
                else:
                    self.modalities_dicts['charge_density'] = torch.load(self.modality_to_path[modality][1])

        else:
            if file_to_keys is None:
                print("Computing intersection keys..")
                self.keys, self.modalities_dicts = self.get_intersection_keys()
            else:
                print(f"Loading intersection keys from {file_to_keys}")
                self.keys = torch.load(file_to_keys)
                self.modalities_dicts = torch.load(file_to_modalities_dicts)
                assert(self.keys is not None)
            
        self.normalize()

       
    def get_intersection_keys(self):
        
        # modalities must be the the same as in the above dictionary (self.modality_to_path)
        assert set(self.modalities).issubset(set(list(self.modality_to_path.keys()) + ['scalars']))

        # The scalar modality (concatenation of scalars) needs to be handled differently
        include_scalar_modality = True if 'scalars' in self.modalities else False
        if include_scalar_modality: 
            self.modalities.remove('scalars')

        # fetch modalities with list comprehension
        modalities_dicts = {}
        for modality in self.modalities:
            if modality != 'charge_density':
                modalities_dicts[modality] = torch.load(self.modality_to_path[modality])
            else:
                modalities_dicts['charge_density'] = torch.load(self.modality_to_path[modality][1])

        # check tht data is alright
        # modalities_with_problems = []
        for modality in self.modalities:
            self.check_data(modality, modalities_dicts)

        # find intersection of keys (first modality can be crystal and that needs to be handled separately)
        try:
            keys = set(modalities_dicts[self.modalities[0]].keys())
        except:
            keys = set(modalities_dicts['charge_density'])
            
        for modality in modalities_dicts:
            if modality != 'charge_density':
                keys = keys.intersection(set(modalities_dicts[modality].keys()))
            else:
                keys = keys.intersection(set(modalities_dicts['charge_density']))
        
        keys = list(keys)
        keys.sort(key=sort_key)   # need to sort since set is unordered --> causes problem when splitting dataset

        # create scalars dataset if needed
        if include_scalar_modality:
            scalar_data_missing = {}
            for scalar in self.scalars_to_use:
                if scalar not in self.modalities:
                    scalar_data_missing[scalar] = torch.load(self.modality_to_path[scalar])

                    # check that data is alright
                    self.check_data(scalar, scalar_data_missing)
            
            # combine dicts so that all scalars are available in the same dict
            scalar_data_missing.update(modalities_dicts)

            # create scalars data dict
            scalars_dict = {}
            for key in keys:
                scalars_list = []
                for scalar in self.scalars_to_use:
                    try:
                        scalars_list.append(scalar_data_missing[scalar][key])
                    except:
                        scalars_list.append(torch.nan)

                scalars_dict[key] = torch.tensor(scalars_list).unsqueeze(0)

            # add scalars to modalities_dicts and include in modalities again
            modalities_dicts['scalars'] = scalars_dict
            self.modalities.append('scalars')

        return keys, modalities_dicts
    
    def normalize(self):
        # normalization (only targets and concatenation of scalars)
        self.mean = {}
        self.std = {}
        if not self.non_normalize_targets:
            tasks_to_normalize = ['efermi', 'eform', 'bandgap', 'bulk_modulus', 'shear_modulus', 'dielectric', 
                                  'dielectric_eig', 'elastic_tensor', 'compliance_tensor', 'scalars']
            for task in self.modalities:
                if task in tasks_to_normalize:
                    to_stack = True if type(self.modalities_dicts[task][self.keys[0]]) == torch.Tensor else False
                    data_values = [self.modalities_dicts[task][material_id] for material_id in self.keys]
                    data_values = torch.stack(data_values) if to_stack else torch.tensor(data_values)
                    mean = torch.mean(data_values, dim=0)
                    std = torch.std(data_values, dim=0)
                    for material_id in self.keys:
                        self.modalities_dicts[task][material_id] = (self.modalities_dicts[task][material_id] - mean) / std

                    self.mean[task] = mean
                    self.std[task] = std
    
    def check_data(self, modality, modalities_dicts):
        # skip crystal
        if modality == 'crystal':
            return 
        
        # material ids
        if self.mask_non_intersect:
            material_ids = list(modalities_dicts[modality].keys())
        else:
            if modality == 'charge_density':
                material_ids = modalities_dicts[modality]
            else:
                material_ids = list(modalities_dicts[modality].keys())

        # remove materials containing infs or nans (if applicable)
        num_materials_with_problems = 0
        if self.mask_non_intersect:
            if modality == 'charge_density':
                pass

            elif type(modalities_dicts[modality][material_ids[0]]['data']) == torch.Tensor:
                for material_id in material_ids:
                    material_data = modalities_dicts[modality][material_id]['data']
                    if torch.isnan(material_data).any() or torch.isinf(material_data).any():
                        modalities_dicts[modality].pop(material_id)
                        num_materials_with_problems += 1
            else:
                for material_id in material_ids:
                    material_data = modalities_dicts[modality][material_id]['data']
                    if not isinstance(material_data, (int, float)):
                        modalities_dicts[modality].pop(material_id)
                        num_materials_with_problems += 1
        else:
            if modality == 'charge_density':
                pass
            elif type(modalities_dicts[modality][material_ids[0]]) == torch.Tensor:
                for material_id in material_ids:
                    material_data = modalities_dicts[modality][material_id]
                    if torch.isnan(material_data).any() or torch.isinf(material_data).any():
                        modalities_dicts[modality].pop(material_id)
                        num_materials_with_problems += 1
            else:
                for material_id in material_ids:
                    material_data = modalities_dicts[modality][material_id]
                    if not isinstance(material_data, (int, float)):
                        modalities_dicts[modality].pop(material_id)
                        num_materials_with_problems += 1
        print(f"Number of materials with problems for {modality}: {num_materials_with_problems}")

    def __len__(self):
        return len(self.keys)
                   

    def __getitem__(self, idx):
        data = {}
        if self.mask_non_intersect:
            exist_label = {}
            for modality in self.modalities:
                if modality == 'crystal':
                    data[modality] = self.modalities_dicts[modality][self.keys[idx]]
                    exist_label[modality] = 1
                elif modality != 'charge_density':
                    data[modality] = self.modalities_dicts[modality][self.keys[idx]]['data']
                    exist_label[modality] = self.modalities_dicts[modality][self.keys[idx]]['exists']
                else:
                    charge_density_path = os.path.join(self.path_to_charge_density_tensors, self.keys[idx] + '.pt')
                    charge_density_tensor = torch.load(charge_density_path)
                    data['charge_density'] = charge_density_tensor
                    exist_label['charge_density'] = self.modalities_dicts[modality][self.keys[idx]]
            return data, exist_label
        else:
            for modality in self.modalities:
                if modality != 'charge_density':
                    data[modality] = self.modalities_dicts[modality][self.keys[idx]]
                else:
                    charge_density_path = os.path.join(self.path_to_charge_density_tensors, self.keys[idx] + '.pt')
                    charge_density_tensor = torch.load(charge_density_path)
                    data['charge_density'] = charge_density_tensor
            return data
    
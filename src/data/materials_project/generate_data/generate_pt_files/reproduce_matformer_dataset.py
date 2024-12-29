# Reproduce matformer dataset

import sys
from src.model.matformer.data import get_train_val_loaders


target = 'gap pbe'
batch_size = 64
if target == "e_form" or target == "gap pbe":
    n_train = 60000
    n_val = 5000
    n_test = 4239 
else:
    n_train = 4664
    n_val = 393
    n_test = 393


(train_loader, val_loader, test_loader,prepare_batch, mean_train, std_train) = get_train_val_loaders(
    dataset='megnet',
    target=target,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size=batch_size,
    atom_features="cgcnn",
    neighbor_strategy="k-nearest",
    standardize=False,
    line_graph=False,
    id_tag="id",
    pin_memory=False,
    workers=8,
    save_dataloader=False,
    use_canonize=True,
    filename='not_save',
    cutoff=8.0,
    max_neighbors=12,
    output_features=1,
    classification_threshold=None,
    target_multiplication_factor=None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_dir='datasets',
    matrix_input=False,
    pyg_input=True,
    use_lattice=True,
    use_angle=False,
    use_save=False,
    mp_id_list=None)

print('Done!')

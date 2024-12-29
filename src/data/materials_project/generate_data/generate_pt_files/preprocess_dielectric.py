import torch

normalize = True 

if normalize:
    path = './data/dielectric_unnormalized_eig.pt'
else:
    path = './data/dielectric_unnormalized.pt'

data = torch.load(path)

all_data = torch.zeros((len(data), 6))

# loop thru the data
for i, (k, v) in enumerate(data.items()):
    if normalize:
        all_data[i] = data[k]
    else:
        # obtains the matrices
        mat1 = v[:9].view(3, 3)
        mat1 = (mat1 + mat1.t()) / 2
        mat2 = v[9:].view(3, 3)
        mat2 = (mat2 + mat2.t()) / 2

        # extracts the eigenvalues
        eig1, _ = torch.linalg.eig(mat1)
        eig2, _ = torch.linalg.eig(mat2)
        
        # sorts the tensors
        eig1 = eig1.real.sort()[0]
        eig2 = eig2.real.sort()[0]


        data[k] = torch.cat([eig1, eig2])
        print(f'Finished {i}th iteration', data[k])

# normalize the data
if normalize:
    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0)
    for i, (k, v) in enumerate(data.items()):
        data[k] = (v - mean) / std
        print(f'Finished {i}th iteration', data[k])
    torch.save(data, path[:-3] + '_normalized.pt')
else:
    torch.save(data, path[:-3] + '_eig.pt')
print('Saved the data!')




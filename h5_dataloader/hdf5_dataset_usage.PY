'''
structure of database: train.h5 files saved in './folder' for example,
dtype = np.float32

structure of h5 files: 
f = h5.File('train.h5')
f.keys(): ['group1', 'group2',...,'groupN'] (group names can be customized),
g1 = f['group1']
g1.keys(): ['data', 'label'], which has been modified to enable customized dataset structure, e.g. ['img', 'csm','rrs']
'''
from hdf5_dataset import HDF5Dataset
from torch.utils import data as data_vtils

folder = 'data/test'
loader_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 2}
structure=['img', 'csm','rrs']

if __name__ == '__main__':
   dataset = HDF5Dataset(folder, recursive=True, load_data=False, data_cache_size=2, transform=None, structure=structure)

   data_loader = data_vtils.DataLoader(dataset, **loader_params)

   _, (img,csm,rrs) = next(enumerate(data_loader))
   print((img.shape, csm.shape, rrs.shape), (img.dtype, csm.dtype, rrs.dtype))

   num_epochs = 2

   for i in range(num_epochs):
      for j, (img,csm,rrs) in enumerate(data_loader):
         print('(E, iter):(%d, %d)'%(i, j), *structure, (img.shape, csm.shape, rrs.shape), (img.dtype, csm.dtype, rrs.dtype))

   # python hdf5_dataset_usage.py

# f = h5.File('test2.h5','w')

# for i in range(3):
#    img = np.random.randn(2, 64, 32).astype(np.float32)
#    csm = np.random.randn(2, 64, 32, 15).astype(np.float32)
#    rrs = np.random.randn(64, 32).astype(np.float32)
#    g = f.create_group('group%d'%i)
#    g.create_dataset('img', data=img)
#    g.create_dataset('csm', data=csm)
#    g.create_dataset('rrs', data=rrs)

# f.close()


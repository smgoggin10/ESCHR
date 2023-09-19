import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import zarr


def csv_to_zarr(csv_path, zarr_loc):
    csv_in = pd.read_csv(csv_path, index_col=0)
    print("csv in")
    csv_in = coo_matrix(csv_in.to_numpy())
    
    z1 = zarr.open(zarr_loc, mode='w')
    X = z1.create_group('X')
    
    data_row = X.create_dataset(name='row', shape=csv_in.row.shape, chunks=False, dtype='int32', overwrite=True)
    data_row[:] = csv_in.row
    
    data_col = X.create_dataset(name='col', shape=csv_in.col.shape, chunks=False, dtype='int32', overwrite=True)
    data_col[:] = csv_in.col
    
    data_data = X.create_dataset(name='data', shape=csv_in.data.shape, chunks=False, dtype='float32', overwrite=True)
    data_data[:] = csv_in.data


def make_zarr(data, zarr_loc):
    data_coo = coo_matrix(data)
    
    z1 = zarr.open(zarr_loc, mode='w')
    X = z1.create_group('X')
    
    data_row = X.create_dataset(name='row', shape=data_coo.row.shape, chunks=False, dtype='int32', overwrite=True)
    data_row[:] = data_coo.row
    
    data_col = X.create_dataset(name='col', shape=data_coo.col.shape, chunks=False, dtype='int32', overwrite=True)
    data_col[:] = data_coo.col
    
    data_data = X.create_dataset(name='data', shape=data_coo.data.shape, chunks=False, dtype='float32', overwrite=True)
    data_data[:] = data_coo.data

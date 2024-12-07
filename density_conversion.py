import numpy as np
import os
import glob
import time as tm

def name_reader(directory_path, pattern):
    return sorted(glob.glob(directory_path + '/' + pattern))[:]

def save(folder, name):
    directory_path = os.getcwd() + folder
    matching_file = sorted(glob.glob(directory_path + '/' + 'mask*'))
    
    mask = np.load(matching_file[0])
    limit = 25000
    file_paths = name_reader(directory_path,  'coorx*')
    coorx = np.concatenate([np.load(file_path)[:limit, :] for file_path in file_paths], axis=0)
    file_paths = name_reader(directory_path,  'coory*')
    coory = np.concatenate([np.load(file_path)[:limit, :] for file_path in file_paths], axis=0)
    ln = coory.shape[0]
    print(ln)
    timestart = 100
    timelimit = 300
    coorx = coorx[:, timestart:timelimit]
    coory = coory[:, timestart:timelimit]

    m = 1
    grd = np.zeros((mask.shape[1] // m, mask.shape[0] // m))
    combined = np.column_stack((np.floor(coorx).astype(int).ravel(), np.floor(coory).astype(int).ravel()))
    start_time = tm.time()
    coordinates, values = np.unique(combined, axis=0, return_counts=True)
    end_time = tm.time()
    print("Время выполнения:", end_time - start_time, "секунд")
    grd[coordinates[:, 0], coordinates[:, 1]] = values
    grd = np.rot90(grd, k=1) 
    np.save(name + '.npy', grd)

save('/folder', 'data/folder')  

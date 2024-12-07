import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import copy
import pandas as pd
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 16}) 


def load(filename, title, section, sigma):
    s = np.round(np.load('data/' + filename))
    s = s[10:s.shape[0] - 10, 30:s.shape[1] - 120]
    global x
    global y
    x = int(round(s.shape[1], - 2))
    y = int(round(s.shape[0], - 2))
    if show_prof:
        plt.figure(figsize=(10, 5))
        s_show = copy.deepcopy(s)
        s_show[:, section] = 42
        plt.title(title)
        plt.imshow(1 + s_show, norm=LogNorm(), cmap='plasma')  
        plt.show()
    s = gaussian_filter(s, sigma=sigma)
    return s
    
def characteristics(slice, lab):
    square = np.trapz(slice)
    arg_semi = np.argmin(np.abs(slice - np.max(slice) / 2))
    arg_full = np.argmin(np.abs(slice - np.max(slice)))
    semi_wid = np.abs(arg_full - arg_semi)
    print(semi_wid, square)
    new_row = [lab, semi_wid, square]
    data.loc[len(data)] = new_row

data = pd.read_excel("Results.xlsx", index_col=0) 
show_prof = 1
smooth = 2

name = 'now'
lab = name
slice = load(name + '.npy', lab, 400, smooth)[:, 400]
window_size = 15
kernel = np.ones(window_size) / window_size
slice = np.convolve(slice, kernel, mode='valid')
plt.plot(slice, label = lab)
characteristics(slice, lab)

#data.to_excel('Results.xlsx')  
#Output in the required axes

'''
numx = 6
x_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in np.linspace(0, y * 0.25, num=numx)]
plt.xticks(np.linspace(0, y, num=numx), x_ticks)
plt.xlabel('y, мм')
plt.ylabel('Плотность')
plt.grid(color='black', linestyle='-', linewidth=0.2)
plt.legend()
plt.show()'''


'''
numx = 11
numy = 5
x_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in np.linspace(0, x * 0.25, num=numx)]
y_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in np.linspace(0, y * 0.25, num=numy)]
plt.title(title)
plt.xticks(np.linspace(0, x, num=numx), x_ticks)
plt.yticks(np.linspace(0, y, num=numy), y_ticks)
plt.xlabel('x, мм')
plt.ylabel('y, мм')
plt.colorbar()'''


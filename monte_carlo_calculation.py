import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

'''
размер одной ячейки - 2.5e-5 м
вce размерные переменные задаются по принципу - нужный размер делить на размер одной ячейки
shape_y - высота расчетной области
shape_x - ширина расчетной области
time - количество временных шагов
size - количество частиц
fullsize - количество частиц с учетом дублирования частиц
N - максимальное количество частиц в ячейке
height - полувысота капилляра
Vx - скорость по x при начальном заполнении капилляра
Vy - скорость по y при начальном заполнении капилляра
thresh - длина капилляра
x_min_lim - координата по x где частицы уходят из расчета
mask - маска диафрагмы
bias - отступ по y где частицы уходят из расчета
t_step - временной шаг
V_pot - скорость потока в капилляре
l - ширина диафрагмы
l2 - расстояние между осью и нижней точкой диафрагмы
coor - координата по x левого края диафрагмы
'''

#Расчет скорости частицы в соответствии с распределением Максвелла
def inverse_maxwell_distribution(x):
    mHel = 6.6464764e-27 
    k = 1.380649e-23  
    T = 300
    return  np.sqrt((-2 * k * T / mHel) * np.log(1 - x)) * 1e4


#Расчет относительной скорости частиц и абсолютной скорости каждой частицы
def calc_particle_velocities(particles):
    l = len(particles)
    l2 = int((l - 1) * l // 2)
    pairs = np.zeros((l2, 9))
    n = 0
    for j in range(l):
        for i in range(j + 1, l):
            pairs[n][0] = particles[i][0]
            pairs[n][1] = particles[j][0]
            pairs[n][2] = np.sqrt((particles[i][1] - particles[j][1])**2 + (particles[i][2] - particles[j][2])**2)
            pairs[n][3] = particles[i][1] - particles[j][1]
            pairs[n][4] = particles[i][2] - particles[j][2]
            pairs[n][5] = particles[i][1]
            pairs[n][6] = particles[j][1]
            pairs[n][7] = particles[i][2]
            pairs[n][8] = particles[j][2]
            n += 1
    return pairs


#Расчет скорости после отражения от стенки
def new_vel(n, vx, vy):
    sc1 = n[0] * vx
    sc2 = n[1] * vy
    len = n[0]**2 + n[1]**2 
    vx_out = vx - 2 * (sc1 * n[0] + sc2 * n[0]) / len
    vy_out = vy - 2 * (sc1 * n[1] + sc2 * n[1]) / len
    return 0.9 * vx_out, 0.9 * vy_out


#Расчет геометрии диафрагмы
def wall(l, l2):
    x = np.arange(0, l)
    y = np.arange(0, l)
    X, Y = np.meshgrid(x, y)
    wall = np.zeros((l2 - l, l))
    mask = + np.where(Y > l - 2 * X - 1, 4, 0) + np.where(Y > l - 0.2 * X - 1, 8, 0) + np.where(Y > 0.8 * X, 2, 0)
    #mask =+ np.where(Y > l - 1.5 * X - 1, 4, 0) + np.where(Y > l - X - 1, 8, 0) + np.where(Y > X - l, 2, 0)
    #mask = np.flip(mask, axis = 1)
    #mask =+ np.where(Y > l - X - 1, 4, 0) + np.where(Y > l - 1, 8, 0) + np.where(Y > X, 2, 0)

    mask[(mask == 2) | (mask == 4)] = 0
    mask[(mask == 10) | (mask == 14) | (mask == 12)] = 18
    mask[(mask == 20) | (mask == 18)] = 35
    wall = np.concatenate((wall, mask), axis = 0)
    return wall


#Компановка диафрагм
def mesh(shape_y, shape_x, height, m):
    shape_y, shape_x = shape_y * m, shape_x * m
    height = height * m
    mask = 35 * np.ones((shape_y, shape_x))
   
    l, l2, coor = 20, int(shape_y // 2 - 4 * height), 160
    l, l2, coor = l * m, l2, coor * m
    mask[:l2, coor:coor + l] = wall(l, l2)
    mask[shape_y - l2:, coor:coor + l] = np.flipud(wall(l, l2)) + 2
    
    mask[(mask == 10) | (mask == 12)] = 35
    return mask

#Задание параметров
shape_x = 600
shape_y = 240
time = 50
size = int(5e3)
ln = int(5e3)
fullsize = int(10e3)
prob = 0.2
N = 7
height = 1
Vx = 1e7
Vy = height
thresh = 80
x_min_lim = 70
m = 1
mask = mesh(shape_y, shape_x, height, m)
bias = 10
t_step = 1e-7
V_pot = 1e7
folder = 'now'

#Задание массивов координат и скоростей
coory = np.ones((fullsize, time))
coorx = np.ones((fullsize, time))
coorx[:, 0] = np.random.uniform(5, thresh, fullsize)
coory[:, 0] =  np.random.uniform(shape_y//2 - 0.5 -height, shape_y//2 - 0.5 + height, fullsize)
vy = np.zeros(fullsize)
vx = np.zeros(fullsize)
vx_f = np.zeros(fullsize)
velocities = []
is_in = np.ones(fullsize)

#Проход по временному циклу
for i in range(1,time):
    print(size, ' - len,', i, ' - time')
    grd = np.zeros((shape_y, shape_x, N, 3))
    grd_ch = np.zeros((shape_y, shape_x))

    #Проход по частицам
    for j in range(1, size): 
        if is_in[j]:
            vel = np.sqrt(vx[j]**2 + vy[j]**2) * 1e-8
            if vel != 0 and vx_f[j] == 1: velocities.append(vel)
        
            if coorx[j, i - 1] > thresh and vx_f[j] == 0:      
                vx_f[j] = 1
                v = inverse_maxwell_distribution(np.random.rand(1))
                ang = 2 * np.pi * np.random.uniform(0, 1)
                vy[j] = v * np.sin(ang)
                vx[j] = Vx + v * np.cos(ang)
                if (coory[j, i - 1] > shape_y // 2 + height - 0.5): coory[j, i - 1] = shape_y // 2 + height - 0.5
                if (coory[j, i - 1] < shape_y // 2 - height - 0.5): coory[j, i - 1] = shape_y // 2 - height - 0.5

            if  vx_f[j] == 0:
                vy[j] = 1
                vx[j] = V_pot
                if (coory[j, i - 1]  > shape_y // 2 + height - 0.5): coory[j, i - 1] = shape_y // 2 + height - 0.5
                if (coory[j ,i - 1]  < shape_y // 2 - height - 0.5): coory[j, i - 1] = shape_y // 2 - height - 0.5
                    
            #Размножение частиц
            if size < fullsize:
                out = vx[j] > 0 and vx_f[j] == 1
                if out and vx_f[j] == 1:
                    cond1 = coorx[j, i - 1] < 159
                    cond2 = coorx[j, i - 1] > shape_y - coory[j, i - 1] + 20
                    cond3 = coorx[j, i - 1] > coory[j, i - 1] + 20
                    ccond1 = cond1 and cond2 and cond3
                    cond1 = coorx[j, i - 1] < 159 + 100
                    cond2 = coorx[j, i - 1] > shape_y - coory[j, i - 1] + 20 + 100
                    cond3 = coorx[j, i - 1] > coory[j, i - 1] + 20 + 100
                    ccond2 = cond1 and cond2 and cond3
                    
                    if ccond1 or ccond2:
                        if np.random.uniform(0, 1) < prob:
                            coorx[size, :i - 2] = coorx[j, i - 1]
                            coorx[size, i - 2:] = coorx[j, i - 1] + 0.5 * np.sign(np.random.uniform(-1, 1)) * np.random.uniform(0, 1)
                            coory[size, :i - 2] = coory[j, i - 1]
                            coory[size, i - 2:] = coory[j, i - 1] + 0.5 * np.sign(np.random.uniform(-1, 1)) * np.random.uniform(0, 1)
                            vx[size] = vx[j]
                            vy[size] = vy[j]
                            vx_f[size] = vx_f[j]
                            size += 1
            
            calcDouble = vx[j] > 0 or j < ln
            #Добавление частиц в расчетную сетку для метода Монте-Карло
            if calcDouble:
                a = grd[round(coory[j, i - 1]), round(coorx[j, i - 1])] 
                index = int(grd_ch[round(coory[j, i - 1]), round(coorx[j, i - 1])])
                if index < N:
                    a[index][0] = j
                    a[index][1] = vx[j]
                    a[index][2] = vy[j]
                    grd_ch[round(coory[j, i - 1]), round(coorx[j, i - 1])] += 1
                    
    #Проход по сетке для расчета скоростей частиц по методу Монте-Карло
    for cell_i in range(thresh, thresh + 250):
        for cell_j in range(bias, shape_y - bias):
            num_part = int(grd_ch[cell_j, cell_i])
            if num_part > 0:

                a = grd[cell_j, cell_i]
                a = a[:num_part, :]

                if num_part >= 2:
                    b = calc_particle_velocities(a)
                    max_element = np.max(b[:, 2])

                    for k in b: 
                        if k[2] > max_element * np.random.uniform(0, 1):
                            q =  k[0].astype(int)
                            w =  k[1].astype(int)
                            zn = np.sign(np.random.uniform(-1, 1))
                            zn2 = np.sign(np.random.uniform(-1, 1))
                            ang1 = 2 * np.pi * np.random.uniform(0, 1)
                        
                            vx[q] = (k[5] + k[6]) / 2 + zn * k[2] / 2 * np.cos(ang1)
                            vy[q] = (k[7] + k[8]) / 2 + zn2 * k[2] / 2 * np.sin(ang1)
                            vx[w] = (k[5] + k[6]) / 2 - zn * k[2] / 2 * np.cos(ang1)
                            vy[w] = (k[7] + k[8]) / 2 - zn2 * k[2] / 2 * np.sin(ang1)
            
    #Второй проход по частицам                        
    for j in range(1, size): 
        if is_in[j]:
            a = mask[round(m * coory[j, i - 1]), round(m * coorx[j, i - 1])]
            if a == 0 or a == 2:
                nor = np.array([-1, 0])
                vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                coorx[j, i - 1] = coorx[j,i - 2]
                coory[j, i - 1] = coory[j,i - 2]
            
            if a == 6:
                nor = np.array([2, 10])
                vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                coorx[j, i - 1] = coorx[j,i - 2]
                coory[j, i - 1] = coory[j,i - 2]
                
            if a == 8:
                nor = np.array([2, -10])
                vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                coorx[j, i - 1] = coorx[j,i - 2]
                coory[j, i - 1] = coory[j,i - 2]
    
            #nor = np.array([0, -1])
            #nor = np.array([1, -1])


        coory[j, i] = coory[j, i - 1] + vy[j] * t_step
        coorx[j, i] = coorx[j, i - 1] + vx[j] * t_step

        if vx_f[j] == 1:
            cond1 = coory[j, i] > shape_y - bias or coory[j, i] < bias 
            cond3 = coorx[j, i] > shape_x - bias or coorx[j, i] < x_min_lim
            if cond1 or cond2 or cond3:
                is_in[j] = 0
                if coorx[j, i] > shape_x - bias: coorx[j, i:] = shape_x - bias
                if coorx[j, i] < x_min_lim: coorx[j, i:] = x_min_lim
                if coory[j, i] > shape_y - bias: coory[j, i:] = shape_y - bias
                if coory[j, i] < bias: coory[j,i:] = bias / 2

coorx = coorx[:size, :] 
coory = coory[:size, :] 
#Вывод траекторий, если требуется
if 0:
    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    for i in range(1, size):
        plt.plot(m * coorx[i,:], m * coory[i,:], linewidth = 0.5)
        #plt.scatter(coorx[i,:], coory[i,:], s = 0.6)
    plt.imshow(mask)
    plt.show()

os.system('afplay /System/Library/Sounds/Glass.aiff')

#Сохранение координат частиц в файл
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

params = {
    'sh': shape_x,
    'len': size,
    'time': time,
    'height': height,
    'bias': bias,
    'thresh': thresh,
    'current_datetime': current_datetime
}


filename_mask = 'data/mask.npy'
filename_coorx = 'data/coorx.npy'
filename_coory = 'data/coory.npy'
filename_mask2 = folder + '/mask_sh={sh}_time={time}_len={len}_height={height}_bias={bias}_thresh={thresh}_{current_datetime}.npy'.format(**params)
filename_coorx2 = folder + '/coorx_sh={sh}_time={time}_len={len}_height={height}_bias={bias}_thresh={thresh}_{current_datetime}.npy'.format(**params)
filename_coory2 = folder + '/coory_sh={sh}_time={time}_len={len}_height={height}_bias={bias}_thresh={thresh}_{current_datetime}.npy'.format(**params)

np.save(filename_coorx, coorx)
np.save(filename_coory, coory)
np.save(filename_mask, mask)
np.save(filename_coorx2, coorx)
np.save(filename_coory2, coory)
np.save(filename_mask2, mask)

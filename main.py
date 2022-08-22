import numpy as np
import cv2 as cv
import sys

from pyx import *

nop = 30
frec = 3

def dfs(data, points, cur_num_of_points, x, y):
    flag = 0
    stk = []
    stk.append((x, y))
    data[x][y] = 2
    points[-1].append([x, y])
    
    while(len(stk)):
        for i in range(min(stk[-1][0] + 1, data.shape[0] - 1), max(stk[-1][0] - 2, 0), -1):
            for j in range(min(stk[-1][1] + 1, data.shape[1] - 1), max(stk[-1][1] - 2, 0), -1):
                if (data[i][j] == 255):
                    flag = 1
                    stk.append((i ,j))
                    data[i][j] = 1
                    if (len(stk) % frec == 0):
                        if (data[max(i - 1, 0):min(i + 2, data.shape[0]), max(j - 1, 0):min(j + 2, data.shape[1])] == 2).sum() == 0:
                            data[i, j] = 2
                            points[-1].append([i, j])
                if flag:
                    break
            if flag:
                break
        if flag:
            flag = 0
        else:
            points.append([])
            stk.pop()

def creat_mtx(size): # num_of_points
    a = np.zeros((size, size))
    a[0, 0] = 2
    a[0, 1] = -1
    a[size - 1, size - 2] = -1
    a[size - 1, size - 1] = 2
    i = np.arange(1, size - 1, 2)
    j = i - 1
    a[i, j] = 1
    a[i, j + 1] = -2
    a[i, j + 2] = 2
    a[i, j + 3] = -1
    a[i + 1, j + 1] = 1
    a[i + 1, j + 2] = 1
    return a

def vect(points):
    size = points.shape[0] * 2
    A = creat_mtx(size)
    Bx = np.zeros(size)
    By = np.zeros(size)
    
    Bx[2:size:2] = 2*points[1:points.shape[0],0]
    Bx[0] = points[0, 0]
    By[2:size:2] = 2*points[1:points.shape[0],1]
    By[0] = points[0, 1]
    
    Bx[-1] = points[-1][0]
    By[-1] = points[-1][1]
    X = np.linalg.solve(A, Bx)
    Y = np.linalg.solve(A, By)
    return np.vstack((X, Y)).T

def add_curves(supp, points, sizes, data):
    AP = np.zeros((len(points) + len(supp), 2))
    AP[::3] = points
    AP[1::3] = supp[::2]
    AP[2::3] = supp[1::2]
    
    
    for i in range(0, AP.shape[0] - 3, 3):
        data.stroke(path.curve(AP[i][0], AP[i][1], AP[i + 1][0], AP[i + 1][1], AP[i + 2][0], AP[i + 2][1], AP[i + 3][0], AP[i + 3][1]), [style.linewidth.THICK])
        

if __name__ == '__main__':

    img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    img = np.rot90(img, -1)
    sizes = img.shape
    
    edge = cv.Canny(img, 200, 300)
    
    edge[edge < 255] = 100
    img = edge.copy()
#    cv.imwrite('edge.jpg', np.rot90(edge))
    points = []
    
    data = canvas.canvas()
    form = document.paperformat(300, (sizes[1] * 300) // sizes[0], 'myform')
    
    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            if (img[i, j] == 255):
                points.append([])
                dfs(img, points, 0, i, j)
                
    for p in points:
        if (len(p) >= 1):
            p = np.array(p)
            supp = vect(p)
            add_curves(supp, p, img.shape, data)
    pg = document.page(data, paperformat = form, fittosize = 1)
    doc = document.document(pages = [pg])
    doc.writePDFfile('result')

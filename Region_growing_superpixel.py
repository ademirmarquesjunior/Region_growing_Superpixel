# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:42:01 2019

@author: ADEJUNIOR
"""

# Library imports

import cv2
import numpy as np
import PySimpleGUI as sg


from PIL import Image
import math

import csv

import base64
from io import BytesIO

from skimage.segmentation import slic

# from numba import jit


#  Screen layout definitions

menu_def = [['File', ['Open image', 'Open Superpixel', 'Generate Superpixels',
                      'Load CSV file', 'Save mask', 'Exit']],
            ['Help', ['About...', 'Reload']]]


layout = [[sg.Menu(menu_def, tearoff=True)],
          [sg.Graph(canvas_size=(500, 500), graph_bottom_left=(0, 0),
                    enable_events=True, drag_submits=True,
                    graph_top_right=(500, 500), key="_Canvas1_"),
           sg.Slider(range=(100, 0), orientation='v', size=(20, 10),
                     enable_events=True, disable_number_display=True,
                     default_value=0, key="_Sl_vertical_"),
           sg.Frame("", [[sg.Frame("Class annotation", [
                [sg.Button('Original', button_color=['green', 'white'])],
                [sg.Button('Class 1', button_color=['white', 'red']),
                 sg.Button('Class 2', button_color=['white', 'green']),
                 sg.Button('Class 3', button_color=['white', 'blue'])],
                [sg.Text('Class 1 selected', key='_Tx_selected_')],
                [sg.Input(key='_In_sigmaH_', size=(5, 5), default_text='19'),
                 sg.Button('Extrapolate'), sg.Button('Fill'),
                 sg.Button('Reset')], ])],
                ])],
          [sg.Slider(range=(0, 100), orientation='h', size=(40, 10),
                     enable_events=True, disable_number_display=True,
                     default_value=0, key="_Sl_horizontal_")], ]

window = sg.Window('Superpixel Growing Segmentation', layout)
canvas1 = window.Element("_Canvas1_")
temp = np.zeros(canvas1.CanvasSize)

# Initial global values
y = 0
x = 0
maxDist = 19
active_class = 'class1'
classe = 1

tempColor = []


def updateCanvas(image, hor, ver):
    '''
    Draw a given image in the Canvas given the horizontal and vertical
    position.
    The image is buffered in a binary stream

    Parameters
    ----------
    image : TYPE
        RGB image.
    hor : TYPE
        Initial horizontal position in the image.
    ver : TYPE
        Initial vertical position in the image.

    Returns
    -------
    image : TYPE
        RGB image.

    '''

    if np.size(np.shape(image)) == 3:
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB)

    size = canvas1.CanvasSize
    positionX = int(np.shape(image)[1]/100*hor)
    positionY = int(np.shape(image)[0]/100*ver)

    if positionX > np.shape(image)[1]-size[1]:
        positionX = np.shape(image)[1]-size[1]-1
    if positionY > (np.shape(image)[0]-size[0]):
        positionY = np.shape(image)[0]-size[0]-1

    try:
        buffered = BytesIO()

        Image.fromarray(np.uint8(image[positionY:size[0]+positionY,
                                       positionX:size[1]+positionX])
                        ).save(buffered, format="PNG")

        encoded = base64.b64encode(buffered.getvalue())
        canvas1.DrawImage(data=encoded, location=(0, 500))
        canvas1.Update()
    except Exception as e:
        print(e)

    return image


def paintSuperpixel(superpixel, target, image=None, index=None,
                    color=[255, 255, 255]):
    '''
    Paint certain pixels according to superpixel index given.

    Parameters
    ----------
    superpixel : TYPE
        Superpixel image with indexes for each pixel.
    target : TYPE
        Image to be painted with specific superpixel.
    image : TYPE, optional
        DESCRIPTION. The default is None.
    index : TYPE, optional
        DESCRIPTION. The default is None.
    color : TYPE, optional
        DESCRIPTION. The default is [255,255,255].

    Returns
    -------
    target : TYPE
        DESCRIPTION.

    '''

    for i in range(0, np.shape(image)[0]-1):
        for j in range(0, np.shape(image)[1]-1):
            if superpixel[i, j] == index:
                target[i, j] = color
    return target


def computeSuperpixelColor(image, superpixel):
    '''
    Compute the centroid color in a CIELab space for each superpixel.
    
    @param image: The reference image
    @param superpixel: The superpixel image with the indexes for each pixel
    @return superpixelColor: A list of colors with three colluns for each superpixel
    '''
    
    superpixelColor = np.zeros((superpixel.max()+1, 3))
    
    for i in range(1, superpixel.max()+1):
        colorSum = []
        for j in range(0, np.shape(image)[0]):
            for k in range(0, np.shape(image)[1]):
                if superpixel[j][k] == i:
                    colorSum.append(image[j][k])
                   

        colorSum = np.reshape(colorSum, (np.shape(colorSum)[0],np.shape(colorSum)[1]))
        superpixelColor[i] = [np.median(colorSum[:,0]), np.median(colorSum[:,1]), np.median(colorSum[:,2])]
    print('Colors computed')
    return superpixelColor



def returnNeighbors(index, neighbors):
    '''
    Returns the superpixel neighbors of a given superpixel.
    
    @param index: Reference superpixel index
    @param neighbors: Sparse matrix of superpixel neighbors
    @retun temp: List of neighbors
    '''
    temp = neighbors[index:index+1,:]
    return np.unique(np.sort(np.where(temp == True)[1]))



def setNeighbors(superpixel):
    '''
    Computes neighbors of each superpixel by looking the indexes in each pixel of superpixel image.
    
    @param superpixel: Superpixel image with superpixel indexes
    @return neighbors: Sparse matrix of superpixel neighbors
    
    '''
    offset = 1
    neighbors = np.full((superpixel.max()+1,superpixel.max()+1),False)
    for i in range(1,np.shape(superpixel)[0]-2):
        for j in range(1,np.shape(superpixel)[1]-2):
            temp = superpixel[i-offset:i+offset+1, j-offset:j+offset+1].flatten()
            for k in temp:
                if k != superpixel[i,j]:
                    neighbors[superpixel[i,j]][k] = True
                    neighbors[k][superpixel[i,j]] = True
    print('Neighbors computed')
    return neighbors    


def compareSuperpixel(superpixelColor, target, tempColor, maxDist):
    '''
    Compare two superpixel colors by computing the distance in a CIELab space.
    
    @param superpixelColor: List of colors for each superpixel
    @param target: Reference superpixel index to be compared
    @param tempColor: Reference superpixel color to be compared
    @param maxDist: Maximum acceptable distance
    @return True if the computed distance is under the limit, False otherwise
    
    '''
           
    #L0 = tempColor[0,0]
    L0 = np.median(tempColor[0,0])
    L1 = superpixelColor[target][0]
    
    #a0 = tempColor[0,1]  #np.mean(tempColor[:,1])
    a0 = np.median(tempColor[:,1])
    a1 = superpixelColor[target][1]
    
    #b0 = tempColor[0,2] #np.mean(tempColor[:,2])
    b0 = np.median(tempColor[:,2])
    b1 = superpixelColor[target][2]
    
    dist = math.sqrt( math.pow((L1-L0),2) + math.pow((a1-a0),2) + math.pow((b1-b0),2))

    print(dist)
    if dist < maxDist:
        return True
    else:
        return False
    
    
'''   
def growingSuperpixel(superpixel, superpixelColor, image, mask, neighbors, seed, target, tempColor, classe, maxDist, canvas = False):
    
    #Given a target superpixel index the neighbors are analysed if within the max color distance.
    #Recursive depth search version. (not used)

    if visited[target] == True:
        #print('exit visited ' + str(target))
        return False
    if superpixelClass[target] > 0:
        #print('exit marked')
        return False
        
    if compareSuperpixel(superpixelColor, target, tempColor, sigmaH, sigmaS, sigmaV) == True:
        #print('Comparing ' + str(target) + ' ' + str(seed))
        superpixelClass[target] = classe
        visited[target] = True
        tempColor = np.insert(tempColor, 0, superpixelColor[target], 0)
        tempColor = np.reshape(tempColor, (int(np.size(tempColor)/3),3))
        if classe == 1: color = [255,0,0]
        if classe == 2: color = [0,255,0]
        if classe == 3: color = [0,0,255]
            
        mask = paintSuperpixel(superpixel, mask, image, target, color)
        if canvas:
            temp = updateCanvas(mask, hor, ver)
    else:
        visited[target] = True
        #print('exit outside threshold ' + str(target) + ' ' + str(seed))
        return False
    
    temp = returnNeighbors(target, neighbors)
    
    for k in temp:
        growingSuperpixel(superpixel, superpixelColor, image, mask, neighbors, seed, k, tempColor, classe, maxDist, canvas)
    return
'''


def growingSuperpixelBreadth(superpixel, superpixelColor, image, mask, neighbors, seed, target, tempColor, classe, maxDist, canvas = False): 
    '''
    Given a target superpixel index the neighbors are analysed if within the max color distance.
    Breadth search version.
    
    @param superpixel: 
    @param superpixelColor:
    @param image:
    @param mask:
    @param neighbors:
    @param seed:
    @param target:
    @param tempColor:
    @param classe:
    @param maxDist:
    @param canvas:
    '''
    
    queue = []
    queue.append(target)
    #visited[target] = True
    
    
    if classe == 1: color = [255,0,0]
    if classe == 2: color = [0,255,0]
    if classe == 3: color = [0,0,255]
    
    
    mask = paintSuperpixel(superpixel, mask, image, target, color)
    
    while queue:
        s = queue.pop(0)
        print("Pop " + str(s))
        temp = returnNeighbors(s, neighbors)
        for k in temp:
            #if visited[s] == False:
            if True:
                if compareSuperpixel(superpixelColor, k, tempColor, maxDist) == True:
                    if visited[k] == False or superpixelClass[k] == 0:
                        queue.append(k)
                        print("Push " + str(k))
                        tempColor = np.insert(tempColor, 0, superpixelColor[k], 0)
                        tempColor = np.reshape(tempColor, (int(np.size(tempColor)/3),3))
                        mask = paintSuperpixel(superpixel, mask, image, k, color)
                        if canvas:
                            temp = updateCanvas(mask, hor, ver)
                        superpixelClass[k] = classe
            visited[k] = True
        print(np.size(queue))
        if np.size(queue) > 50:
            print(queue)
            queue = []

        
def showImage (image):
    '''
    Show a given image in the system's image visualiser
    
    @param image: Image to be shown    
    '''
    pil_img = Image.fromarray(image)
    pil_img.show()
    
  
    

while True:
    event, values = window.Read()
    
    try:
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
    except Exception as e:
        print(e)
        
    
    if event is None or event == 'Exit':
        break  
    elif event == 'Open image':
        #hor = int(values["_Sl_horizontal_"])
        #ver = int(values["_Sl_vertical_"])
        
        address = sg.PopupGetFile('Document to open')
        #address = 'D:/Ademir/Coding/datasets/image1/0055.png'
        
        
        try: 
            file = open(address, 'rb').read()
            try:
                image = cv2.imdecode(np.frombuffer(file, np.uint8), 1) # 0 in the case of grayscale images
                #image = cv2.bilateralFilter(image, 10, 75, 75)
                mask = np.copy(image)
                original = np.copy(image)
                temp = updateCanvas(image, hor, ver)
                #window.Element("_Bt_Smooth_").Update(disabled=False)

            except ValueError:
                sg.Popup(ValueError)
        except Exception as e:
            sg.Popup(e)
            
            
    elif event == 'Open Superpixel':
        #hor = int(values["_Sl_horizontal_"])
        #ver = int(values["_Sl_vertical_"])
        
        
        address = sg.PopupGetFile('Document to open')
        #address = 'D:/Ademir/Coding/datasets/image1-superpixels2000/0055.png'
        
        try:
            superpixel = Image.open(address)
            superpixel = np.asarray(superpixel)
            #superpixel = superpixel.transpose()
        except Exception as e:
            sg.Popup(e)
            
    elif event == 'Generate Superpixels':
        
        sp_layout = [  [sg.Text('Some text on Row 1')],
            [sg.Text('Quantity:'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel')] ]
        
        sp_window = sg.Window('Slic Superpixels', sp_layout)
        
        while True:
            sp_event, sp_values = sp_window.read()
            if sp_event in (None, 'Cancel'):   # if user closes window or clicks cancel
                break
            if sp_event == 'Ok':
                try:
                    del(neighbors)
                except:
                    print('')
                try:
                    del(superpixelColors)
                except:
                    print('')
                try:
                    del(superpixelClass)
                except:
                    print('')
                try:
                    del(visited)
                except:
                    print('')
                break        
        sp_window.close()
        
        try:
            superpixel = slic(image, n_segments=int(sp_values[0]), compactness=40)
        except Exception as e:
            print(e)
            
    elif event == 'Load CSV file':
        
        address = sg.PopupGetFile('Document to open')
        
        if address != None:
            neighbors = setNeighbors(superpixel)
            labImage = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            superpixelColor = computeSuperpixelColor(labImage, superpixel)
            superpixelClass = np.zeros((superpixel.max()+1))
            visited = np.full((superpixel.max()+1), False)
            
            
            superpixelClass = np.zeros((superpixel.max()+1))
            visited = np.full((superpixel.max()+1), False)
            mask = np.copy(image)
            with open(address) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';', )
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        #image[1,2] = red
                        #image[int(row[1]),int(row[0])] = red
                        #image = cv2.circle(image, (int(row[1])-1,int(row[0])-1), 5, red, -1)
                        target = superpixel[int(row[0])-1,int(row[1])-1]
                        tempColor = superpixelColor[target]
                        tempColor = np.reshape(tempColor, (int(np.size(tempColor)/3),3))
                        #growingSuperpixelBreadth(superpixel, superpixelColor, image, mask, neighbors, target_0, target_0, tempColor, 1, 19, False)
                        growingSuperpixelBreadth(superpixel, superpixelColor, image, mask, neighbors, target, target, tempColor, classe, maxDist, True)
                        #pontos.append([int(row[1]),int(row[0])])
                        #image = cv2.circle(image, (500,500), 10, red, 3)
                        line_count = line_count + 1
                    line_count = line_count + 1        
        
        
            
    elif event == '_Sl_horizontal_' or event == '_Sl_vertical_':
        #hor = int(values["_Sl_horizontal_"])
        #ver = int(values["_Sl_vertical_"])
        temp = updateCanvas(temp, hor, ver)
        
    elif event == 'Class 1':
        active_class = 'class1'
        temp = updateCanvas(mask, hor, ver)
        window.Element("_Tx_selected_").Update(value='Class 1 selected')
    elif event == 'Class 2':
        active_class = 'class2'
        temp = updateCanvas(mask, hor, ver)
        window.Element("_Tx_selected_").Update(value='Class 2 selected')
    elif event == 'Class 3':
        active_class = 'class3'
        temp = updateCanvas(mask, hor, ver)
        window.Element("_Tx_selected_").Update(value='Class 3 selected')
        
    elif event == 'Original':
        try:
            temp = updateCanvas(original, hor, ver)
        except Exception as e:
            print(e)
    elif event == 'Extrapolate':
        
        try:
            maxDist = float(values["_In_sigmaH_"])
        except Exception as e:
            print(e)
            
        
        
        visited = np.full((superpixel.max()+1), False)
        
        try:
            np.shape(neighbors)
            np.shape(superpixelColor)
        except:
            neighbors = setNeighbors(superpixel)
            labImage = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            superpixelColor = computeSuperpixelColor(labImage, superpixel)
            superpixelClass = np.zeros((superpixel.max()+1))
            visited = np.full((superpixel.max()+1), False)

          
        try:
            tempColor = superpixelColor[superpixel[y,x]]
            tempColor = np.reshape(tempColor, (int(np.size(tempColor)/3),3))
            print("Origin "+ str(tempColor[:,2][0]))
            #growingSuperpixel(superpixel, superpixelColor, image, mask, neighbors, superpixel[x,y], superpixel[x,y], tempColor, classe, maxDist, True)
            growingSuperpixelBreadth(superpixel, superpixelColor, image, mask, neighbors, superpixel[y,x], superpixel[y,x], tempColor, classe, maxDist, True)
        except Exception as e:
            print(e)
        

    elif event == 'Fill':
        if active_class == 'class1':
            for i in range(0,np.shape(image)[0]-1):
                for j in range(0,np.shape(image)[1]-1):
                    if (mask[i,j][1] !=  255 and mask[i,j][1] !=  0 and mask[i,j][2] !=  0):
                        if (mask[i,j][1] !=  0 and mask[i,j][1] !=  255 and mask[i,j][2] !=  0):
                            if (mask[i,j][1] !=  0 and mask[i,j][1] !=  0 and mask[i,j][2] !=  255):
                                mask[i,j] = [255,0,0]

        temp = updateCanvas(mask, hor, ver)
        if active_class == 'class2':
            for i in range(0,np.shape(image)[0]-1):
                for j in range(0,np.shape(image)[1]-1):
                    if (mask[i,j][1] !=  255 and mask[i,j][1] !=  0 and mask[i,j][2] !=  0):
                        if (mask[i,j][1] !=  0 and mask[i,j][1] !=  255 and mask[i,j][2] !=  0):
                            if (mask[i,j][1] !=  0 and mask[i,j][1] !=  0 and mask[i,j][2] !=  255):
                                mask[i,j] = [0,255,0]

        temp = updateCanvas(mask, hor, ver)
        if active_class == 'class3':
            for i in range(0,np.shape(image)[0]-1):
                for j in range(0,np.shape(image)[1]-1):
                    if (mask[i,j][1] !=  255 and mask[i,j][1] !=  0 and mask[i,j][2] !=  0):
                        if (mask[i,j][1] !=  0 and mask[i,j][1] !=  255 and mask[i,j][2] !=  0):
                            if (mask[i,j][1] !=  0 and mask[i,j][1] !=  0 and mask[i,j][2] !=  255):
                                mask[i,j] = [0,0,255]

        temp = updateCanvas(mask, hor, ver)
        

    elif event == 'Reset':
        superpixelClass = np.zeros((superpixel.max()+1))
        visited = np.full((superpixel.max()+1), False)
        
        if active_class == 'class1':
            for i in range(0,np.shape(image)[0]-1):
                for j in range(0,np.shape(image)[1]-1):
                    if (mask[i,j][0] ==  255 and mask[i,j][1] ==  0 and mask[i,j][2] ==  0):
                        mask[i,j] = original[i,j]
                        
        if active_class == 'class2':
            for i in range(0,np.shape(image)[0]-1):
                for j in range(0,np.shape(image)[1]-1):
                    if (mask[i,j][0] ==  0 and mask[i,j][1] ==  255 and mask[i,j][2] ==  0):
                        mask[i,j] = original[i,j]
                        
        if active_class == 'class3':
            for i in range(0,np.shape(image)[0]-1):
                for j in range(0,np.shape(image)[1]-1):
                    if (mask[i,j][0] ==  0 and mask[i,j][1] ==  0 and mask[i,j][2] ==  255):
                        mask[i,j] = original[i,j]
                        
        temp = updateCanvas(mask, hor, ver)
        
        
    elif event == '_Canvas1_':
        position = values['_Canvas1_']
        #hor = int(values["_Sl_horizontal_"])
        #ver = int(values["_Sl_vertical_"])
        try:
            size = canvas1.CanvasSize
            positionX = int(np.shape(image)[1]/100*hor)
            positionY = int(np.shape(image)[0]/100*ver)
            if positionX > np.shape(image)[1]-size[1]: positionX = np.shape(image)[1]-size[1]-1
            if positionY > (np.shape(image)[0]-size[0]): positionY = np.shape(image)[0]-size[0]-1
            x = positionX+position[0]
            y = positionY+abs(position[1]-size[1])
            print(x,y)
            
            
            try:
                if active_class == 'class1':
                    classe = 1
                    color = [255,0,0]

                if active_class == 'class2':
                    classe = 2
                    color = [0,255,0]
                    

                if active_class == 'class3':
                    classe = 3
                    color = [0,0,255]
                    
                mask = paintSuperpixel(superpixel, mask, image, superpixel[y,x], color)
                temp = updateCanvas(mask, hor, ver)
            except Exception as e:
                print(e)
                                
        except Exception as e:
            print(e)
            
    print(event, values)
     
window.Close()


'''
################################################################################################


new = np.zeros(np.shape(image))

for i in range(1,np.shape(superpixel)[0]):
    for j in range(1,np.shape(superpixel)[1]):
        new[i,j] = superpixelColor[superpixel[i,j]]
        
showImage(cv2.cvtColor(np.uint8(new), cv2.COLOR_Lab2RGB))


showImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
showImage(superpixel/10)
##############################################################################################################


seed = 1921
target = seed
classe = 1
superpixelClass = np.zeros((superpixel.max()+1))
visited = np.full((superpixel.max()+1), False)
mask = np.copy(image)

    
neighbors = setNeighbors(superpixel)
colors0 = superpixelColor
labImage = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
superpixelColor = computeSuperpixelColor(labImage, superpixel)


tempColor = superpixelColor[seed]
tempColor = np.reshape(tempColor, (int(np.size(tempColor)/3),3))
growingSuperpixelBreadth(superpixel, superpixelColor, image, mask, neighbors, seed, target, tempColor, 1, 19, False)
            



##############################################################################################################

import csv

imageNum = '0055'

address = 'D:/Ademir/Coding/datasets/image1/'+ imageNum +'.png'
try: 
    file = open(address, 'rb').read()
    image = cv2.imdecode(np.frombuffer(file, np.uint8), 1) # 0 in the case of grayscale images
    mask = np.copy(image)
except Exception as e:
    sg.Popup(e)  
    
#image = cv2.bilateralFilter(image, 10, 75, 75)
#image = cv2.GaussianBlur(image,(7,7),1)
    
address = 'D:/Ademir/Coding/datasets/image1-superpixels2000/'+ imageNum +'.png' #0056
try:
    superpixel = Image.open(address)
    superpixel = np.asarray(superpixel)
    #superpixel = superpixel.transpose()
except Exception as e:
    sg.Popup(e)

neighbors = setNeighbors(superpixel)
labImage = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
superpixelColor = computeSuperpixelColor(labImage, superpixel)


superpixelClass = np.zeros((superpixel.max()+1))
visited = np.full((superpixel.max()+1), False)
mask = np.copy(image)
with open('D:/Ademir/Coding/datasets/Eniuce/image1points26tiles/PointsInsideFracturesArapuaPrio4'+ imageNum + '.png.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';', )
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            #image[1,2] = red
            #image[int(row[1]),int(row[0])] = red
            #image = cv2.circle(image, (int(row[1])-1,int(row[0])-1), 5, red, -1)
            target_0 = superpixel[int(row[0])-1,int(row[1])-1]
            tempColor = superpixelColor[target_0]
            tempColor = np.reshape(tempColor, (int(np.size(tempColor)/3),3))
            growingSuperpixelBreadth(superpixel, superpixelColor, image, mask, neighbors, target_0, target_0, tempColor, 1, 19, False)
            #pontos.append([int(row[1]),int(row[0])])
            #image = cv2.circle(image, (500,500), 10, red, 3)
            line_count = line_count + 1
        line_count = line_count + 1
showImage(mask)


image_points = np.copy(image)
red = [255,0,0]
with open('D:/Ademir/Coding/datasets/Eniuce/image1points26tiles/PointsInsideFracturesArapuaPrio4'+imageNum+'.png.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';', )
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            image_points = cv2.circle(image_points, (int(row[1])-1,int(row[0])-1), 2, red, 1)
            line_count = line_count + 1
        line_count = line_count + 1
showImage(image_points)
'''
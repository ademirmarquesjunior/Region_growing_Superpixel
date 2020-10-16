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
from numba import jit


#  View functions and definitions
SCREEN_SIZE = 500

menu_def = [['File', ['Open image', 'Open Superpixel', 'Generate Superpixels',
                      'Load CSV file', 'Save mask', 'Exit']],
            ['Help', ['About...', 'Reload']]]


layout = [[sg.Menu(menu_def, tearoff=True)],
          [sg.Button("process", size=(6, 2), key="process"),
           sg.Button("drag", size=(4, 2), key="drag"),
           sg.Button("brush", size=(4, 2), key="brush"),
           sg.Button("grow", size=(4, 2), key="grow"),
           sg.Button("path", size=(4, 2), key="path")
           ],
          [sg.Graph(canvas_size=(SCREEN_SIZE, SCREEN_SIZE), graph_bottom_left=(0, 0),
                    enable_events=True, drag_submits=True,
                    background_color='black',
                    graph_top_right=(SCREEN_SIZE, SCREEN_SIZE), key="_Canvas1_"),
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
                 sg.Button('Expand'), sg.Button('Fill'),
                 sg.Button('Clean')], ])],
                ])],
          [sg.Slider(range=(0, 100), orientation='h', size=(40, 10),
                     enable_events=True, disable_number_display=True,
                     default_value=0, key="_Sl_horizontal_")], ]

window = sg.Window('Superpixel Growing Segmentation', layout)
canvas1 = window.Element("_Canvas1_")
temp = np.zeros(canvas1.CanvasSize)

# Global values initialization

y = 0
x = 0
maxDist = 19
active_class = 'class1'
classe = 1
mode = 0  # 0: drag; 1: point; 2: grow; 3: path

click = -1
first_click = []
second_click = []

tempColor = []

# View functions


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

    # if np.size(np.shape(image)) == 3:
    #     image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB)

    size = canvas1.CanvasSize
    positionX = int(np.shape(image)[1]/100*hor)
    positionY = int(np.shape(image)[0]/100*ver)

    if positionX > np.shape(image)[1]-size[1]:
        positionX = np.shape(image)[1]-size[1]-1
    if positionY > (np.shape(image)[0]-size[0]):
        positionY = np.shape(image)[0]-size[0]-1

    try:
        # Create buffer
        buffered = BytesIO()

        # Save image to buffer
        Image.fromarray(np.uint8(image[positionY:size[0]+positionY,
                                       positionX:size[1]+positionX])
                        ).save(buffered, format="PNG")

        # Encode buffer to binary
        encoded = base64.b64encode(buffered.getvalue())

        # Load encoded image buffer to canvas
        canvas1.DrawImage(data=encoded, location=(0, SCREEN_SIZE))
        canvas1.Update()
    except Exception as e:
        print("Update canvas " + str(e))
    return image

@jit
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

    target[:, :, 0] = np.where((superpixel == index), color[0],
                               target[:, :, 0])
    target[:, :, 1] = np.where((superpixel == index), color[1],
                               target[:, :, 1])
    target[:, :, 2] = np.where((superpixel == index), color[2],
                               target[:, :, 2])

    return target


@jit(nopython=True)
def computeSuperpixelColor(labImage, superpixel):
    '''
    Compute the centroid color in a CIELab space for each superpixel.

    Parameters
    ----------
    image : TYPE
        The reference image in CIELab colors.
    superpixel : TYPE
        The superpixel image with the indexes for each pixel.

    Returns
    -------
    superpixelColor : TYPE
        Array of CieLAB colors for each Superpixel.

    '''

    superpixel_f = superpixel.flatten()
    L_values = labImage[:, :, 0].flatten()
    a_values = labImage[:, :, 1].flatten()
    b_values = labImage[:, :, 2].flatten()

    superpixel_max = superpixel.max()+1

    superpixelColor = [(0, 0, 0)]

    for i in range(1, superpixel_max):
        L = np.nanmedian(np.where((superpixel_f == i), L_values, np.nan))
        a = np.nanmedian(np.where((superpixel_f == i), a_values, np.nan))
        b = np.nanmedian(np.where((superpixel_f == i), b_values, np.nan))

        superpixelColor.append((L, a, b))

    return superpixelColor


def returnNeighbors(index, neighbors):
    '''
    Returns the superpixel neighbors of a given superpixel.

    Parameters
    ----------
    index : TYPE
        Reference superpixel index.
    neighbors : TYPE
        Sparse matrix of superpixel neighbors.

    Returns
    -------
    TYPE
        List of neighbors.

    '''

    temp = neighbors[index:index+1, :]
    return np.unique(np.sort(np.where(temp == True)[1]))


def setNeighbors(superpixel):
    '''
    Computes neighbors of each superpixel by looking the indexes in each pixel
    of superpixel image.

    Parameters
    ----------
    superpixel : TYPE
        Superpixel array with annotated Superpixel for each pixel.

    Returns
    -------
    neighbors : TYPE
        Sparse matrix of superpixel neighbors.

    '''

    offset = 1
    neighbors = np.full((superpixel.max()+1, superpixel.max()+1), False)
    for i in range(1, np.shape(superpixel)[0]-2):
        for j in range(1, np.shape(superpixel)[1]-2):
            temp = superpixel[i-offset:i+offset+1,
                              j-offset:j+offset+1].flatten()
            for k in temp:
                if k != superpixel[i, j]:
                    neighbors[superpixel[i, j]][k] = True
                    neighbors[k][superpixel[i, j]] = True
    print('Neighbors computed')
    return neighbors


def compute_color_distance(color_0, color_1):

    L0, a0, b0 = color_0
    L1, a1, b1 = color_1

    dist = math.sqrt(math.pow((L1-L0), 2) + math.pow((a1-a0), 2)
                     + math.pow((b1-b0), 2))

    return dist




def compareSuperpixel(superpixelColor, target, tempColor, maxDist):
    '''
    Compare two superpixel colors by computing the distance in a CIELab space.

    Parameters
    ----------
    superpixelColor : TYPE
        List of colors for each superpixel.
    target : TYPE
        Reference superpixel index to be compared.
    tempColor : TYPE
        Reference superpixel index to be compared.
    maxDist : TYPE
        Maximum acceptable distance in the CIELab space.

    Returns
    -------
    bool
        True if the computed distance is under maxDist value. False otherwise.

    '''

    # L0 = tempColor[0, 0]
    L0 = np.median(tempColor[:, 0])
    L1 = superpixelColor[target][0]

    # a0 = tempColor[0,1]  #np.mean(tempColor[:,1])
    a0 = np.median(tempColor[:, 1])
    a1 = superpixelColor[target][1]

    # b0 = tempColor[0,2] #np.mean(tempColor[:,2])
    b0 = np.median(tempColor[:, 2])
    b1 = superpixelColor[target][2]

    dist = compute_color_distance((L0, a0, b0), (L1, a1, b1))

    if dist < maxDist:
        return True
    else:
        return False


def growingSuperpixelBreadth(superpixel, superpixelColor, image, mask,
                             neighbors, seed, target, tempColor, classe,
                             maxDist, canvas=False):
    '''
    Given a target superpixel index the neighbors are analysed if within the
    max color distance. Uses breadth search.

    Parameters
    ----------
    superpixel : TYPE
        Superpixel array with annotated Superpixel for each pixel.
    superpixelColor : TYPE
        List of colors for each superpixel.
    image : TYPE
        RGB image.
    mask : TYPE
        RGB mask image to be drawn.
    neighbors : TYPE
        DESCRIPTION.
    seed : TYPE
        Initial target.
    target : TYPE
        Initial target.
    tempColor : TYPE
        Initial reference color.
    classe : TYPE
        Which class to be painted.
    maxDist : TYPE
        Maximum acceptable distance in the CIELab space.
    canvas : TYPE, optional
        If True forces canvas update. The default is False.

    Returns
    -------
    None.

    '''

    queue = []
    queue.append(target)
    # visited[target] = True

    if classe == 1:
        color = [255, 0, 0]
    if classe == 2:
        color = [0, 255, 0]
    if classe == 3:
        color = [0, 0, 255]

    mask = paintSuperpixel(superpixel, mask, image, target, color)

    while queue:  # queue
        s = queue.pop(0)
        # print("Pop " + str(s))
        temp = returnNeighbors(s, neighbors)
        for k in temp:
            # if visited[s] == False:
            if True:
                if compareSuperpixel(superpixelColor, k, tempColor,
                                     maxDist) is True:
                    if visited[k] is False or superpixelClass[k] == 0:
                        queue.append(k)
                        # print("Push " + str(k))
                        tempColor = np.insert(tempColor, 0,
                                              superpixelColor[k], 0)
                        tempColor = np.reshape(tempColor,
                                               (int(np.size(tempColor)/3), 3))
                        mask = paintSuperpixel(superpixel, mask, image, k,
                                               color)
                        if canvas:
                            temp = updateCanvas(mask, hor, ver)
                        superpixelClass[k] = classe
            visited[k] = True
        # print(np.size(queue))
        if np.size(queue) > 50:
            # print(queue)
            queue = []
    return


def least_path(neighbors, start, end, superpixelColor):

    visited = []
    unvisited = list(range(np.shape(neighbors)[0]))

    path_cost = np.full((np.size(unvisited), 2), np.inf)

    # start = 0
    path_cost[start, 0] = 0

    while np.size(unvisited) != 0:

        current_vertex = unvisited[np.argmin(path_cost[unvisited, 0])]

        if current_vertex == end:
            break

        unvisited_neighbour = np.intersect1d(
            returnNeighbors(current_vertex, neighbors), unvisited)

        for i in unvisited_neighbour:
            dist = compute_color_distance(superpixelColor[current_vertex],
                                          superpixelColor[i])
            if dist + path_cost[current_vertex, 0] < path_cost[i, 0]:
                path_cost[i] = (dist +
                                path_cost[current_vertex, 0]), current_vertex

        visited.append(current_vertex)
        unvisited.remove(current_vertex)

    # backtrack
    backtrack = []
    backtrack.append(current_vertex)
    while current_vertex != start:

        current_vertex = np.uint(path_cost[current_vertex][1])
        backtrack.append(current_vertex)

    return backtrack


def two_points_drawing(mask, neighbors, start, end, superpixelColor):
    if classe == 1:
        color = [255, 0, 0]
    if classe == 2:
        color = [0, 255, 0]
    if classe == 3:
        color = [0, 0, 255]

    path = least_path(neighbors, start, end, superpixelColor)

    for target in path:
        mask = paintSuperpixel(superpixel, mask, image, target, color)

    return mask


def showImage(image):
    '''
    Open a image with the OS image viewer

    Parameters
    ----------
    image : TYPE
        Numpy array of a RGB or grayscale image.

    Returns
    -------
    None.

    '''

    pil_img = Image.fromarray(image)
    pil_img.show()

    return

# Controler section


while True:
    event, values = window.Read()

    try:
        hor = int(values["_Sl_horizontal_"])
        ver = int(values["_Sl_vertical_"])
    except Exception as e:
        print(e)

    if event is None or event == 'Exit':
        break
    elif event == 'Open image':  # -------------------------------------------

        address = sg.PopupGetFile('Document to open')
        # address = 'D:/Ademir/Coding/datasets/image1/0055.png'

        try:
            file = open(address, 'rb').read()
            try:
                image = cv2.imdecode(np.frombuffer(file, np.uint8), 1)
                # 0 in the case of grayscale images

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # image = cv2.bilateralFilter(image, 10, 75, 75)
                mask = np.copy(image)
                original = np.copy(image)
                temp = updateCanvas(image, hor, ver)
                # window.Element("_Bt_Smooth_").Update(disabled=False)

            except ValueError:
                sg.Popup(ValueError)
        except Exception as e:
            sg.Popup(e)

    elif event == 'Open Superpixel':  # --------------------------------------

        address = sg.PopupGetFile('Document to open')

        # address = 'D:/Ademir/Coding/datasets/image1-superpixels2000/0055.png'

        try:
            superpixel = Image.open(address)
            superpixel = np.asarray(superpixel)
            # superpixel = superpixel.transpose()
        except Exception as e:
            sg.Popup(e)

    elif event == 'Generate Superpixels':  # ---------------------------------

        # View popup to receive Superpixel parameters
        sp_layout = [[sg.Text('Insert the number of superpixel desired.')],
                     [sg.Text('Quantity:'), sg.InputText()],
                     [sg.Button('Ok'), sg.Button('Cancel')]]

        sp_window = sg.Window('Slic Superpixels', sp_layout)

        while True:
            sp_event, sp_values = sp_window.read()
            if sp_event in (None, 'Cancel'):
                break
            if sp_event == 'Ok':
                try:
                    del(neighbors)
                    del(superpixelColors)
                    del(superpixelClass)
                    del(visited)
                except Exception:
                    print('No data to clean')
                break
        sp_window.close()

        try:
            superpixel = slic(image, n_segments=int(sp_values[0]),
                              compactness=40)
        except Exception as e:
            print(e)

    elif event == 'drag':
        mode = 0
    elif event == 'brush':
        mode = 1
    elif event == 'grow':
        mode = 2
    elif event == 'path':
        print("path mode")
        mode = 3

    elif event == 'Load CSV file':  # ----------------------------------------

        try:
            maxDist = float(values["_In_sigmaH_"])
        except Exception as e:
            print(e)

        address = sg.PopupGetFile('Document to open')

        if address is not None:
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
                seed_image = np.copy(image)
                for row in csv_reader:
                    if line_count != 0:
                        # image[1,2] = red
                        # image[int(row[1])-1, int(row[0])-1] = (255, 0, 0)
                        seed_image = cv2.circle(seed_image,
                                                (int(row[1])-1,
                                                 int(row[0])-1), 5,
                                                (255, 0, 0), 0)
                        target = superpixel[int(row[0])-1, int(row[1])-1]
                        tempColor = superpixelColor[target]
                        tempColor = np.reshape(tempColor, (int(np.size(
                                                           tempColor)/3), 3))
                        growingSuperpixelBreadth(superpixel, superpixelColor,
                                                 image, mask, neighbors,
                                                 target, target, tempColor,
                                                 classe, maxDist, False)
                        temp = updateCanvas(mask, hor, ver)
                        # pontos.append([int(row[1]),int(row[0])])
                        # image = cv2.circle(image, (SCREEN_SIZE,SCREEN_SIZE), 10, red, 3)
                        line_count = line_count + 1
                    line_count = line_count + 1
                cv2.imwrite(address+".png",
                            cv2.cvtColor(seed_image, cv2.cv2.COLOR_BGR2RGB))

    elif event == 'Save mask':  # --------------------------------------------

        save_layout = [[
            sg.InputText(visible=True, enable_events=True, key='fig_path',
                         size=(10, 50)),
            sg.FileSaveAs(
                key='fig_save',
                file_types=(('PNG', '.png'), ('JPG', '.jpg')),
            ), sg.OK()
        ]]
        save_window = sg.Window('Demo Application', save_layout, finalize=True)

        while True:  # Event Loop
            save_event, save_values = save_window.Read()
            print(save_event)
            if (save_event == 'fig_path') and (save_values['fig_path'] != ''):
                save_address = save_values['fig_path']
                save_window.Close()
            if save_event is None or save_event == "OK":
                break

        save_window.Close()

        if save_address != None:
            try:
                cv2.imwrite(save_address, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            except Exception:
                print("Impossible to save")

    elif event == '_Sl_horizontal_' or event == '_Sl_vertical_':

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

    elif event == 'Expand':  # ------------------------------------------

        try:
            maxDist = float(values["_In_sigmaH_"])
        except Exception as e:
            print(e)

        visited = np.full((superpixel.max()+1), False)

        try:
            np.shape(neighbors)
            np.shape(superpixelColor)
        except Exception:
            neighbors = setNeighbors(superpixel)
            labImage = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            superpixelColor = computeSuperpixelColor(labImage, superpixel)
            superpixelClass = np.zeros((superpixel.max()+1))
            visited = np.full((superpixel.max()+1), False)

        try:
            tempColor = superpixelColor[superpixel[y, x]]
            tempColor = np.reshape(tempColor, (int(np.size(tempColor)/3), 3))
            print("Origin " + str(tempColor[:, 2][0]))

            growingSuperpixelBreadth(superpixel, superpixelColor, image, mask,
                                     neighbors, superpixel[y, x],
                                     superpixel[y, x], tempColor, classe,
                                     maxDist, False)
            temp = updateCanvas(mask, hor, ver)
        except Exception as e:
            print(e)

    elif event == 'Fill':  # -------------------------------------------------

        if active_class == 'class1':
            for i in range(0, np.shape(image)[0]-1):
                for j in range(0, np.shape(image)[1]-1):
                    if (mask[i, j][1] != 255 and mask[i, j][1] != 0
                            and mask[i, j][2] != 0):

                        if (mask[i, j][1] != 0 and mask[i, j][1] != 255
                                and mask[i, j][2] != 0):

                            if (mask[i, j][1] != 0 and mask[i, j][1] != 0
                                    and mask[i, j][2] != 255):
                                mask[i, j] = [255, 0, 0]

        temp = updateCanvas(mask, hor, ver)
        if active_class == 'class2':
            for i in range(0, np.shape(image)[0]-1):
                for j in range(0, np.shape(image)[1]-1):
                    if (mask[i, j][1] != 255 and mask[i, j][1] != 0
                            and mask[i, j][2] != 0):

                        if (mask[i, j][1] != 0 and mask[i, j][1] != 255
                                and mask[i, j][2] != 0):

                            if (mask[i, j][1] != 0 and mask[i, j][1] != 0
                                    and mask[i, j][2] != 255):
                                mask[i, j] = [0, 255, 0]

        temp = updateCanvas(mask, hor, ver)
        if active_class == 'class3':
            for i in range(0, np.shape(image)[0]-1):
                for j in range(0, np.shape(image)[1]-1):
                    if (mask[i, j][1] != 255 and mask[i, j][1] != 0
                            and mask[i, j][2] != 0):

                        if (mask[i, j][1] != 0 and mask[i, j][1] != 255
                                and mask[i, j][2] != 0):

                            if (mask[i, j][1] != 0 and mask[i, j][1] != 0
                                    and mask[i, j][2] != 255):
                                mask[i, j] = [0, 0, 255]

        temp = updateCanvas(mask, hor, ver)

    elif event == 'Clean':  # ------------------------------------------------
        superpixelClass = np.zeros((superpixel.max()+1))
        visited = np.full((superpixel.max()+1), False)

        if active_class == 'class1':
            for i in range(0, np.shape(image)[0]-1):
                for j in range(0, np.shape(image)[1]-1):
                    if (mask[i, j][0] == 255 and mask[i, j][1] == 0
                            and mask[i, j][2] == 0):
                        mask[i, j] = original[i, j]

        if active_class == 'class2':
            for i in range(0, np.shape(image)[0]-1):
                for j in range(0, np.shape(image)[1]-1):
                    if (mask[i, j][0] == 0 and mask[i, j][1] == 255
                            and mask[i, j][2] == 0):
                        mask[i, j] = original[i, j]

        if active_class == 'class3':
            for i in range(0, np.shape(image)[0]-1):
                for j in range(0, np.shape(image)[1]-1):
                    if (mask[i, j][0] == 0 and mask[i, j][1] == 0
                            and mask[i, j][2] == 255):
                        mask[i, j] = original[i, j]

        temp = updateCanvas(mask, hor, ver)

    elif event == 'process':
        try:
            neighbors = setNeighbors(superpixel)
            labImage = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            superpixelColor = computeSuperpixelColor(labImage, superpixel)
            superpixelClass = np.zeros((superpixel.max()+1))
            visited = np.full((superpixel.max()+1), False)
            mask = np.copy(image)
        except Exception as e:
            print(e)

    elif event == '_Canvas1_':  # --------------------------------------------
        position = values['_Canvas1_']

        try:
            size = canvas1.CanvasSize
            positionX = int(np.shape(image)[1]/100*hor)
            positionY = int(np.shape(image)[0]/100*ver)

            if positionX > np.shape(image)[1]-size[1]:
                positionX = np.shape(image)[1]-size[1]-1
            if positionY > (np.shape(image)[0]-size[0]):
                positionY = np.shape(image)[0]-size[0]-1

            x = positionX+position[0]
            y = positionY+abs(position[1]-size[1])
            print(x, y)

            try:
                if active_class == 'class1':
                    classe = 1
                    color = [255, 0, 0]

                if active_class == 'class2':
                    classe = 2
                    color = [0, 255, 0]

                if active_class == 'class3':
                    classe = 3
                    color = [0, 0, 255]

                if mode == 1:
                    mask = paintSuperpixel(superpixel, mask, image,
                                           superpixel[y, x], color)
                    temp = updateCanvas(mask, hor, ver)
                if mode == 2:
                    try:
                        tempColor = superpixelColor[superpixel[y, x]]
                        tempColor = np.reshape(tempColor, (
                            int(np.size(tempColor)/3), 3))
                        print("Origin " + str(tempColor[:, 2][0]))

                        growingSuperpixelBreadth(superpixel, superpixelColor,
                                                 image, mask, neighbors,
                                                 superpixel[y, x],
                                                 superpixel[y, x], tempColor,
                                                 classe, maxDist, False)
                        temp = updateCanvas(mask, hor, ver)
                    except Exception as e:
                        print(e)

            except Exception as e:
                print(e)

        except Exception as e:
            print(e)
    elif event == "_Canvas1_+UP":  # -----------------------------------------
        try:
            size = canvas1.CanvasSize
            positionX = int(np.shape(image)[1]/100*hor)
            positionY = int(np.shape(image)[0]/100*ver)

            if positionX > np.shape(image)[1]-size[1]:
                positionX = np.shape(image)[1]-size[1]-1
            if positionY > (np.shape(image)[0]-size[0]):
                positionY = np.shape(image)[0]-size[0]-1

            x = positionX+position[0]
            y = positionY+abs(position[1]-size[1])
            print(x, y)

            if mode == 3:
                print("_______path")
                click += 1
                if click == 0:
                    first_click = [y, x]
                    mask = paintSuperpixel(superpixel, mask, image,
                                           superpixel[y, x], color)
                    temp = updateCanvas(mask, hor, ver)
                elif click == 1:
                    second_click = [y, x]
                    mask = two_points_drawing(mask, neighbors,
                                              superpixel[first_click[0],
                                                         first_click[1]],
                                              superpixel[second_click[0],
                                                         second_click[1]],
                                              superpixelColor)
                    temp = updateCanvas(mask, hor, ver)
                    click = -1
        except Exception as e:
            print(e)

    print(event, values)

window.Close()

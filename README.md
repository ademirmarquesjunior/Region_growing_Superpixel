# Region_growing_Superpixel
Technique to implement region growing segmentation in Superpixel data.

This program developed in Python (in prototype state) generates and loads superpixel data (grouped pixel regions) allowing faster segmentation to create masks of the desired features. It converts the superpixel data in a graph of interconnected objects to determine the superpixel neighborhood. The region growing algorithm uses color similarity to limit growing while navigating the superpixel graph. The RGB image data is converted to CIELab color space where then each superpixel has its CIELab color computed based on the centroid position given the median of the L, a, and b coordinates of the CIELab color space. The similarity between superpixels is given by the euclidean distance (linear distance in a 3d space), where the user establishes the maximum distance. The user has the option to manually segment the desired pixel (clicking and dragging), manually chose a superpixel for region growing, and load a CSV file with seed positions.

To implement the region growing, records of visited and already segmented superpixels are stored to avoid visiting the same superpixel over and over. Also, the segmentation is drawn to a mask to maintain the original image in memory for comparison.
The region growing method starts by loading seed information, either by obtaining an XY position of a mouse click on the designed interface or by loading CSV files with multiple seeds.

Given a seed, the superpixel in the area is identified and then segmented with the designated color. Using breadth search the reference superpixel is stacked and its neighbors are added to a temporary list (if not yet visited) and compared with the seed superpixel. If the neighbors are within a defined color distance they are segmented in the mask and stacked, and the process starts over unstacking a new superpixel and comparing it with its neighbors. Also, as new superpixels are added, the reference color for the segmentation is recalculated. The process is finished when there is no superpixel to unstack.

## Install

To install this software/script download and unpack the software package in any folder.

This software require Python 3.x and the following libraries:

 - [opencv-python](https://pypi.org/project/opencv-python/) (for image processing)
 - [numpy](https://numpy.org/) (for data structures handling)
 - [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/) (to create user interfaces)
 - [PIL](https://pillow.readthedocs.io/en/stable/) (for image processing)
 - [math](https://docs.python.org/3/library/math.html) (for mathematical oerations)
 - [csv](https://docs.python.org/3/library/csv.html) (to read and write csv files)
 - [base64](https://docs.python.org/3/library/base64.html) (to encode binary objects)
 - [io](https://docs.python.org/3/library/io.html) (to handle binary data streams)
 - [skimage](https://scikit-image.org/) (for image processing  and machine learning)
 - [numba](http://numba.pydata.org/) (for code pre-compiling and optimization)
 
 To install the required libraries navigate to script folder and run:
 
     pip install -r requirements.txt
     
 
 ## Usage
 
 To run this program use:
 
     python Region_growing_superpixel.py
     
     
The use workflow follow the order: Load image -> Generate or Load superpixels -> User click in the desired regions -> region growing operation (optional) -> change or save masks.
 
To access the main functions go to menu bar and then in "Files", to load each file.

<img src="https://github.com/ademirmarquesjunior/Region_growing_Superpixel/blob/master/docs/images/menu_bar.png" width="500" alt="Segmented image">

To generate superpixel data go to "Generate superpixels" that opens a new window asking the method (in this version only the SLIC method is available) to create the superpixel and number of superpixels desired.

After loading or generating a superpixel file click and drag to segment an area or clik in the image and click in "Expand" button considering the class selected. The number given in the box is the maximum dissimilarity allowed when incorporate superpixel neighbors.

<img src="https://github.com/ademirmarquesjunior/Region_growing_Superpixel/blob/master/docs/images/image_segmented.png" width="500" alt="Segmented image">

Select a class and use "Fill" to segment rest of the image that is not marked by other classes.


Use "Clean" to clean the segmentation done with an active class.

Use "Save masks" to save each mask.

## TODO

Next iterations expect to improve and incorporate:

 - A superpixel method based on Iterative Spanning Forest
 - Save and load georreferenced files
 - Save masks routine (change folder)
 - MVC structure
 - User interface and usability
 - Two clicks implementation with least path algorithm (fork?)
 
 
## Credits	
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://vizlab.unisinos.br/) and the following developers:	[Ademir Marques Junior](https://www.researchgate.net/profile/Ademir_Junior).

## License

    MIT Licence (https://mit-license.org/)
    
## How to cite

Yet to be published.
 

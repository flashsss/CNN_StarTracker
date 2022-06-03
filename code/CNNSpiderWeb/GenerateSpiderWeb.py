import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from operator import itemgetter
import shutil
import math
import imutils
import pandas as pd

def createDataSetDirectories(featureOption):
    os.mkdir('./dataset')
    os.mkdir('./dataset/'+featureOption)
    os.mkdir('./dataset/'+featureOption+'/train')
    os.mkdir('./dataset/'+featureOption+'/test')

# From feature_detector.py
def displayImg(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Multitriangles Algorithm
def Spiderweb_detector(image,n):
    # img = cv2.imread(path) 
    img = image 

    #Find the center of the image
    height,width,col = img.shape
    coordinate = [height/2,width/2]
    y = int(coordinate[0])
    x = int(coordinate[1])

    #Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = 50
    params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.minArea = 1
    detector = cv2.SimpleBlobDetector_create(params)

    #Detect stars
    keypoints = detector.detect(img)
    coord = []
    for index,keypoint in enumerate(keypoints):
        x_centralstar = int(round(keypoints[index].pt[0]))
        y_centralstar = int(round(keypoints[index].pt[1]))
        distance_to_center = sqrt(((x_centralstar-x)**2)+((y_centralstar-y)**2))
        coord.append([x_centralstar,y_centralstar,distance_to_center])

    coord = sorted(coord,key = itemgetter(2))
    stars_coordinate = coord[:n]
    for item in stars_coordinate:
        cv2.circle(img,center=(item[0],item[1]),radius=2,color=(255,255,255),thickness=3)

    pivot_star_coord = tuple(coord[0][0:2])
    del coord[0]

    # Draw lines from pivot point to other stars
    for coordinate in stars_coordinate:
        coordinate = coordinate[0:2]
        cv2.line(img,pivot_star_coord,tuple(coordinate),(0,255,0),1)
    
    c = len(coord)

    if c > 0 :
        del stars_coordinate[0]
        axis = np.array(stars_coordinate)
        x = np.array(axis[:, 0])
        y = np.array(axis[:, 1])

        line=[]
        for i in range(int(len(x))):
            theta = math.atan2(y[i]-112,x[i]-112) 
            line.append([x[i],y[i],theta])

        line = (sorted(line, key=itemgetter(2)))
        line = np.array(line,dtype=int)
        line = np.delete(line, 2, 1)
        rows, columns = line.shape

        for a in range(rows-1) :
          cv2.line(img, tuple(line[a]), tuple(line[a+1]), (0,0,255), 1)

        cv2.line(img, tuple(line[rows-1]),tuple(line[0]),(0,0,255),1)
        return img
    else :
      return img

def centroiding(path):
    img = cv2.imread(path)   
    print("ORIGINAL SHAPE: ",img.shape)

    #Find the center of the image
    height,width,col = img.shape
    coordinate = [height/2,width/2]
    y = int(coordinate[0])
    x = int(coordinate[1])

    #Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = 50
    params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.minArea = 1
    detector = cv2.SimpleBlobDetector_create(params)

    #Find the pivot point (Star closest to the center)
    i = 2
    stars_coordinate = []
    while True:
        top = y-i
        down = y+i
        vertical = abs(top-down)
        if vertical >= height:
            new_i = height/2
            top = int(round(y-new_i))
            down = int(round(y+new_i))
        left = x-i
        right = x+i
        horizontal = abs(left-right)
        if horizontal >= width:
            new_i = width/2
            left = int(round(x-new_i))
            right =int(round(x+new_i))
        croppedimg = img[top:down,left:right,:]
        y_crop,x_crop,col = croppedimg.shape
        keypoints = detector.detect(croppedimg)
        print(len(keypoints))
        if len(keypoints) > 3:
            print("Cropped size: \nx: {}\ny: {}".format(x_crop,y_crop))
            print("Full size: \nx: {}\ny: {}".format(width,height))
            for index,keypoint in enumerate(keypoints):
                x_centralstar_crop = int(round(keypoints[index].pt[0]))
                y_centralstar_crop = int(round(keypoints[index].pt[1]))
                print("Crop Coordinates: \nx: {}\ny: {}".format(x_centralstar_crop,y_centralstar_crop))
                coord_x_centralstar = int(round(x_centralstar_crop + ((width-x_crop)/2)))
                coord_y_centralstar = int(round(y_centralstar_crop + ((height-y_crop)/2)))
                stars_coordinate.append([coord_x_centralstar,coord_y_centralstar])
                print("COORDINATES: \n","x: ",coord_x_centralstar,"\n","y: ",coord_y_centralstar)
            break
        i+=2

    #Draw circles on important stars and find the pivot star
    dist_to_center = []
    for coord in stars_coordinate:
        center = (coord[0],coord[1])
        dist_x_to_center = abs(coord[0] - x)
        dist_y_to_center = abs(coord[1] - y)
        resultant = (dist_x_to_center**2+dist_y_to_center**2)**1/2
        dist_to_center.append(resultant)
        cv2.circle(img,center,2,(255,0,0),2)
    
    return img

#Reading in Pandas Dataframe and Cleaning Data
excel_catalogue = pd.read_csv('./SAO.csv')
# tidy_catalogue = excel_catalogue.rename(columns = {'Unnamed: 0': 'Star ID', 'Unnamed: 1': 'RA', 'Unnamed: 2': 'DE', 'Unnamed: 3': "Magnitude"}, inplace=False)
tidy_catalogue = excel_catalogue


#Filtering Magnitude
magnitude_filter = 6.0
less_and_equal_6 = tidy_catalogue['Magnitude']<=magnitude_filter
filtered_catalogue = tidy_catalogue[less_and_equal_6]

#Saving to CSV File
file_name = "Below_" + str(magnitude_filter) + "_SAO.csv"
filtered_catalogue.to_csv("./class_image/"+file_name)

from math import radians,degrees,sin,cos,tan,sqrt,atan,pi,exp


def displayImg(img,cmap='gray'):
    """[Displays image]

    Args:
        img ([numpy array]): [the pixel values in the form of numpy array]
        cmap ([string], optional): [can be 'gray']. Defaults to None.
    """
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

def create_star_image(ra,de,roll,catalogue_path,f=0.00304,myu=1.12*(10**-6)):
    """[summary]

    Args:
        ra ([float]): [right ascension in degrees]
        de ([float]): [declination in degrees]
        roll ([float]): [roll in degrees]
    """


    def create_M_matrix(ra,de,roll,method=2):
        """[summary]

        Args:
            ra ([int]): [right ascension of sensor center]
            de ([int]): [declination of sensor center]
            roll ([int]): [roll angle of star sensor]
            method ([int]): [1 for method 1(Calculating each elements),2 for method 2(calculating rotation matrices)]
        """
        if method == 1:
            a1 = (sin(ra)*cos(roll)) - (cos(ra)*sin(de)*sin(roll))
            a2 = -(sin(ra)*sin(roll)) - (cos(ra)*sin(de)*cos(roll))
            a3 = -(cos(ra)*cos(de))
            b1 = -(cos(ra)*cos(roll)) - (sin(ra)*sin(de)*sin(roll))
            b2 = (cos(ra)*sin(roll)) - (sin(ra)*sin(de)*cos(roll))
            b3 = -(sin(ra)*cos(de))
            c1 = (cos(ra)*sin(roll))
            c2 = (cos(ra)*cos(roll))
            c3 = -(sin(de))
            M = np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
        if method == 2:
            ra_exp = ra - (pi/2)
            de_exp = de + (pi/2)
            M1 = np.array([[cos(ra_exp),-sin(ra_exp),0],[sin(ra_exp),cos(ra_exp),0],[0,0,1]])
            M2 = np.array([[1,0,0],[0,cos(de_exp),-sin(de_exp)],[0,sin(de_exp),cos(de_exp)]])
            M3 = np.array([[cos(roll),-sin(roll),0],[sin(roll),cos(roll),0],[0,0,1]])
            first_second = np.matmul(M1,M2)
            M = np.matmul(first_second,M3)
        return M


    def dir_vector_to_star_sensor(ra,de,M_transpose):
        """[Converts direction vector to star sensor coordinates]

        Args:
            ra ([int]): [right ascension of the object vector]
            de ([int]): [desclination of the object vector]
            M_transpose ([numpy array]): [rotation matrix from direction vector to star sensor transposed]
        """    
        x_dir_vector = (cos(ra)*cos(de))
        y_dir_vector = (sin(ra)*cos(de))
        z_dir_vector = (sin(de))
        dir_vector_matrix = np.array([[x_dir_vector],[y_dir_vector],[z_dir_vector]])
        return M_transpose.dot(dir_vector_matrix)


    def draw_star(x,y,magnitude,gaussian,background,ROI=5):
        """[Draws the star in the background image]

        Args:
            x ([int]): [The x coordinate in the image coordinate system (starting from left to right)]
            y ([int]): [The y coordinate in the image coordinate system (starting from top to bottom)]
            magnitude ([float]): [The stellar magnitude]
            gaussian ([bool]): [True if using the gaussian function, false if using own function]
            background ([numpy array]): [background image]
            ROI ([int]): [The ROI of each star in pixel radius]
        """
        if gaussian:
            H = 2000*exp(-magnitude+1)
            sigma = 5
            for u in range(x-ROI,x+ROI+1):
                for v in range(y-ROI,y+ROI+1):
                    dist = ((u-x)**2)+((v-y)**2)
                    diff = (dist)/(2*(sigma**2))
                    exponent_exp = 1/(exp(diff))
                    raw_intensity = int(round((H/(2*pi*(sigma**2)))*exponent_exp))
                    background[v,u] = raw_intensity
        else:
            mag = abs(magnitude-7) #1 until 9
            radius = int(round((mag/9)*(5)+3))
            color = int(round((mag/9)*(155)+100))
            cv2.circle(background,(x,y),radius,color,thickness=-1)
        return background

    def add_noise(low,high,background):
        """[Adds noise to an image]

        Args:
            low ([int]): [lower threshold of the noise generated]
            high ([int]): [maximum pixel value of the noise generated]
            background ([numpy array]): [the image that is put noise on]
        """
        row,col = np.shape(background)
        background = background.astype(int)
        noise = np.random.randint(low,high=high,size=(row,col))
        noised_img = cv2.addWeighted(noise,0.1,background,0.9,0)
        return noised_img


    #Right ascension, declination and roll
    ra = radians(float(ra))
    de = radians(float(de))
    roll = radians(float(roll))

    #Star sensor pixel
    l = 224
    w = 224

    #Star sensor FOV
    FOVy = degrees(2*atan((myu*w/2)/f))
    FOVx = degrees(2*atan((myu*l/2)/f))

    #STEP 1: CONVERSION OF CELESTIAL COORDINATE SYSTEM TO STAR SENSOR COORDINATE SYSTEM
    M = create_M_matrix(ra,de,roll)
    M_transpose = np.round(np.matrix.transpose(M),decimals=5)

    #Search for image-able stars
    col_list = ["Star ID","RA","DE","Magnitude"]
    star_catalogue = pd.read_csv(catalogue_path,usecols=col_list)
    R = (sqrt((radians(FOVx)**2)+(radians(FOVy)**2))/2)
    alpha_start = (ra - (R/cos(de)))
    alpha_end = (ra + (R/cos(de)))
    delta_start = (de - R)
    delta_end = (de + R)
    star_within_ra_range = (alpha_start <= star_catalogue['RA']) & (star_catalogue['RA'] <= alpha_end)
    star_within_de_range = (delta_start <= star_catalogue['DE']) & (star_catalogue['DE'] <= delta_end)
    star_in_ra = star_catalogue[star_within_ra_range]
    star_in_de = star_catalogue[star_within_de_range]
    star_in_de = star_in_de[['Star ID']].copy()
    stars_within_FOV = pd.merge(star_in_ra,star_in_de,on="Star ID")

    #Converting to star sensor coordinate system
    ra_i = list(stars_within_FOV['RA'])
    de_i = list(stars_within_FOV['DE'])
    star_sensor_coordinates = []
    for i in range(len(ra_i)):
        coordinates = dir_vector_to_star_sensor(ra_i[i],de_i[i],M_transpose=M_transpose)
        star_sensor_coordinates.append(coordinates)

    #STEP 2: CONVERSION OF STAR SENSOR COORDINATE SYSTEM TO IMAGE COORDINATE SYSTEM
    star_loc = []
    for coord in star_sensor_coordinates:
        x = f*(coord[0]/coord[2])
        y = f*(coord[1]/coord[2])
        star_loc.append((x,y))

    xtot = 2*tan(radians(FOVx)/2)*f
    ytot = 2*tan(radians(FOVy)/2)*f
    xpixel = l/xtot
    ypixel = w/ytot

    magnitude_mv = list(stars_within_FOV['Magnitude'])
    filtered_magnitude = []

    #Rescaling to pixel sizes
    pixel_coordinates = []
    delete_indices = []
    for i,(x1,y1) in enumerate(star_loc):
        x1 = float(x1)
        y1 = float(y1)
        x1pixel = round(xpixel*x1)
        y1pixel = round(ypixel*y1)
        if abs(x1pixel) > l/2 or abs(y1pixel) > w/2:
            delete_indices.append(i)
            continue
        pixel_coordinates.append((x1pixel,y1pixel))
        filtered_magnitude.append(magnitude_mv[i])

    background = np.zeros((w,l))

    for i in range(len(filtered_magnitude)):
        x = round(l/2 + pixel_coordinates[i][0])
        y = round(w/2 - pixel_coordinates[i][1])
        background = draw_star(x,y,filtered_magnitude[i],False,background)

    #Adding noise
    background = add_noise(0,50,background=background)

    return background

path_catalog = "./class_image/"+"Below_" + str(4.0) + "_SAO.csv"
path_catalog_6 = "./class_image/"+"Below_" + str(6.0) + "_SAO.csv"
saving_path = './class_image_4/'


catalogue = pd.read_csv(path_catalog)
ra_list = list(catalogue['RA'])
de_list = list(catalogue['DE'])
star_id_list = list(catalogue['Star ID'])
    
#Iterating through all stars
for i in range(len(star_id_list)):
    print("STAR {0} of {1}".format(i+1,len(star_id_list)))
    star_id = star_id_list[i]
    ra = degrees(ra_list[i])
    de = degrees(de_list[i])
    image = create_star_image(ra,de,0,path_catalog_6)
    cv2.imwrite(saving_path+str(i)+'.jpg',image)

#Getting the raw images from the directory
path = './class_image_4/'
featureMethod = 'SpiderWeb'

#prepare new dataset directories
if(os.path.isdir('./dataset')):
  shutil.rmtree('./dataset')
  createDataSetDirectories(featureMethod)
else:
  createDataSetDirectories(featureMethod)

dataset = []
for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path,filename))
    if img is not None:
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dataset.append(gray_img)

#Saving the default image to a directory
for index,image in enumerate(dataset):
  path = './dataset/'+featureMethod+'/train/'+'STAR ID '+str(index)+'/'
  os.mkdir(path)
  os.mkdir('./dataset/'+featureMethod+'/test/'+'STAR ID '+str(index)+'/')
  file_name = '0.jpg'
  cv2.imwrite(path+file_name,image)

featureMethod = 'SpiderWeb'
images = []

for i in range(480):
    path = './dataset/'+featureMethod+'/train/'+'STAR ID '+str(i)
    img = cv2.imread(path+'/0.jpg')
    # img = img[:,50:350]
    images.append(img)

for index,image in enumerate(images):
    count = 1
    path = './dataset/'+featureMethod+'/train/'+'STAR ID '+str(index)+'/'
    for angle in np.arange(0,360,120):
      rotated = imutils.rotate_bound(image,angle)
      rotated = Spiderweb_detector(rotated,6)
      rotated = cv2.resize(rotated, (224, 224), interpolation=cv2.INTER_NEAREST)
      cv2.imwrite(path+str(count)+'.jpg',rotated)
      print("DONE SAVING...",path+str(count)+'.jpg')
      count+=1

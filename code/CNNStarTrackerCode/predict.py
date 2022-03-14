from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

#Net Algorithm
from math import sqrt
from operator import itemgetter
def net_feature(image,n):
    # img = cv2.imread(path)
    img = image   
    
    #Find the center of the image
    height,width,col = img.shape
    coordinate = [height/2,width/2]
    y = int(coordinate[0])
    x = int(coordinate[1])
    print(x,y)

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

    coord = sorted(coord,key=itemgetter(2))
    coord = coord[:n]
    for item in coord:
        cv2.circle(img,center=(item[0],item[1]),radius=2,color=(0,0,255),thickness=2)

    pivot_star_coord = tuple(coord[0][0:2])
    del coord[0]

    #Draw lines from pivot point to other stars
    for coordinate in coord:
        coordinate = coordinate[0:2]
        cv2.line(img,pivot_star_coord,tuple(coordinate),(0,0,255),2)

    return img

#finding brightness star code

def displayImg(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

img = cv2.imread("./class_image_6/302.jpg") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
cv2.circle(img, maxLoc, 5, (0, 0, 255), 2)
displayImg(img)
print(img.shape)
print(maxLoc)
a,b = maxLoc
newimg = img[b-112:b+112, a-112:a+112]
print("CROPED SHAPE: ",newimg.shape)
gol = net_feature(newimg,5)
displayImg(gol)
print("gol shape: ",gol.shape)

model = keras.models.load_model('./Results/preprocessed_features_model.h5)
prediction = model.predict(gol)
print(prediction)
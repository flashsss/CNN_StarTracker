from tensorflow.keras.models import load_model
model = load_model('./Results/preprocessed_features_model.h5')

img = cv2.imread("./class_image_4/300.jpg") 
# img = multitriangles_detector(img,5)
displayImg(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
cv2.circle(img, maxLoc, 5, (255, 0, 0), 2)
displayImg(img)
print(maxLoc)
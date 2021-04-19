import pytesseract
import cv2

im = cv2. imread("image_test1.png")

roi = cv2.selectROI(im)

im_cropped = im[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

cv2.imwrite("imageText.png",im_cropped)

#cv2.imshow("Cropped Image", im_cropped)
#cv2.waitKey(0)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('imageText.png',cv2.IMREAD_COLOR) #Open the image from which charectors has to be recognized

#img = cv2.resize(img, (620,480) ) #resize the image if required



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey to reduce detials

gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise


configuration = ("-l eng --oem 1 --psm 8")
original = pytesseract.image_to_string(gray, config="configuration")

#test = (pytesseract.image_to_data(gray, lang=None, config='', nice=0) ) #get confidence level if required

#print(pytesseract.image_to_boxes(gray))


print(original)

f = open("result.txt", "w")
f.write(original)
f.close()


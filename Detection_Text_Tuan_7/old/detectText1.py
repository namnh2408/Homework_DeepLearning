##Loading the necessary packages 
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt

# Tạo từ điển đối số cho các đối số mặc định cần thiết trong mã.
args = {"image":"image_test.png", "east":"east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}

# Lấy hình ảnh.
image = cv2.imread(args['image'])

# Lưu hình ảnh với hình dạng ban đầu
orig = image.copy()

#origH là chiều cao của ảnh, origW là chiều rộng của ảnh
(origH, origW) = image.shape[:2]

# set the new height and width to default 320 by using args #dictionary.  
(newW, newH) = (args["width"], args["height"])

#Calculate the ratio between original and new image for both height and weight. 
#This ratio will be used to translate bounding box location on the original image. 
rW = origW / float(newW)
rH = origH / float(newH)

# resize the original image to new dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# tạo một đốm màu từ hình ảnh để chuyển tiếp nó sang mô hình EAST
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    
# tải mô hình EAST được đào tạo trước để phát hiện văn bản 
net = cv2.dnn.readNet(args["east"])

# Lấy hai đầu ra từ mô hình EAST.
# 1. Điểm xác suất cho khu vực có chứa văn bản hay không.
# 2. Hình học của văn bản - Tọa độ của hộp giới hạn phát hiện văn bản
# Hai lớp sau cần được lấy từ mô hình EAST để đạt được điều này.
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
    
# Chuyển tiếp vượt qua đốm màu từ hình ảnh để có được các lớp đầu ra mong muốn
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

## Trả về hộp giới hạn và điểm xác suất nếu nó lớn hơn độ tin cậy tối thiểu
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# vòng lặp qua các hàng
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# vòng lặp qua số cột
		for i in range(0, numC):
			if scoresData[i] < args["min_confidence"]:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# trích xuất góc quay để dự đoán và tính toán sin và cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# sử dụng khối lượng địa lý để nhận kích thước của hộp giới hạn
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# tính toán bắt đầu và kết thúc cho hộp thư trước văn bản
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# hộp giới hạn trả về và trust_val được liên kết
	return (boxes, confidence_val)



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tìm dự đoán và áp dụng phương pháp triệt tiêu không cực đại
(boxes, confidence_val) = predictions(scores, geometry)
#print(predictions(scores, geometry))
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

##Text Detection and Recognition 

# initialize the list of results
results = []

# loop over the bounding boxes to find the coordinate of bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	#extract the region of interest
	r = orig[startY:endY, startX:endX]
    
	#configuration setting to convert image to string.  
	configuration = ("-l eng --oem 1 --psm 8")
    ##This will recognize the text from the image of bounding box
	text = pytesseract.image_to_string(r, config=configuration)

	# append bbox coordinate and associated text to the list of results 
	results.append(((startX, startY, endX, endY), text))

#Display the image with bounding box and recognized text
orig_image = orig.copy()

# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
	# display the text detected by Tesseract
	print("{}".format(text))

	# Displaying text
	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 0, 255), 2)
	cv2.putText(orig_image, text, (start_X, start_Y - 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)



plt.imshow(orig_image)
plt.title('Output')
plt.show()
plt.show()
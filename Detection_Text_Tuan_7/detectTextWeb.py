
import pytesseract
import cv2 
import matplotlib.pyplot as plt 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Nhận diên Text thông qua webcame
font_scale = 1.5

font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("Record.mp4")

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcame")

cntr = 0
while True:
    ret, frame = cap.read()
    cntr = cntr + 1;
    
    # Screenshot
    k = cv2.waitKey(1)
    if k%256 == 27: # Esc
        print("Escape hit, closing the app")
        break
    elif k%256 == 32: # Space
        img_name = "image_test1.png"
        cv2.imwrite(img_name,frame)
        print("Screenshot taken")
        
    if((cntr%20)==0):
        imgH, imgW, _ = frame.shape
        
        x1, y1, w1, h1 = 0, 0, imgH, imgW
        
        imgchar = pytesseract.image_to_string(frame)
        
        imgboxes = pytesseract.image_to_boxes(frame)
        
        for boxes in imgboxes.splitlines():
            boxes = boxes.split(' ')
            x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
            cv2.rectangle(frame, (x, imgH-y), (w, imgH-h), (0,0,255), 3)
            
        cv2.putText(frame,imgchar, (x1 + int(w1/50), y1 + int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.imshow('Text Detection Tutorial', frame)
        
        if cv2.waitKey(2) & 0xFF == ord ('q'):
            break

cap.release()
cv2.destroyAllWindows()
        
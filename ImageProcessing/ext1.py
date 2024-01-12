"""
Page segmentation modes: 
O Orientation and script detection (OSD) only
1 Automatic page segmentation with OSD. 
2 Automatic page segmentation, but no OSD, or OCR.
3 Fully automatic page segmentation, but no OSD. (Default)
4 Assume a single column of text of variable sizes.
5 Assume a single uniform block of vertically aligned text.
6 Assume a single uniform block of text
7 Treat the image as a single text line.
8 Treat the image as a single word.
9 Treat the image as a single word in a circle.
10 Treat the image as a single character.
11 Sparse text. Find as much text as possible in no particular order.
12 Sparse text with OSD.
13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract—specific.
"""

import pytesseract
from pytesseract import Output
import PIL.Image
import cv2

myconfig = r"--psm 6 --oem 3"

# text = pytesseract.image_to_string(PIL.Image.open('img1.png'),config=myconfig)
# print(text)

# img = cv2.imread('2.JPEG')

# data = pytesseract.image_to_data(img,config = myconfig, output_type=Output.DICT)
# # print(data.keys())

# nBoxes = len(data['text'])

# for i in range(nBoxes):
#     if float(data['conf'][i]) > 80:
#         (x,y,width,height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#         img = cv2.rectangle(img, (x,y),(x+width, y+height), (0,0,255), 2)
#         img = cv2.putText(img, data['text'][i], (x, y+width+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


# # cv2.imshow("img", img)
# # cv2.waitKey(0)

# scale_percent = 40  # Adjust the scale percentage as needed
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)

# # Resize the image
# img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# for i in range(nBoxes):
#     if float(data['conf'][i]) > 10:
#         (x, y, box_width, box_height) = (
#             int(data['left'][i] * scale_percent / 100),
#             int(data['top'][i] * scale_percent / 100),
#             int(data['width'][i] * scale_percent / 100),
#             int(data['height'][i] * scale_percent / 100),
#         )
        
#         # Draw rectangles and text on the resized image
#         img_resized = cv2.rectangle(
#             img_resized, (x, y), (x + box_width, y + box_height), (0, 0, 255), 2
#         )
#         img_resized = cv2.putText(
#             img_resized,
#             data['text'][i],
#             (x, y + box_height + 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 0, 255),
#             1,
#         )

# # Display the resized image
# cv2.imshow("Resized Image", img_resized)
# cv2.waitKey(0)


import cv2
import pytesseract
import numpy as np

# Load the image
img = cv2.imread('2.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding or other preprocessing to highlight text
_, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours to find the contour with the maximum area (assuming it's the text area)
max_area = 0
max_contour = None

for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Draw a bounding box around the maximum area contour
x, y, w, h = cv2.boundingRect(max_contour)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Extract the region of interest (ROI) containing the text
roi = img[y:y+h, x:x+w]

# Change the perspective of the ROI to straighten the text
pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(roi, matrix, (w, h))

# Apply OCR to extract text from the straightened image
text = pytesseract.image_to_string(result, config='--psm 12')

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Text Area', roi)
cv2.imshow('Straightened Text', result)
print("Extracted Text:", text)
cv2.waitKey(0)
cv2.destroyAllWindows()
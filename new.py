import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#이미지를 가져오고, HSV, GRAY 포맷으로 변환
# 2진 mask를 만들기위한 color segmentation
# color segmentation to create a binary mask
image = cv2.imread("wash.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lower = np.array([0, 0, 218])
upper = np.array([157, 54, 255])
mask = cv2.inRange(hsv, lower, upper)

#  horizontal kernel 생성, 텍스트 각각을 붙이기 위해 dilate 생성
kernel = np.ones((3,3), np.uint8)
gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
adap_src2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
dilate = cv2.dilate(adap_src2, kernel, iterations=2)

#경계선 찾고, 비율로 안되는거 버리기
#텍스트가 아닌거 없애기
cnts, hierachy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
for i in cnts:
    cv2.drawContours(image, [i], -1, (0,255,0), 3)

# print(cnts[0])
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ar = w / float(h)
    if ar < 5:
        cv2.drawContours(dilate, [c], -1, (0,0,0), 1)

result = 255 - cv2.bitwise_and(dilate, mask)
data = pytesseract.image_to_string(mask, lang='kor', config='--psm 6')
print(type(data))
print(data)

save_idx = []
idx = 0
while idx < len(data):
    if ord(data[idx]) == 32:
        # print("공백시작 : ", idx)
        count = 0
        i = idx + 1
        while ord(data[i]) == 32:
            # print("공백 또 발생 : ", i)
            count += 1
            i += 1
            if i >= len(data):
                break
        # print("공백끝 (i, idx, count) :  ", i, idx, count)
        if count >= 1:
            save_idx.append((idx + 1, i-1))
        idx = i
    elif ord(data[idx]) == 34 or \
            ord(data[idx]) == 39 or \
            ord(data[idx]) == 40 or \
            ord(data[idx]) == 41 or \
            ord(data[idx]) == 44 or \
            ord(data[idx]) == 45 or \
            ord(data[idx]) == 59 or \
            ord(data[idx]) == 94 or \
            ord(data[idx]) == 95 or \
            ord(data[idx]) == 123 or \
            ord(data[idx]) == 124 or \
            ord(data[idx]) == 125 or \
            ord(data[idx]) == 126:
        save_idx.append((idx, idx))
        idx += 1
    else:
        idx += 1

new_data = ""
print(save_idx)
now_idx = 0
for (start, end) in save_idx:
    new_data += data[now_idx:start]
    now_idx = end + 1
    if now_idx >= len(data):
        break

if now_idx < len(data):
    new_data += data[now_idx:]
print(new_data)


cv2.imshow('image', image)
cv2.imwrite('saved_image2.jpg', image)
cv2.waitKey()
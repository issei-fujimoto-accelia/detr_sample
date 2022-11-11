import cv2

filename = "./images/ninjin-rotate-crop.jpeg"
img = cv2.imread(filename, 0) # グレースケール
 
height, width = img.shape

# ret, thresh = cv2.threshold(img, 96, 255, cv2.THRESH_BINARY)# 閾値で2値化
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)# 閾値で2値化

img_r = cv2.bitwise_not(thresh) # 反転処理

contours, hierarchy = cv2.findContours(img_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 輪郭抽出

img_color = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR) # 輪郭を赤い線で描くためにカラー画像へ変換

contours = list(filter(lambda x: cv2.contourArea(x) > 5000, contours)) # 小さい輪郭は無視する

cv2.drawContours(img_color, contours, -1, color=(0, 0, 255), thickness=2) # 輪郭を描画

cv2.imwrite("ninjin_rinkaku.jpg", img_color)



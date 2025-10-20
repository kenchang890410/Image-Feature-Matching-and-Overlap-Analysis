# Image Feature Matching and Overlap Analysis

這個 Python 程式是一個基於電腦視覺 (Computer Vision) 的分析工具，用於測量兩張影像之間的重疊程度 (Overlap)，並評估影像配準的準確性。

程式中使用了 SIFT 特徵偵測和 FLANN，計算出將一張影像轉換到另一張影像平面所需的單應性矩陣 (Homography)，最終輸出重疊的百分比和配準誤差。

## 核心功能與技術

程式首先進行特徵點偵測，採用 SIFT (尺度不變特徵轉換) 演算法來提取兩張影像中的關鍵點和描述子。

隨後，使用高效的 FLANN (Fast Library for Approximate Nearest Neighbors) 匹配器。

配合 K-Nearest Neighbors (KNN) 算法，找到最佳的匹配點。

獲得高品質的匹配點後，使用 RANSAC (Random Sample Consensus) 演算法來計算單應性矩陣 (Homography Matrix)。

單應性矩陣能將第二張影像的座標系映射到第一張影像的座標系上。這個步驟是影像配準的關鍵。

## 輸入與輸出

### 輸入

在程式同目錄下放置兩張待分析的影像，檔案名稱必須命名為 image1.jpg 和 image2.jpg。

### 輸出指標

Number of successfully matched feature points : 27 //匹配點數量

Overlap : 97.67% //兩張影像重疊百分比

Average error: 0.40 px //配準精確度的平均像素誤差

Max error: 3.48 px //所有匹配點中最大的像素誤差

### 輸出影像檔案

output_matches.jpg : 將兩張影像拼接在一起，並用彩色線段連接所有成功的特徵點對，如下圖。

![image](https://github.com/kenchang890410/Image-Feature-Matching-and-Overlap-Analysis/blob/37a74bcc011a721c807a91ae20b8f7c1c6dabe98/output_matches.jpg)

output_error_check.jpg : 在第一張影像上，以綠圈標記原圖特徵點，以紅圈標記投影點，用藍線連接表示配準偏差，如下圖。

![image](https://github.com/kenchang890410/Image-Feature-Matching-and-Overlap-Analysis/blob/37a74bcc011a721c807a91ae20b8f7c1c6dabe98/output_error_check.jpg)

output_warp.jpg : 將第二張影像根據單應性矩陣轉換到第一張影像視角後的影像，如下圖。

![image](https://github.com/kenchang890410/Image-Feature-Matching-and-Overlap-Analysis/blob/37a74bcc011a721c807a91ae20b8f7c1c6dabe98/output_warp.jpg)

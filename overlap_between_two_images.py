import cv2
import numpy as np

def getSIFTkps(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def matchSIFTkps(kp1, des1, kp2, des2):
    good_kps1, good_kps2, good_matches = [], [], []

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good_matches.append(m)
            good_kps1.append(kp1[m.queryIdx])
            good_kps2.append(kp2[m.trainIdx])

    return good_kps1, good_kps2, good_matches

def compute_overlap(img1, img2):
    kp1, des1 = getSIFTkps(img1)
    kp2, des2 = getSIFTkps(img2)

    if des1 is None or des2 is None:
        print("Unable to obtain features")
        return

    good_kps1, good_kps2, good_matches = matchSIFTkps(kp1, des1, kp2, des2)
    print(f"Number of successfully matched feature points : {len(good_matches)}")

    if len(good_matches) >= 4:
        src_pts = np.float32([kp.pt for kp in good_kps2]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp.pt for kp in good_kps1]).reshape(-1, 1, 2)

        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if h is not None:
            
            dst = cv2.warpPerspective(img2, h, (img1.shape[1], img1.shape[0]))

            total_pixel = img1.shape[0] * img1.shape[1]
            num_blank = np.sum(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) == 0)
            blank_percent = (num_blank / total_pixel) * 100
            overlap_percent = 100 - blank_percent
            print(f"Overlap : {overlap_percent:.2f}%")

            projected_pts = cv2.perspectiveTransform(src_pts, h)
            errors = np.linalg.norm(projected_pts - dst_pts, axis=2)
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            print(f"Average error: {mean_error:.2f} px")
            print(f"Max error: {max_error:.2f} px")

            debug_img = img1.copy()

            for idx, (pt1, pt2) in enumerate(zip(dst_pts, projected_pts)):
                p1 = tuple(np.int32(pt1[0]))
                p2 = tuple(np.int32(pt2[0]))

                cv2.circle(debug_img, p1, 4, (0, 255, 0), -1)
                cv2.circle(debug_img, p2, 4, (0, 0, 255), 1)
                cv2.line(debug_img, p1, p2, (255, 0, 0), 1)

            cv2.imwrite("output_warp.jpg", dst)
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            height = max(h1, h2)
            separator = 10

            concat_width = w1 + w2 + separator
            concat_img = np.zeros((height, concat_width, 3), dtype=np.uint8)

            concat_img[:h1, :w1] = img1
            concat_img[:h2, w1+separator:] = img2
            concat_img[:, w1:w1+separator] = 220

            for m in good_matches:
                pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))

                pt2_shifted = (pt2[0] + w1 + separator, pt2[1])

                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(concat_img, pt1, pt2_shifted, color, 1, cv2.LINE_AA)
                cv2.circle(concat_img, pt1, 3, color, -1)
                cv2.circle(concat_img, pt2_shifted, 3, color, -1)

            cv2.imwrite("output_matches.jpg", concat_img)
            cv2.imwrite("output_error_check.jpg", debug_img)

            cv2.imshow("Matches", concat_img)
            cv2.imshow("Warp Result", dst)
            cv2.imshow("Error Check", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Unable to calculate homography matrix")
    else:
        print("Insufficient matching points to estimate the homography matrix")

if __name__ == '__main__':
    img1 = cv2.imread("image1.jpg")
    img2 = cv2.imread("image2.jpg")

    if img1 is None or img2 is None:
        print("Unable to load image")
    else:
        compute_overlap(img1, img2)

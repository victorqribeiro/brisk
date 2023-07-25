import cv2 as cv
import matplotlib.pyplot as plt

image1 = cv.imread(filename = 'image1.jpg', flags = cv.IMREAD_GRAYSCALE)

image2 = cv.imread(filename = 'image2.jpg', flags = cv.IMREAD_GRAYSCALE)
                   
BRISK = cv.BRISK_create()

keypoints1, descriptors1 = BRISK.detectAndCompute(image1, None)
keypoints2, descriptors2 = BRISK.detectAndCompute(image2, None)

BFMatcher = cv.BFMatcher(normType = cv.NORM_HAMMING, crossCheck = True)

matches = BFMatcher.match(queryDescriptors = descriptors1, trainDescriptors = descriptors2)

matches = sorted(matches, key = lambda x: x.distance)

output = cv.drawMatches(img1 = image1,
                        keypoints1 = keypoints1,
                        img2 = image2,
                        keypoints2 = keypoints2,
                        matches1to2 = matches[:15],
                        outImg = None,
                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(output)
plt.show()

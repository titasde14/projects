import numpy as np 
from skimage.feature import hog
import glob
import pdb
import cv2

img_list = glob.glob('/Users/titas/Desktop/labelled_videos/images/*.jpg')
txt_list = glob.glob('/Users/titas/Desktop/labelled_videos/labels_3class/*.txt')

N_data = len(img_list)

x_feat = []
y_feat = []

for ctr in range(N_data):
	#print(ctr+1)
	image = cv2.imread(img_list[ctr],0)
	Nrows,Ncols = image.shape[0],image.shape[1]
	
	with open(txt_list[ctr],'r') as file:
		labels = file.readlines()

	for label in labels:
		content = label.split(' ')

		class_label = int(content[0])
		x_cent = int(float(content[1])*Ncols)
		y_cent = int(float(content[2])*Nrows)
		width = int(float(content[3])*Ncols)
		height = int(float(content[4])*Nrows)

		if width<50 or height<50:
			continue

		xmin = max(1,x_cent-width//2)
		xmax = min(Ncols-1,x_cent+width//2)

		ymin = max(1,y_cent-height//2)
		ymax = min(Ncols-1,y_cent+height//2)

		#print(xmin,xmax,ymin,ymax)
		roi = cv2.resize(image[ymin:ymax,xmin:xmax], (50, 50) , interpolation = cv2.INTER_CUBIC)
		#print(roi.shape)
		#pdb.set_trace()
		#x_feat.append(hog(roi, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(3, 3), visualize=False, multichannel=False, block_norm='L1'))
		x_feat.append(roi.ravel())
		y_feat.append(class_label)

x_feat = np.array(x_feat).astype(np.float16)
print(x_feat.shape)
y_feat = np.array(y_feat).astype(np.uint8)
np.savez_compressed('labelled_video_pixel_feat', x_feat=x_feat, y_feat=y_feat)
pdb.set_trace()




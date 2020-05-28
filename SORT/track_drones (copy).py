import cv2 as cv
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import time
import argparse
import os
import glob
import pdb
from sort import Sort
import json


class Tracker:

	def __init__(self, files, img_size = [512, 640], min_area=95,display=True, use_gt=False):
		self.min_bbox_area = min_area
		self.files = files
		self.img_size = img_size
		self.display = display
		self.use_gt = use_gt
		self.start_time = time.time()
		self.count = 0
		self.count1 = 0
        
		self.output_dir='./output/'
        
		fname='ground_truth.csv'
		self.all_bbox = np.loadtxt(fname ,delimiter=',') 
		self.all_bbox[:,2:]=self.all_bbox[:,2:]/2
		self.all_bbox1=self.all_bbox
		self.all_bbox=self.all_bbox.astype(int)
		self.gt_bbox_count = 0
                
		self.initSortAlgorithm()
	'''
		Function to get detections from the files which are bundled 
	'''
	def doDetect(self,file):

		bundle_img = cv.imread(file)
		# Input image
		img  = bundle_img[:,:960]
		print(bundle_img.shape)
		if bundle_img.shape[1] == 3*self.img_size[1]:
			if self.use_gt:
				seg = bundle_img[:,1*960:2*960]
			else:
				seg = bundle_img[:,2*self.img_size[1]:]
    	# Find the binary mask (note that segmentation has blue color
		mask = seg[:,:,0] > 200
		mask = 255*mask.astype(np.uint8)

		contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

		bbox_list = []

		if len(contours) == 0:
			return img, []

		for cnt in contours:
			rect = cv.boundingRect(cnt)
			x, y, w, h = rect 
			if w*h > self.min_bbox_area:
				new_list = [rect[0], rect[1], rect[0]+w, rect[1]+h,1]
				bbox_list.append(new_list)

# 		import pdb
# 		pdb.set_trace()
		for box in bbox_list:
			start_point = tuple(box[0:2])
			end_point   = tuple(box[2:4])
			cv.rectangle(img, start_point, end_point, color=(255, 30, 50),	thickness = 2)
# 		plt.imshow(img)
# 		plt.show()
		# print(bbox_list)

		return img, np.array(bbox_list)

	def doDetect_gt(self,file,f_id):
		bundle_img = cv.imread(file)
		# Input image
		img  = bundle_img[:,:960]
# 		print(bundle_img.shape)
		bbox_list = []

		if self.all_bbox[self.gt_bbox_count][1] != f_id:
			return img, []

		while self.all_bbox[self.gt_bbox_count][1] == f_id:
			x1, y1, x2, y2 = self.all_bbox[self.gt_bbox_count][2:] 

			if (x2-x1)*(y2-y1) > self.min_bbox_area:
				new_list = [x1, y1, x2, y2,1]
				bbox_list.append(new_list)
			self.gt_bbox_count+=1

# 		import pdb
# 		pdb.set_trace()
		for box in bbox_list:
 			start_point = tuple(box[0:2])
 			end_point   = tuple(box[2:4])
 			cv.rectangle(img, start_point, end_point, color=(255, 30, 50),	thickness = 2)
# 		plt.imshow(img)
# 		plt.show()
		# print(bbox_list)

		return img, np.array(bbox_list)



	def getData(self, trackers):
		dataList = []
		timestamp = time.time() - self.start_time
		data = {}
		data["timestamp"] = timestamp
		data["num"] = self.count
		self.count+=1
		data["class"] = "frame"
		annotations = [] 
		if self.count==451:
			print("here")
		for d in trackers:
			
			if self.all_bbox1[self.count1][1] == self.count-1:
				an_data = {}
				zz=self.all_bbox1[self.count1]
				self.count1+=1            
				if self.use_gt:
					an_data["dco"] = True
				an_data["height"] = round((zz[4] - zz[2]),2)
				an_data["width"] = round((zz[5] - zz[3]),2)
				an_data["id"] = int(d[4])
				an_data["x"] = (zz[2])
				an_data["y"] = (zz[3])
				annotations.append(an_data)
		if self.use_gt:
			data["annotations"] = annotations
		else:
			data["hypotheses"] = annotations
		return data


	def initSortAlgorithm(self,use_gt=False):
		frames = []
        
		self.trackObj = Sort(max_age=3, iou_threshold=0.05)
		total_time = 0.0
	  	# for disp
		if self.display:
			colours = np.random.rand(32, 3)  # used only for display
			plt.ion()
			fig = plt.figure()		

		is_started_tracking = False
		for f_id, f in enumerate(self.files):
            
			#img, detections = self.doDetect(f)
			img, detections = self.doDetect_gt(f,f_id)
			print(f_id)
			if self.display:
				ax1 = fig.add_subplot(111, aspect='equal')
				ax1.imshow(img)
				

			if detections is None:
				if not is_started_tracking:
					print(">> NO detection on first frames yet. Continue")
				else:
					trackers = prev_trackers 
			else:
				# start_time = time.time()
				#update tracker
				is_started_tracking = True 
				trackers = self.trackObj.update(detections,img)
				prev_trackers = trackers 
			# cycle_time = time.time() - start_time
			# total_time += cycle_time
			frames.append(self.getData(trackers))
			for d in trackers:
				
                # f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[4], frame, 1, 1, d[0], d[1], d[2], d[3]))
				if (self.display):
					d = d.astype(np.int32)
					ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
					                                ec=colours[d[4] % 32, :]))
					ax1.set_adjustable('box')
					#label
					ax1.annotate('id = %d' % (d[4]), xy=(d[0], d[1]), xytext=(d[0], d[1]))
 					#if detections != []:#detector is active in this frame
					ax1.annotate(" DETECTOR", xy=(5, 45), xytext=(5, 45))
# 			ax1.imshow(img)
			#if (self.display):
				#plt.axis('off')
				#fig.canvas.flush_events()
				#plt.show()
				#fig.tight_layout()
				#if f_id==2191 or f_id==2208 or f_id==2225 or f_id==2242 or f_id==2259 or f_id==2276 or f_id==2293 or f_id==2310:	
				#fig.savefig('output/'+str(f_id).zfill(4)+'_final_seg.png', dpi=20)                
				#save the frame with tracking boxes
				#ax1.cla()
		
		if self.use_gt:
			filename = os.path.join(self.output_dir,"seg.json")
		else:
			filename = os.path.join(self.output_dir, "prediction.json")

		fileOutput = []
		fileData = {}	
		with open(filename, "w") as file:
			fileData["frames"] = frames
			fileData["class"] = "video"
			fileData["filename"] = filename
			fileOutput.append(fileData)
			# print(json.dumps(fileOutput))
			json.dump(fileOutput, file,ensure_ascii=False, indent=4)





if __name__ == "__main__":

	filePath = './data/mobilenetskipadd_cross_entropy_augmented_rgb/'
	files    = sorted(glob.glob(os.path.join(filePath, '*.jpg')))

	t = Tracker(files,use_gt=True)
	print("Done")




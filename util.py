import torch
import torch.utils.data as data_utils

from PIL import Image
import numpy as np


imfile = 'images/rgb.png'
gtfile = 'images/gt.png'
mean_image_file = 'images/mean.npy'
pix_threshold = 0.5 # percentage of allowed blank pixels in patches (for training)

# Opens the image, apply zero padding so that it can be cropped into WxW patches
def load_data_as_np(W):
	im = Image.open(imfile)
	gt = Image.open(gtfile)

	# calculate new size to evenly crop
	w_orig = im.size[0]
	h_orig = im.size[1]

	w_new  = (w_orig + (W - w_orig%W))
	h_new  = (h_orig + (W - h_orig%W))

	# create new image (and gt) with zero padding
	im_new = Image.new(im.mode, (w_new, h_new), (0,0,0,0))
	im_new.paste(im, im.getbbox())
	gt_new = Image.new(gt.mode, (w_new, h_new), (255,255,255))
	gt_new.paste(gt, gt.getbbox())

	# allocate np array
	x = np.zeros((int(w_new/W * h_new/W), 3, W, W))
	y = np.zeros((int(w_new/W * h_new/W), W, W))

	imCounter = 0

	# start cropping
	for w in range(0, w_new, W):
		for h in range(0, h_new, W):
			box = (w, h, w+W, h+W) # a WxW box at (w,h)
			cropped_im = im_new.crop(box).convert('RGB')
			cropped_gt = gt_new.crop(box).convert('1')
			
			# do cropping
			xim = np.array(cropped_im)
			xgt = np.array(cropped_gt)

			# TODO double check the dim ordering
			x[imCounter, :, :, :] = np.rollaxis(xim, 2, 0)
			y[imCounter, :, :] = xgt
			imCounter = imCounter+1

	return x, y

# Wrapper function for load_data_as_np, return Dataset object
def load_data_simple(W):
	x, y = load_data_as_np(W)

	# preprocess data, convert it to tensor, then into TensorDataset
	x, y = preprocess(x, y)
	x = torch.from_numpy(x)
	y = torch.from_numpy(y)
	x = x.type(torch.FloatTensor)
	y = y.type(torch.LongTensor)
	dataset = data_utils.TensorDataset(x, y)

	return dataset

# Loads WxW patches (via load_data_as_np(W)) and removes "some" for training purposes
# returns Dataset object
def load_data_analyzed(W):
	x, y = load_data_as_np(W)
	x_analyzed = np.zeros_like(x)
	y_analyzed = np.zeros_like(y)

	imCounter = 0

	# go through patches to eliminate ones that not conform to the criteria
	for i in range(x.shape[0]):		
		# analyze cropped image with following criteria:
		xim = x[i, :, :, :]
		xgt = y[i, :, :]
		# more than 50% of the image is blank
		isNotEnoughData = (np.count_nonzero(xim) / np.size(xim)) < pix_threshold
		# no houses on the cropped image
		isNoGt = np.count_nonzero(xgt) == np.size(xgt)

		if (isNoGt and isNotEnoughData):
			continue

		# TODO double check the dim ordering (PIL has rows-cols ordering)
		x_analyzed[imCounter, :, :, :] = xim
		y_analyzed[imCounter, :, :] = xgt
		imCounter = imCounter+1


	# trim the array
	x_analyzed = x_analyzed[0:imCounter, :, :, :]
	y_analyzed = y_analyzed[0:imCounter, :, :]

	# save it to use for zero centering
	save_mean_image(x_analyzed)

	x_analyzed, y_analyzed = preprocess(x_analyzed, y_analyzed)
	x_analyzed = torch.from_numpy(x_analyzed)
	y_analyzed = torch.from_numpy(y_analyzed)
	x_analyzed = x_analyzed.type(torch.FloatTensor)
	y_analyzed = y_analyzed.type(torch.LongTensor)
	dataset = data_utils.TensorDataset(x_analyzed, y_analyzed)

	return dataset

# Opens the image, apply zero padding so that it can be cropped into WxW patches,
# apply augmentation according to the params
#		stride_ratio: amount of stride in terms of W (see nested loops below)
def load_data_augmented(W, stride_ratio=1.0):
	im = Image.open(imfile)
	gt = Image.open(gtfile)

	# calculate new size to evenly crop
	w_orig = im.size[0]
	h_orig = im.size[1]

	w_new  = (w_orig + (W - w_orig%W))
	h_new  = (h_orig + (W - h_orig%W))

	# create new image (and gt) with zero padding
	im_new = Image.new(im.mode, (w_new, h_new), (0,0,0,0))
	im_new.paste(im, im.getbbox())
	gt_new = Image.new(gt.mode, (w_new, h_new), (255,255,255))
	gt_new.paste(gt, gt.getbbox())

	# allocate np array
	x = np.zeros((int(w_new/W * 1/stride_ratio * h_new/W * 1/stride_ratio), 3, W, W))
	y = np.zeros((int(w_new/W * 1/stride_ratio * h_new/W * 1/stride_ratio), W, W))

	imCounter = 0

	# start cropping
	stride = int(W*stride_ratio)
	for w in range(0, w_new-stride, stride):
		for h in range(0, h_new-stride, stride):
			box = (w, h, w+W, h+W) # a WxW box at (w,h)
			cropped_im = im_new.crop(box).convert('RGB')
			cropped_gt = gt_new.crop(box).convert('1')
			
			# do cropping
			xim = np.array(cropped_im)
			xgt = np.array(cropped_gt)

			# more than 50% of the image is blank
			isNotEnoughData = (np.count_nonzero(xim) / np.size(xim)) < pix_threshold
			# no houses on the cropped image
			isNoGt = np.count_nonzero(xgt) == np.size(xgt)

			if (isNoGt and isNotEnoughData):
				continue

			# TODO double check the dim ordering
			x[imCounter, :, :, :] = np.rollaxis(xim, 2, 0)
			y[imCounter, :, :] = xgt
			imCounter = imCounter+1

	# trim the array
	x = x[0:imCounter, :, :, :]
	y = y[0:imCounter, :, :]

	# save it to use for zero centering
	save_mean_image(x)

	x, y = preprocess(x, y)
	x = torch.from_numpy(x)
	y = torch.from_numpy(y)
	x = x.type(torch.FloatTensor)
	y = y.type(torch.LongTensor)
	dataset = data_utils.TensorDataset(x, y)

	return dataset

# Saves the mean image of a given set
def save_mean_image(x):
	mean_image = np.mean(x, axis=0)
	np.save(mean_image_file, mean_image)

# Loads the mean image, zero center the data using it
def preprocess(x, y):
	# use the mean image obtained from training set
	try:
		mean_image = np.load(mean_image_file)
	except IOError:
		print("File not found: ", mean_image_file)
		print("pix4d_util.load_data_analyzed(W) generates this file during training.")
		return

	x -= mean_image
	# x = x/255.0 # also scale between [-1, 1]

	# invert y, so that "1" is house and 0 is background
	y = np.logical_not(y).astype(int)
	return x, y

# Given network output + ground truth, calculates precision/recall/F1 score
# Both predictions and gt assumed Tensor
def evaluate(predictions, gt):

	acc = (predictions == gt)
	accuracy = acc.sum() / acc.numel()

	tp = np.logical_and(predictions, gt).sum() # True positives

	precision = tp / predictions.sum() # tp over all house predictions
	recall = tp / gt.sum()       # tp over all houses from ground truth
	F1score = 2*precision*recall / (precision + recall)

	return accuracy, precision, recall, F1score

# Loads and returns the zero padded image
# (TODO redundant code with load_data_simple(W))
def get_zero_padded_img(W):
	im = Image.open(imfile)

	# calculate new size to evenly crop
	w_orig = im.size[0]
	h_orig = im.size[1]

	w_new  = (w_orig + (W - w_orig%W))
	h_new  = (h_orig + (W - h_orig%W))

	# create new image (and gt) with zero padding
	im_new = Image.new(im.mode, (w_new, h_new), (0,0,0,0))
	im_new.paste(im, im.getbbox())

	return im_new


# This function creates an overlaid image with predictions.
# predictions are assumed a numpy array
def save_predictions_image(predictions, imprefix):
	W = predictions.shape[-1]

	# prepare an overlaid image
	zero_padded_im = get_zero_padded_img(W) # get zero padded image
	full_w = zero_padded_im.size[0]
	full_h = zero_padded_im.size[1]
	predicted_fullsize = np.zeros((full_h, full_w))

	# # rebuild a full-size prediction map
	imCounter = 0
	for w in range(0, full_w, W):
		for h in range(0, full_h, W):
			# xpred = np.logical_not(predicted[imCounter, :, :]) # 1 should be house predictions
			xpred = predictions[imCounter, :, :]
			predicted_fullsize[h:h+W, w:w+W] = xpred*255
			imCounter = imCounter+1
			


	mask = Image.fromarray(predicted_fullsize.astype('uint8'), mode='L')
	annot_im = Image.new('RGB', (full_w, full_h), (255, 0, 0))
	zero_padded_im.paste(annot_im, mask=mask)

	# display the result
	zero_padded_im.save("predicted_images/" + imprefix + '.png')
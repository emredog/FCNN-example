import torch
from torch.autograd import Variable

import util

# Predicts the data in the dataloader with the given model
def predict(model, test_loader):

	# Create empty Tensor to store predictions
	all_out = torch.Tensor() 
	all_gt = torch.Tensor()
	all_gt = all_gt.type(torch.LongTensor)
	model.eval()

	for i, (input, target) in enumerate(test_loader):
		# transfer tensors to GPU if possible
		if torch.cuda.is_available():
			#target = target.cuda()
			input = input.cuda()

		input_var = Variable(input, volatile=True) # no need to keep track in validation/test mode
		#target_var = Variable(target, volatile=True) # FIXME is this necessary?

		out = model(input_var) # compute output of the model

		# create a copy of the output. we'll use it for score calculation
		out_copy = out.clone()
		out_copy = out_copy.cpu()

		all_out = torch.cat((all_out, out_copy.data), 0) # append output of the batch onto all_out
		all_gt = torch.cat((all_gt, target))


	# get predictions (class indices, 0: background, 1:house)
	_, predictions = torch.max(all_out, 1)
	accuracy, precision, recall, f1score = pix4d_util.evaluate(predictions, all_gt)
	
	return predictions, accuracy, precision, recall, f1score
	







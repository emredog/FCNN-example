import torch
from torch.autograd import Variable
import torch.utils.data as data_utils

import train
import predict
import util
import FCNN
import FCNN2

import pickle as pkl
import sys


T = 2 # epochs

# set hyperparams
Ns = [1] # batch size 
N_test = 32 # test batch size (will be constant in our experiments)
Ws = [128] # image size

useAugmentedData = True
stride_ratio = 0.5

learning_rates = [1e-4]
lr_decay = 0.5
lr_stepsize = 40
momentums = [0.9]
weight_decays = [5e-3] 
upscale_method = [2] # 1: constant upscaling, 2: "deconv" (or conv transosed to be precise)
class_weights = [torch.Tensor([1.0, 6.0])] 




'''
-------------------------------------------------------------------------------
	TRAINING
-------------------------------------------------------------------------------
'''
exp = 235
for w in Ws:

	# load data & ground truth
	if useAugmentedData:
		training_set = util.load_data_augmented(w, stride_ratio=stride_ratio)
	else:		
		training_set = util.load_data_analyzed(w)

	test_set = util.load_data_simple(w)

	for n in Ns:
	# Create data loaders
		train_loader = data_utils.DataLoader(
			training_set, batch_size=n, shuffle=False, num_workers=1)
		test_loader = data_utils.DataLoader(
			test_set, batch_size=N_test, shuffle=False, num_workers=1)
	

		for u in upscale_method:		
			for lr in learning_rates:
				for m in momentums:
					for wd in weight_decays:
						for cw in class_weights:
												
							torch.manual_seed(1984)
							
							if u == 1:
								model = FCNN.FCNN()
							elif u == 2:
								model = FCNN2.FCNN2()
							else:
								print("WARNING Unexpected upscale method!")	
								continue

							expid = str(exp).zfill(3)
							exp += 1
							print("\n\n", expid, "Model LR:", lr, "Momentum:", m, "WDecay:", wd, "Class Weights:", cw, "Batch size:", n, "Imsize:", w)
							
							model = train.train(expid, model, train_loader, test_loader, n, T, w, lr, m, wd, cw, lr_stepsize, lr_decay)							
							del model


'''
-------------------------------------------------------------------------------
	TEST
-------------------------------------------------------------------------------
'''

model_file = "weights/234_20171213_210829_BEST_F1_Epoch200.pkl"

try:
	with open(model_file, 'rb') as handle:
		weights = pkl.load(handle)
except IOError:
	sys.exit(("Model weights not found: " + model_file))	

model = FCNN2.FCNN2()
model.load_state_dict(weights)
if torch.cuda.is_available():
	model.cuda()
	
model.eval()

predictions, _, _, _, _ = predict.predict(model, test_loader)
util.save_predictions_image(predictions.numpy(), expid)
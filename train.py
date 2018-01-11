'''
[1] Long, J., Shelhamer, E., & Darrell, T. (2015). 
    Fully convolutional networks for semantic segmentation. 
    In Proceedings of the IEEE CVPR (pp. 3431-3440).
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # don't display plots
import pickle as pkl
import numpy as np
import datetime

import predict
import FCNN
import FCNN2
import CrossEntropyLoss2d





def train(expid, model, train_loader, val_loader, batch_size, epochs, W, 
	learning_rate, momentum, weight_decay, class_weights=[1.0, 1.0], 
	lr_stepsize=100000, lr_decay=1.0):
	
	# define loss & create optimizer
	criterion = CrossEntropyLoss2d.CrossEntropy2d()

	# SGD with momentum, following [1]
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

	# learning rate decay
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=lr_decay)
	
	# manually keep track of scores etc.
	losses = []
	precisions = []
	recalls = []
	F1scores = []
	accuracies = []
	best_f1_weights = []
	best_f1_epoch = -1
	best_acc_weights = []
	best_acc_epoch = -1

	# transfer model and other stuff to GPU, if possible
	if torch.cuda.is_available():
		model.cuda()
		class_weights = class_weights.cuda()
		criterion = criterion.cuda()

	# start training loop
	for t in range(epochs):
		print("EPOCH ", t+1, " -----------")
		scheduler.step()
	
		for i, (input, target) in enumerate(train_loader):
			# transfer tensor to GPU if possible
			if torch.cuda.is_available():
				target = target.cuda()
				input = input.cuda()

			input_var = Variable(input)
			target_var = Variable(target)

			output = model(input_var) # compute prediction for batch
			loss = criterion(output, target_var, weight=class_weights) # compute loss
			losses.append(loss.data[0]) # record loss for plotting purposes

			# compute gradient & perform SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('Iteration: {0}/{1} \tLoss: {2}'.format(i+1, len(train_loader), loss.data[0]), end="\r")

		# Epoch completed. Check accuracy over validation set
		_, accuracy, precision, recall, f1score = predict.predict(model, val_loader)
		model.train() # set model back to training mode

		print("\n==> Accuracy:", 100*accuracy, "\tPrecision:", 100*precision, "\tRecall:", 100*recall, "\tF1 Score:", 100*f1score)
		if (t==0 or accuracy > max(accuracies)):
			print("Best weights so far!")
			best_acc_weights = model.state_dict()
			best_acc_epoch = t

		if (t==0 or f1score > max(F1scores)):
			print("Best weights so far!")
			best_f1_weights = model.state_dict()
			best_f1_epoch = t

		# record scores for plotting purposes
		precisions.append(precision)
		recalls.append(recall)
		F1scores.append(f1score)
		accuracies.append(accuracy)

		

	# Training completed.

	# save model weights with best accuracy
	name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	name = expid + "_" + name
	name += "_BEST_ACC_Epoch" + str(best_acc_epoch+1)
	with open('weights/'+ name + '.pkl', 'wb') as f:
		pkl.dump(best_acc_weights, f, pkl.HIGHEST_PROTOCOL)

	print("Model weights with best accuracy are saved to:", "weights/"+name+".pkl, from epoch", best_acc_epoch+1)

	# save model weights with best f1score
	name += "_BEST_F1_Epoch" + str(best_f1_epoch+1)
	with open('weights/'+ name + '.pkl', 'wb') as f:
		pkl.dump(best_f1_weights, f, pkl.HIGHEST_PROTOCOL)

	print("Model weights with best accuracy are saved to:", "weights/"+name+".pkl, from epoch", best_f1_epoch+1)

	# save last model weights

	name += "_LAST_Epoch" + str(epochs)
	with open('weights/'+ name + '.pkl', 'wb') as f:
		pkl.dump(model.state_dict(), f, pkl.HIGHEST_PROTOCOL)

	print("Last model weights saved to:", "weights/"+name+".pkl")

	# visualize training losses
	plt.plot(np.asarray(losses))
	plt.title('Cross Entropy Loss')
	plt.ylabel('loss')
	plt.xlabel('Iterations')
	plt.legend(['loss'], loc='upper right')
	plt.ylim(ymin=0, ymax=10)
	plt.savefig('plots/' + expid + '_losses.png')	
	plt.close()

	# visualize evalution metrics over time 
	plt.figure()
	plt.plot(np.asarray(accuracies))
	plt.plot(np.asarray(precisions))
	plt.plot(np.asarray(recalls))
	plt.plot(np.asarray(F1scores))
	plt.title('Model Scores')
	plt.ylabel('Scores')
	plt.xlabel('Epochs')
	plt.legend(['Accuracy', 'Precision', 'Recall', 'F1 Score'], loc='lower right')
	plt.savefig('plots/' + expid + '_acc.png')
	plt.close()

	return model
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: memo
Migrated to PyTorch

Main app
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import time
import torch
import cv2

import params
import gui

import msa.utils
from msa.capturer import Capturer
from msa.framestats import FrameStats

# PyTorch model imports
import sys
import os
sys.path.append('../pytorch-CycleGAN-and-pix2pix')
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions

class PyTorchPredictor:
	"""PyTorch replacement for the TensorFlow Predictor class"""
	
	def __init__(self, model_path, model_name, epoch='latest'):
		self.device = 'cpu'
		
		# Check for Apple Silicon GPU
		if torch.backends.mps.is_available():
			self.device = 'mps'
			print(f"Using MPS (Apple Silicon GPU)")
		elif torch.cuda.is_available():
			self.device = 'cuda'
			print(f"Using CUDA GPU")
		else:
			print(f"Using CPU")
		
		# Create a simple options object with required attributes
		class SimpleOptions:
			def __init__(self):
				self.num_threads = 0
				self.batch_size = 1
				self.serial_batches = True
				self.no_flip = True
				self.display_id = -1
				self.isTrain = False
				self.model = 'pix2pix'
				self.direction = 'AtoB'
				self.input_nc = 3
				self.output_nc = 3
				self.ngf = 64
				self.ndf = 64
				self.netD = 'basic'
				self.netG = 'unet_256'
				self.n_layers_D = 3
				self.norm = 'batch'
				self.init_type = 'normal'
				self.init_gain = 0.02
				self.no_dropout = False
				self.verbose = False
				self.device = 'cpu'
				self.preprocess = 'resize_and_crop'
				self.load_iter = 0  # Add missing load_iter attribute
		
		opt = SimpleOptions()
		opt.name = model_name
		opt.epoch = epoch
		opt.checkpoints_dir = model_path
		opt.device = self.device  # Set to the detected device
		
		# Load the model
		self.model = Pix2PixModel(opt)
		self.model.setup(opt)
		self.model.eval()
		
		# Move to appropriate device
		if hasattr(self.model, 'netG'):
			self.model.netG.to(self.device)
		
		self.input_shape = (256, 256, 3)  # Expected input shape
		print(f"Model loaded: {model_name} epoch {epoch}")
	
	def preprocess_image(self, img):
		"""Apply the same edge detection used in training"""
		
		# Before edge detection:
		# Reduce noise
# 		img = cv2.bilateralFilter(img, 9, 75, 75)
# 		# Enhance contrast  
# 		img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
		
		if img.shape[:2] != (256, 256):
			img = cv2.resize(img, (256, 256))
		
		# Convert from float [0,1] to uint8 [0,255] if needed
		if img.dtype == np.float32:
			img = (img * 255).astype(np.uint8)
		
		# Apply your trained edge detection pipeline
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 		blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
		# More aggressive blur to match training preprocessing
		blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)  # Instead of (3,3), 0.8
		
		# Multi-scale Canny (same as training)
		edges1 = cv2.Canny(blurred, 40, 120, apertureSize=3)
		edges2 = cv2.Canny(blurred, 80, 160, apertureSize=5)
		edges_combined = cv2.bitwise_or(edges1, edges2)
		
		# Light cleanup
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
		edges_final = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel, iterations=1)
		
		# Convert to RGB
		edges_rgb = cv2.cvtColor(edges_final, cv2.COLOR_GRAY2RGB)
		
		return edges_rgb
	
	def predict(self, img):
		"""PyTorch inference - replaces TensorFlow session.run()"""
		
		# Preprocess: apply edge detection
		edges_img = self.preprocess_image(img)
		
		# Convert to PyTorch tensor
		# Normalize from [0,255] to [-1,1] (pix2pix expects this range)
		tensor = torch.from_numpy(edges_img).permute(2, 0, 1).float()
		tensor = (tensor / 255.0 - 0.5) / 0.5  # [0,255] -> [0,1] -> [-1,1]
		tensor = tensor.unsqueeze(0)  # Add batch dimension
		
		# Move to device
		tensor = tensor.to(self.device)
		
		# Run inference
		with torch.no_grad():
			fake = self.model.netG(tensor)
		
		# Convert back to numpy image
		output = fake[0].cpu().permute(1, 2, 0).numpy()
		output = (output + 1) * 127.5  # [-1,1] -> [0,255]
		output = np.clip(output, 0, 255).astype(np.uint8)
		
		# Convert BGR to RGB for display (if needed)
		# output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
		
		# Debug: print tensor stats before inference
		print(f"Input tensor shape: {tensor.shape}")
		print(f"Input tensor range: {tensor.min():.3f} to {tensor.max():.3f}")

		# Run inference
		with torch.no_grad():
			fake = self.model.netG(tensor)

		# In the predict() method, after conversion:
# 		output = (fake[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5
# 		output = np.clip(output, 0, 255).astype(np.uint8)

		# Debug actual values
# 		print(f"Output shape: {output.shape}")
# 		print(f"Output range: {output.min()} to {output.max()}")
# 		print(f"Output dtype: {output.dtype}")

		# Check a few pixel values
# 		print(f"Corner pixels: {output[0,0]}, {output[0,-1]}, {output[-1,0]}, {output[-1,-1]}")

		# Save a test image to disk to see what the model is actually generating
# 		cv2.imwrite('debug_model_output.png', output)
# 		print("Saved debug_model_output.png")		

		# Convert back to numpy image  
		output = fake[0].cpu().permute(1, 2, 0).numpy()
		output = (output + 1) * 127.5  # [-1,1] -> [0,255]
		output = np.clip(output, 0, 255).astype(np.uint8)

		# Convert to float [0,1] to match capturer format
		output = output.astype(np.float32) / 255.0

		return [output]  # Return as list to match original interface


#%%
capture = None # msa.capturer.Capturer, video capture wrapper
predictor = None # PyTorchPredictor, model for prediction

img_cap = np.empty([]) # captured image before processing
img_in = np.empty([]) # processed capture image
img_out = np.empty([]) # output from prediction model

#%% init gui and params

gui.init_app()

pyqt_params = gui.init_params(params.params_list, target_obj=params, w=320)

# reading & writing to pyqtgraph.parametertree seems to be slow,
# so going to cache in an object for direct access
gui.params_to_obj(pyqt_params, target_obj=params, create_missing=True, verbose=True)

# create main window
gui.init_window(x=320, w=(gui.screen_size().width()-320), h=int((gui.screen_size().width()-320)*0.4))

#%%

# Load PyTorch predictor model
predictor = PyTorchPredictor(
	model_path='./models',  # Path to your checkpoints folder
	model_name='cactus_clean',   # Your model name
	epoch='latest'				   # Or 'latest'
)

# init capture device
def init_capture(capture, output_shape):
	if capture:
		capture.close()
		
	capture_shape = (params.Capture.Init.height, params.Capture.Init.width)
	capture = Capturer(sleep_s = params.Capture.sleep_s,
					   device_id = params.Capture.Init.device_id,
					   capture_shape = capture_shape,
					   capture_fps = params.Capture.Init.fps,
					   output_shape = output_shape
					   )
	
	capture.update()
	
	if params.Capture.Init.use_thread:
		capture.start()
	
	return capture

capture = init_capture(capture, output_shape=predictor.input_shape)

# keep track of frame count and frame rate
frame_stats = FrameStats('Main')

# main update loop
while not params.Main.quit:
	
	# reinit capture device if parameters have changed
	if params.Capture.Init.reinitialise:
		params.child('Capture').child('Init').child('reinitialise').setValue(False)
		capture = init_capture(capture, output_shape=predictor.input_shape)
		
	capture.enabled = params.Capture.enabled
	if params.Capture.enabled:
		# update capture parameters from GUI
		capture.output_shape = predictor.input_shape
		capture.verbose = params.Main.verbose
		capture.freeze = params.Capture.freeze
		capture.sleep_s = params.Capture.sleep_s
		for p in msa.utils.get_members(params.Capture.Processing):
			setattr(capture, p, getattr(params.Capture.Processing, p))
		
		# run capture if multithreading is disabled
		if params.Capture.Init.use_thread == False:
			capture.update()
			
		img_cap = np.copy(capture.img) # create copy to avoid thread issues

	# interpolate (temporal blur) on input image
	img_in = msa.utils.np_lerp(img_in, img_cap, 1 - params.Prediction.pre_time_lerp)

	# run prediction
	if params.Prediction.enabled and predictor:
		img_predicted = predictor.predict(img_in)[0]
	else:
		img_predicted = capture.img0

	# interpolate (temporal blur) on output image
	img_out = msa.utils.np_lerp(img_out, img_predicted, 1 - params.Prediction.post_time_lerp)

	# update frame states
	frame_stats.verbose = params.Main.verbose
	frame_stats.update()
	
	# DEBUG DISPLAY
	# In the main loop, before gui.update_image calls:
# 	print(f"img_cap shape: {img_cap.shape}, dtype: {img_cap.dtype}, range: {img_cap.min():.3f}-{img_cap.max():.3f}")
# 	print(f"img_in shape: {img_in.shape}, dtype: {img_in.dtype}, range: {img_in.min():.3f}-{img_in.max():.3f}")  
# 	print(f"img_out shape: {img_out.shape}, dtype: {img_out.dtype}, range: {img_out.min():.3f}-{img_out.max():.3f}")
	
	# update gui
	gui.update_image(0, capture.img0)
	gui.update_image(1, img_in)
	gui.update_image(2, img_out)
	gui.update_stats(frame_stats.str + "   |   " + capture.frame_stats.str)
	gui.process_events()

	time.sleep(params.Main.sleep_s)

# cleanup
capture.close()
gui.close()

capture = None
predictor = None
	
print('Finished')
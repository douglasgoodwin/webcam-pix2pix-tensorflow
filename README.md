## Pytorch implementation, Fall 2025.

## Overview of Changes

This repository has been migrated from TensorFlow 1.x to PyTorch with Apple Silicon (M-series) GPU support. The migration enables modern deep learning workflows while maintaining the original real-time webcam functionality.

## Key Changes Made

### Core Architecture

- **Replaced TensorFlow Predictor** with PyTorchPredictor class
- **Added Apple Silicon GPU support** via Metal Performance Shaders (MPS)
- **Updated all files to Python 3** compatibility
- **Integrated edge detection** into the predictor for consistency with training

### File Modifications

**webcam-pix2pix.py** (Major rewrite):

- New PyTorchPredictor class with MPS/CUDA/CPU device detection
- SimpleOptions class to bypass command line argument parsing
- Integrated edge detection preprocessing matching training pipeline
- Proper tensor format handling for real-time inference

**params.py** (Updated):

- Removed duplicate parameters that caused conflicts
- Added EdgeDetection section with training-matched defaults
- Updated parameter structure for PyTorch workflow

**gui.py** (Minor updates):

- Python 3 compatibility
- Updated window titles

**msa/ modules** (Python 3 compatibility):

- capturer.py: Updated threading and camera handling
- framestats.py: Switched to time.perf_counter() for better precision
- utils.py: Python 3 compatibility maintained



## Setup Instructions for Mac ARM (M-series) Chips

### Prerequisites

bash

```bash
# Create virtual environment
python3 -m venv pix2pix_env
source pix2pix_env/bin/activate

# Clone both repositories
git clone https://github.com/douglasgoodwin/pytorch-CycleGAN-and-pix2pix.git
git clone https://github.com/douglasgoodwin/webcam-pix2pix-tensorflow.git
```

### Install Dependencies

bash

```bash
# Install PyTorch with Apple Silicon support
pip install torch torchvision

# Install other requirements
cd webcam-pix2pix-tensorflow
pip install -r requirements.txt
```

### Model Training

bash

```bash
cd pytorch-CycleGAN-and-pix2pix

# Train your model
python train.py \
  --dataroot ../your_training_dataset \
  --name your_model_name \
  --model pix2pix \
  --direction AtoB \
  --batch_size 1 \
  --n_epochs 100 \
  --n_epochs_decay 100
```

### Using the Webcam App

1. **Update model paths** in webcam-pix2pix.py:

python

```python
predictor = PyTorchPredictor(
    model_path='./pytorch-CycleGAN-and-pix2pix/checkpoints',
    model_name='your_model_name',
    epoch='latest'  # or specific epoch like '50'
)
```

1. **Run the webcam application:**

bash

```bash
cd webcam-pix2pix-tensorflow
python webcam-pix2pix.py
```

1. **Enable edge detection** in the GUI:
   - In the parameter window, navigate to Capture → Processing
   - Check the "canny" option
   - Adjust thresholds as needed for your lighting conditions



## Critical Configuration for ARM Macs

### GPU Acceleration

The app automatically detects and uses Apple Silicon GPU via MPS. You should see:

```
Using MPS (Apple Silicon GPU)
Initialized with device mps
```

### Edge Detection Consistency

The webcam edge detection must match your training preprocessing. Default parameters are set to match typical training configurations:

- Canny thresholds: 40/120 (fine), 80/160 (coarse)
- Gaussian blur: kernel=3, sigma=0.8
- Multi-scale edge combination

### Camera Settings

For optimal results on Mac ARM:

- The app sets manual exposure and disables autofocus
- Higher resolution capture (1280x720) downscaled to 256x256
- 30fps capture with real-time processing

## Troubleshooting

### Resuming Interrupted Training

bash

```bash
python train.py \
  --continue_train \
  --epoch_count [LAST_EPOCH + 1] \
  --dataroot ../your_dataset \
  --name your_model_name \
  --model pix2pix \
  --direction AtoB
```

### Edge Detection Mismatch

If webcam results don't match training quality:

- Ensure "canny" parameter is enabled in GUI
- Verify preprocessing parameters match training exactly
- Adjust Canny thresholds for your lighting conditions

### GPU Not Detected

bash

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Performance Notes

- **Training time on M4 Mac:** ~3-6 hours for 200 epochs (3000+ images)
- **Real-time inference:** 15-30 fps depending on model complexity
- **Memory usage:** ~2-4GB GPU memory for typical pix2pix models

## Model Compatibility

The PyTorch implementation loads models trained with the pytorch-CycleGAN-and-pix2pix framework. Models are expected in this structure:

```
checkpoints/model_name/
├── latest_net_G.pth
├── latest_net_D.pth
├── [epoch]_net_G.pth
└── [epoch]_net_D.pth
```

The webcam app uses only the Generator (net_G.pth) for real-time inference.



---

# Data Preparation and Training Guidelines

## Video Source Requirements

### Source Material

- **Duration:** 5-30 minutes of footage for good results
- **Content:** Consistent visual domain (e.g., desert gardens, urban scenes, portraits)
- **Quality:** 720p or higher resolution recommended
- **Frame rate:** 24-60fps (will be downsampled during extraction)
- **Lighting:** Varied but consistent conditions within your domain

### Frame Extraction

**Extract frames using ffmpeg:**

bash

```bash
# From 30fps video to 12fps training frames
ffmpeg -i source_video.mp4 -vf "fps=12" frames/frame_%06d.png

# Alternative: Extract every Nth frame for more variety
ffmpeg -i source_video.mp4 -vf "select='not(mod(n\,5))'" -vsync vfr frames/frame_%06d.png

# For different output frame rates:
ffmpeg -i source_video.mp4 -vf "fps=1" frames/frame_%06d.png    # 1 fps (sparse)
ffmpeg -i source_video.mp4 -vf "fps=24" frames/frame_%06d.png   # 24 fps (dense)
```

## Dataset Size Guidelines

### Frame Count Recommendations

- **Minimum viable:** 1,000-2,000 frames
- **Good results:** 3,000-5,000 frames
- **Professional quality:** 5,000+ frames
- **Example:** 3,580 frames from 5 minutes of 30fps footage extracted at 12fps

### Calculation Examples

bash

```bash
# 10 minutes at 30fps → extract at 2fps = 1,200 frames
# 15 minutes at 24fps → extract at 4fps = 3,600 frames  
# 5 minutes at 60fps → extract at 12fps = 3,600 frames
```

### Train/Validation Split

The preprocessing script automatically creates:

- **90% training data** (e.g., 3,222 images from 3,580 total)
- **10% validation data** (e.g., 358 images from 3,580 total)

## Data Processing Pipeline

### Create Training Dataset

bash

```bash
# Run the preprocessing script
python create_training_dataset.py

# This creates:
training_dataset/
├── train/          # 90% of frames (edge|original pairs)
├── val/           # 10% of frames (edge|original pairs)
```

### Edge Detection Parameters

The preprocessing uses these critical parameters:

python

```python
# Multi-scale Canny edge detection
edges1 = cv2.Canny(blurred, 40, 120, apertureSize=3)   # Fine details
edges2 = cv2.Canny(blurred, 80, 160, apertureSize=5)   # Main structure
blur = cv2.GaussianBlur(gray, (3, 3), 0.8)             # Noise reduction
```

**Note:** These exact parameters must be replicated in the webcam app for consistent results.

## Training Configuration

### Epoch Recommendations

bash

```bash
# Standard training schedule
python train.py \
  --dataroot ../training_dataset \
  --name your_model \
  --model pix2pix \
  --direction AtoB \
  --n_epochs 100 \      # Linear learning rate phase
  --n_epochs_decay 100  # Decay learning rate phase
```

**Total epochs:** 200 (100 + 100)

- **Epochs 1-20:** Basic structure learning
- **Epochs 20-50:** Feature refinement
- **Epochs 50-100:** Quality improvement
- **Epochs 100-200:** Fine-tuning with learning rate decay

### Early Results Timeline

- **Epoch 5:** Basic shapes and colors appear
- **Epoch 15:** Recognizable scene generation
- **Epoch 25:** Good quality for testing
- **Epoch 50+:** Production-ready results

### Training Time Estimates

**Apple Silicon (M4/M3/M2) with MPS:**

- 3,000 frames: ~3-4 hours for 200 epochs
- 5,000 frames: ~5-6 hours for 200 epochs
- 10,000 frames: ~8-12 hours for 200 epochs

**CPU only (not recommended):**

- Add 3-4x to the above times

### Batch Size Considerations

bash

```bash
# For Apple Silicon Macs
--batch_size 1    # Recommended for stability
--batch_size 4    # If you have 32GB+ unified memory
```

**Memory usage:**

- Batch size 1: ~2-4GB GPU memory
- Batch size 4: ~6-12GB GPU memory

### Storage Requirements

- **Raw frames:** ~50-200MB per 1000 frames
- **Processed dataset:** ~100-400MB per 1000 frames (includes edge pairs)
- **Model checkpoints:** ~200MB per saved epoch
- **Training logs/web:** ~50-100MB

## Quality Optimization

### Frame Selection Strategy

**Maximize visual diversity:**

- Different lighting conditions within your domain
- Various angles and compositions
- Range of subjects within your theme
- Avoid near-duplicate frames

**Bad extraction rates:**

- Too sparse (fps=0.5): May miss important transitions
- Too dense (fps=30): Many duplicate/similar frames waste training time

**Optimal extraction:**

- **Static scenes:** 1-2 fps
- **Dynamic content:** 5-12 fps
- **Mixed content:** 3-8 fps

### Domain Consistency

Train on visually cohesive content:

- **Good:** All desert garden footage
- **Good:** All urban architecture
- **Good:** All portrait photography
- **Bad:** Mixed domains (gardens + cities + portraits)

The model learns a specific visual language from your training domain. Mixing drastically different visual styles dilutes the learning and produces poor results.

### Validation During Training

Monitor progress at:

- `checkpoints/your_model/web/index.html`
- Check sample outputs every 10-20 epochs
- Look for overfitting after epoch 100+

Training can be stopped early if results plateau or if you're satisfied with quality at epoch 50-100.





---

## Memo's original README

This is the source code and pretrained model for the webcam pix2pix demo I posted recently on [twitter](https://twitter.com/memotv/status/858397873712623616) and vimeo. It uses deep learning, or to throw in a few buzzwords: *deep convolutional conditional generative adversarial network autoencoder*. 

[![video 1](https://cloud.githubusercontent.com/assets/144230/25585045/9b932e50-2e90-11e7-9bb2-692ef9629f0a.png)
*video 1*
](https://vimeo.com/215339817)

[![video 2](https://cloud.githubusercontent.com/assets/144230/25584635/b67b0bea-2e8e-11e7-8b12-f8356241728b.png)
*video 2*
](https://vimeo.com/215514169)



# Overview
The code in this particular repo actually has nothing to do with pix2pix, GANs or even deep learning. It just loads *any* pre-trained tensorflow model (as long as it complies with a few constraints), feeds it a processed webcam input, and displays the output of the model. It just so happens that the model I trained and used is pix2pix (details below). 

I.e. The steps can be summarised as:

1. Collect data: scrape the web for a ton of images, preprocess and prepare training data
2. Train and export a model
3. Preprocessing and prediction: load pretrained model, feed it live preprocessed webcam input, display the results. 

# 1. Data
I scraped art collections from around the world from the [Google Art Project on wikimedia](https://commons.wikimedia.org/wiki/Category:Google_Art_Project_works_by_collection). A **lot** of the images are classical portraits of rich white dudes, so I only used about 150 collections, trying to keep the data as geographically and culturally diverse as possible (full list I used is [here](./gart_canny_256_info/collections.txt)). But the data is still very euro-centric, as there might be hundreds or thousands of scans from a single European museum, but only 8 scans  from an Arab museum. 

I downloaded the 300px versions of the images, and ran a batch process to :

- Rescale them to 256x256 (without preserving aspect ratio)
- Run a a simple edge detection filter (opencv canny)

I also ran a batch process to take multiple crops from the images (instead of a non-uniform resizing) but I haven't trained on that yet. Instead of canny edge detection, I also started looking into the much better  'Holistically-Nested Edge Detection' (aka [HED](https://github.com/s9xie/hed)) by Xie and Tu (as used by the original pix2pix paper), but haven't trained on that yet either. 

This is done by the [preprocess.py](preprocess.py) script (sorry no command line arguments, edit the script to change paths and settings, should be quite self-explanatory).


**A small sample of the training data - including predictions of the trained model - can be seen [here](http://memo.tv/gart_canny_256_pix2pix/).**
Right-most column is the original image, left-most column is the preprocessed version. These two images are fed into the pix2pix network as a 'pair' to be trained on. The middle column is what the model learns to produce *given only the left-most column*. (The images show each training iteration - i.e. the number on the left, which goes from 20,000 to 58,000, so it gradually gets better the further down you go on the page). 

[![training_data](https://cloud.githubusercontent.com/assets/144230/25617554/bd2f3c16-2f3a-11e7-9e25-75792fbc3380.png)](http://memo.tv/gart_canny_256_pix2pix/)


I also trained an unconditional GAN (i.e. normal [DCGAN](https://github.com/Newmu/dcgan_code) on this same training data. An example of its output can be seen below. (This is generating 'completely random' images that resemble the training data). 

![dcgan](https://cloud.githubusercontent.com/assets/144230/25617262/58c9dc46-2f39-11e7-97b9-d546cc6cc00c.png)


# 2. Training
The training and architecture is straight up '*Image-to-Image Translation with Conditional Adversarial Nets*' by Isola et al (aka [pix2pix](https://phillipi.github.io/pix2pix/)). I trained with the [tensorflow port](https://github.com/affinelayer/pix2pix-tensorflow) by @affinelayer (Christopher Hesse), which is also what's powering that '[sketch-to-cat](https://affinelayer.com/pixsrv/)'- demo that went viral recently. He also wrote a nice [tutorial](https://affinelayer.com/pix2pix/) on how pix2pix works. Infinite thanks to the authors (and everyone they built on) for making their code open-source!

I only made one infinitesimally tiny change to the tensorflow-pix2pix training code, and that is to add *tf.Identity* to the generator inputs and outputs with a human-readable name, so that I can feed and fetch the tensors with ease. **So if you wanted to use your own models with this application, you'd need to do the same**. (Or make a note of the input/output tensor names, and modify the json accordingly, more on this below). 

**You can download my pretrained model from the [Releases tab](https://github.com/memo/webcam-pix2pix-tensorflow/releases).**

![pix2pix_diff](https://cloud.githubusercontent.com/assets/144230/25583118/4e4f9794-2e88-11e7-8762-889e4113d0b8.png)

# 3. Preprocessing and prediction
What this particular application does is load the pretrained model, do live preprocessing of a webcam input, and feed it to the model. I do the preprocessing with old fashioned basic computer vision, using opencv. It's really very minimal and basic. You can see the GUI below (the GUI uses [pyqtgraph](http://www.pyqtgraph.org/)).

![ruby](https://cloud.githubusercontent.com/assets/144230/25586317/b3f4e65e-2e96-11e7-809d-5a6296d2ed64.png)

Different scenes require different settings.

E.g. for 'live action' I found **canny** to provide better (IMHO) results, and it's what I used in the first video at the top. The thresholds (canny_t1, canny_t2) depend on the scene, amount of detail, and the desired look. 

If you have a lot of noise in your image you may want to add a tiny bit of **pre_blur** or **pre_median**. Or play with them for 'artistic effect'. E.g. In the first video, at around 1:05-1:40, I add a ton of median (values around 30-50).

For drawing scenes (e.g. second video) I found **adaptive threshold** to give more interesting results than canny (i.e. disable canny and enable adaptive threshold), though you may disagree. 

For a completely *static input* (i.e. if you **freeze** the capture, disabling the camera update) the output is likely to flicker a very small amount as the model makes different predictions for the same input - though this is usually quite subtle. However for a *live* camera feed, the noise in the input is likely to create lots of flickering in the output, especially due to the high susceptibility of canny or adaptive threshold to noise, so some temporal blurring can help. 

**accum_w1** and **accum_w2** are for temporal blurring of the input, before going into the model:
new_image = old_image * w1 + new_image * w2 (so ideally they should add up to one - or close to). 

**Prediction.pre_time_lerp** and **post_time_lerp** also do temporal smoothing:
new_image = old_image * xxx_lerp + new_image * (1 - xxx_lerp)
pre_time_lerp is before going into the model, and post_time_lerp is after coming out of the model. 

Zero for any of the temporal blurs disables them. Values for these depend on your taste. For both of the videos above I had all of pre_model blurs (i.e. accum_w1, accum_w2 and pre_time_lerp)  set to zero, and played with different post_time_lerp settings ranging from 0.0 (very flickery and flashing) to 0.9 (very slow and fadey and 'dreamy'). Usually around 0.5-0.8 is my favourite range. 

# Using other models
If you'd like to use a different model, you need to setup a JSON file similar to the one below. 
The motivation here is that I actually have a bunch of JSONs in my app/models folder which I can dynamically scan and reload, and the model data is stored elsewhere on other disks, and the app can load and swap between models at runtime and scale inputs/outputs etc automatically. 

	{
		"name" : "gart_canny_256", # name of the model (for GUI)
		"ckpt_path" : "./models/gart_canny_256", # path to saved model (meta + checkpoints). Loads latest if points to a folder, otherwise loads specific checkpoint
		"input" : { # info for input tensor
			"shape" : [256, 256, 3],  # expected shape (height, width, channels) EXCLUDING batch (assumes additional axis==0 will contain batch)
			"range" : [-1.0, 1.0], # expected range of values 
			"opname" : "generator/generator_inputs" # name of tensor (':0' is appended in code)
		},
		"output" : { # info for output tensor
			"shape" : [256, 256, 3], # shape that is output (height, width, channels) EXCLUDING batch (assumes additional axis==0 will contain batch)
			"range" : [-1.0, 1.0], # value range that is output
			"opname" : "generator/generator_outputs" # name of tensor (':0' is appended in code)
		}
	}


# Requirements
- python 2.7 (likely to work with 3.x as well)
- tensorflow 1.0+
- opencv 3+ (probably works with 2.4+ as well)
- pyqtgraph (only tested with 0.10)

Tested only on Ubuntu 16.04, but should work on other platforms. 

I use the Anaconda python distribution which comes with almost everything you need, then it's (hopefully) as simple as:
1. Download and install anaconda from https://www.continuum.io/downloads
2. Install tensorflow https://www.tensorflow.org/install/ (Which - if you have anaconda - is often quite straight forward since most dependencies are included)
3. Install opencv and pyqtgraph

	conda install -c menpo opencv3
	conda install pyqtgraph
   
    
   
# Acknowledgements
Infinite thanks once again to

* Isola et al for [pix2pix](https://phillipi.github.io/pix2pix/) and @affinelayer (Christopher Hesse) for the [tensorflow port](https://github.com/affinelayer/pix2pix-tensorflow)
* Radford et al for [DCGAN](https://github.com/Newmu/dcgan_code) and @carpedm20 (Taehoon Kim) for the [tensorflow port](https://github.com/carpedm20/DCGAN-tensorflow)
* The [tensorflow](https://www.tensorflow.org/) team
* Countless others who have contributed to the above, either directly or indirectly, or opensourced their own research making the above possible
* My [wife](http://janelaurie.com/) for putting up with me working on a bank holiday to clean up my code and upload this repo. 
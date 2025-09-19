#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: memo
Updated for PyTorch pix2pix

params_list used to initialise a pyqtgraph.parametertree
http://www.pyqtgraph.org/documentation/parametertree/

this module will be populated and updated with parameters from params_list

Avoid double processing with Pytorch:
params.Capture.Processing.canny = False
params.Capture.Processing.adaptive_thresh = False
"""

params_list = [
    {'name': 'Main', 'type': 'group', 'children': [
        {'name': 'quit', 'type': 'bool', 'value': False},
        {'name': 'verbose', 'type': 'bool', 'value': False},
        {'name': 'sleep_s', 'type': 'float', 'value': 0.001, 'limits': (0, 10)},
    ]},
    
    {'name': 'Model', 'type': 'group', 'children': [
        {'name': 'model_name', 'type': 'str', 'value': 'cactus_clean'},
        {'name': 'epoch', 'type': 'str', 'value': '15'},
        {'name': 'device', 'type': 'list', 'values': ['auto', 'mps', 'cuda', 'cpu'], 'value': 'auto'},
        {'name': 'reload_model', 'type': 'bool', 'value': False},
    ]},
    
    {'name': 'Capture', 'type': 'group', 'children': [
        {'name': 'enabled', 'type': 'bool', 'value': True},
        {'name': 'freeze', 'type': 'bool', 'value': False},
        {'name': 'sleep_s', 'type': 'float', 'value': 0.001, 'limits': (0, 10)},
        {'name': 'Init', 'type': 'group', 'children': [
            {'name': 'use_thread', 'type': 'bool', 'value': True},
            {'name': 'device_id', 'type': 'int', 'value': 0},
            {'name': 'width', 'type': 'int', 'value': 640},
            {'name': 'height', 'type': 'int', 'value': 480},
            {'name': 'fps', 'type': 'int', 'value': 30},
            {'name': 'reinitialise', 'type': 'bool', 'value': False},
        ]},
        {'name': 'Processing', 'type': 'group', 'children': [
            {'name': 'canny', 'type': 'bool', 'value': False},  # Changed from True
            {'name': 'adaptive_thresh', 'type': 'bool', 'value': False},  # Keep False
            {'name': 'frame_diff', 'type': 'bool', 'value': False},
            {'name': 'flip_h', 'type': 'bool', 'value': True},
            {'name': 'flip_v', 'type': 'bool', 'value': False},
            {'name': 'grayscale', 'type': 'bool', 'value': False},
            {'name': 'pre_blur', 'type': 'int', 'value': 0},
            {'name': 'pre_median', 'type': 'int', 'value': 1},
            {'name': 'pre_thresh', 'type': 'int', 'value': 0},
            {'name': 'invert', 'type': 'bool', 'value': False},
            {'name': 'accum_w1', 'type': 'float', 'limits': (0, 1), 'value':0, 'step': 0.1},
            {'name': 'accum_w2', 'type': 'float', 'limits': (0, 1), 'value':0, 'step': 0.1},
            {'name': 'post_thresh', 'type': 'int', 'value': 0},
        ]},
    ]},

    {'name': 'EdgeDetection', 'type': 'group', 'children': [
        {'name': 'method', 'type': 'list', 'values': ['training_matched', 'custom_canny', 'adaptive'], 'value': 'training_matched'},
        {'name': 'blur_kernel', 'type': 'int', 'value': 3, 'limits': (1, 15), 'step': 2},
        {'name': 'blur_sigma', 'type': 'float', 'value': 0.8, 'limits': (0.1, 3.0), 'step': 0.1},
        {'name': 'canny_t1_fine', 'type': 'int', 'value': 40, 'limits': (1, 300)},
        {'name': 'canny_t2_fine', 'type': 'int', 'value': 120, 'limits': (1, 300)},
        {'name': 'canny_t1_coarse', 'type': 'int', 'value': 80, 'limits': (1, 300)},
        {'name': 'canny_t2_coarse', 'type': 'int', 'value': 160, 'limits': (1, 300)},
        {'name': 'morph_kernel_size', 'type': 'int', 'value': 2, 'limits': (1, 10)},
        {'name': 'morph_iterations', 'type': 'int', 'value': 1, 'limits': (0, 5)},
    ]},

    {'name': 'Prediction', 'type': 'group', 'children': [
        {'name': 'enabled', 'type': 'bool', 'value': True},
        {'name': 'pre_time_lerp', 'type': 'float', 'limits': (0, 1), 'value':0, 'step': 0.1},
        {'name': 'post_time_lerp', 'type': 'float', 'limits': (0, 1), 'value':0.5, 'step': 0.1},
        {'name': 'show_edges', 'type': 'bool', 'value': False},
    ]},
]
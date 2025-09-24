#!/usr/bin/env python3
"""
Process video files with trained pix2pix model
Creates side-by-side comparison: original | edges | generated
"""

import cv2
import torch
import numpy as np
import os
import sys

# Add PyTorch repo to path
sys.path.append('../pytorch-CycleGAN-and-pix2pix')
from models.pix2pix_model import Pix2PixModel

class VideoProcessor:
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
        
        # Create options object
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
                self.load_iter = 0
        
        opt = SimpleOptions()
        opt.name = model_name
        opt.epoch = epoch
        opt.checkpoints_dir = model_path
        opt.device = self.device
        
        # Load the model
        self.model = Pix2PixModel(opt)
        self.model.setup(opt)
        self.model.eval()
        
        # Move to appropriate device
        if hasattr(self.model, 'netG'):
            self.model.netG.to(self.device)
        
        print(f"Model loaded: {model_name} epoch {epoch}")
    
    def preprocess_frame(self, frame):
        """Apply edge detection matching training pipeline"""
        # Resize to 256x256
        frame = cv2.resize(frame, (256, 256))
        
        # Apply edge detection (same as training)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        
        # Multi-scale Canny
        edges1 = cv2.Canny(blurred, 40, 120, apertureSize=3)
        edges2 = cv2.Canny(blurred, 80, 160, apertureSize=5)
        edges_combined = cv2.bitwise_or(edges1, edges2)
        
        # Light cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_final = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Convert to RGB
        edges_rgb = cv2.cvtColor(edges_final, cv2.COLOR_GRAY2RGB)
        
        return edges_rgb
    
    def generate_from_edges(self, edges_img):
        """Run pix2pix inference"""
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(edges_img).permute(2, 0, 1).float()
        tensor = (tensor / 255.0 - 0.5) / 0.5  # [0,255] -> [-1,1]
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
        
        return output
    
    def process_video(self, input_video, output_video, output_size=512, max_frames=None, debug=False):
        """Process video with simple upscaling (Option 1)"""
        
        cap = cv2.VideoCapture(input_video)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Output resolution: {output_size}x{output_size} (upscaled)")
        print(f"Debug mode: {'ON - 3-up comparison' if debug else 'OFF - processed only'}")
        
        # Set up output video dimensions
        if debug:
            output_width = output_size * 3  # 3-up comparison
        else:
            output_width = output_size       # Generated only
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_size))
        
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and processed_count >= max_frames:
                break
            
            # Generate edges at 256x256 (model's native resolution)
            edges = self.preprocess_frame(frame)
            
            # Generate output at 256x256
            generated = self.generate_from_edges(edges)
            generated_bgr = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)
            
            # Upscale generated to target resolution
            generated_upscaled = cv2.resize(generated_bgr, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
            
            if debug:
                # Create 3-up comparison: original | edges | generated
                original_resized = cv2.resize(frame, (output_size, output_size))
                edges_upscaled = cv2.resize(edges, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
                combined = np.hstack([original_resized, edges_upscaled, generated_upscaled])
                out.write(combined)
            else:
                # Output only the generated video
                out.write(generated_upscaled)
            
            processed_count += 1
            
            if processed_count % 30 == 0:
                print(f"Processed {processed_count}/{total_frames if not max_frames else max_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"Upscaled video processing complete: {output_video}")
        print(f"Processed {processed_count} frames")

    def process_temporal_collage(self, input_video, output_video, target_size=1536, time_offset=30, max_frames=None, debug=False):
        """
        Create Hockney-style temporal collage using different video frames for each tile
        Each tile shows the scene from a different moment in time
        """
        cap = cv2.VideoCapture(input_video)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Creating temporal collage at {target_size}x{target_size}")
        print(f"Time offset: {time_offset} frames between tiles")
        print(f"Debug mode: {'ON - 3-up comparison' if debug else 'OFF - processed only'}")
        
        # Calculate tile configuration
        tile_size = 256
        tiles_per_side = target_size // tile_size
        tiles_total = tiles_per_side ** 2
        
        # Set up output video dimensions
        if debug:
            output_width = target_size * 3  # 3-up comparison
        else:
            output_width = target_size       # Generated only
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, target_size))
        
        processed_count = 0
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
                
            if max_frames and processed_count >= max_frames:
                break
            
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Collect frames for temporal tiles
            temporal_frames = []
            
            # Get frames at different time offsets
            for tile_idx in range(tiles_total):
                # Calculate target frame number with time offset
                target_frame = current_frame_num + (tile_idx * time_offset)
                target_frame = target_frame % total_frames  # Wrap around if needed
                
                # Seek to target frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret_temp, temp_frame = cap.read()
                
                if ret_temp:
                    temporal_frames.append(temp_frame)
                else:
                    temporal_frames.append(current_frame)  # Fallback
            
            # Reset to current position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            
            # Process temporal collage
            edges_full, generated_full = self.process_temporal_tiles(
                temporal_frames, current_frame, target_size
            )
            
            # Convert generated to BGR for video output
            generated_bgr = cv2.cvtColor(generated_full, cv2.COLOR_RGB2BGR)
            
            if debug:
                # Create 3-up comparison: original | edges | generated
                original_resized = cv2.resize(current_frame, (target_size, target_size))
                combined = np.hstack([original_resized, edges_full, generated_bgr])
                out.write(combined)
            else:
                # Output only the generated temporal collage
                out.write(generated_bgr)
            
            processed_count += 1
            
            if processed_count % 5 == 0:
                print(f"Processed {processed_count} temporal collage frames")
        
        cap.release()
        out.release()
        
        print(f"Temporal collage complete: {output_video}")

    def process_temporal_tiles(self, temporal_frames, reference_frame, target_size):
        """
        Process tiles using different temporal source frames
        Creates fragmented time effect like Hockney's photocollages
        """
        tile_size = 256
        tiles_per_side = target_size // tile_size
        
        edges_full = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        generated_full = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        print(f"Creating temporal collage with {len(temporal_frames)} time fragments")
        
        for row in range(tiles_per_side):
            for col in range(tiles_per_side):
                tile_idx = row * tiles_per_side + col
                
                # Use different temporal source for each tile
                source_frame = temporal_frames[tile_idx % len(temporal_frames)]
                source_resized = cv2.resize(source_frame, (target_size, target_size))
                
                # Extract tile from temporal source
                y1, y2 = row * tile_size, (row + 1) * tile_size
                x1, x2 = col * tile_size, (col + 1) * tile_size
                tile = source_resized[y1:y2, x1:x2]
                
                # Process through model
                edges_tile = self.preprocess_frame(tile)
                generated_tile = self.generate_from_edges(edges_tile)
                
                # Place in output
                edges_full[y1:y2, x1:x2] = edges_tile
                generated_full[y1:y2, x1:x2] = generated_tile
        
        return edges_full, generated_full

    def process_frame_tiled(self, frame, target_size=1024):
        """
        Process frame at high resolution using tiling approach
        Splits image into 256x256 tiles, processes each, then reassembles
        """
        # Resize input to target size
        frame_resized = cv2.resize(frame, (target_size, target_size))
        
        # Calculate tile configuration
        tile_size = 256
        tiles_per_side = target_size // tile_size
        
        print(f"Processing {tiles_per_side}x{tiles_per_side} tiles at {tile_size}x{tile_size}")
        
        # Initialize output arrays
        edges_full = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        generated_full = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Process each tile
        for row in range(tiles_per_side):
            for col in range(tiles_per_side):
                # Show progress for large grids
                tile_num = row * tiles_per_side + col + 1
                if tiles_per_side >= 5:  # Show progress for 5x5 and larger
                    print(f"  Processing tile {tile_num}/{tiles_per_side**2} (row {row+1}, col {col+1})")
                
                # Extract tile
                y1, y2 = row * tile_size, (row + 1) * tile_size
                x1, x2 = col * tile_size, (col + 1) * tile_size
                tile = frame_resized[y1:y2, x1:x2]
                
                # Ensure tile is exactly 256x256
                if tile.shape[:2] != (tile_size, tile_size):
                    tile = cv2.resize(tile, (tile_size, tile_size))
                
                # Process tile through model
                edges_tile = self.preprocess_frame(tile)
                generated_tile = self.generate_from_edges(edges_tile)
                
                # Place processed tiles back in full image
                edges_full[y1:y2, x1:x2] = edges_tile
                generated_full[y1:y2, x1:x2] = generated_tile
        
        return edges_full, generated_full
    
    def process_video_tiled(self, input_video, output_video, target_size=1024, max_frames=None, debug=False):
        """Process video using tiled approach for high resolution"""
        
        cap = cv2.VideoCapture(input_video)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Tiled processing at: {target_size}x{target_size}")
        print(f"Debug mode: {'ON - 3-up comparison' if debug else 'OFF - processed only'}")
        
        # Verify target size is multiple of 256
        if target_size % 256 != 0:
            print(f"Warning: target_size {target_size} is not multiple of 256")
            target_size = (target_size // 256) * 256
            print(f"Adjusted to: {target_size}")
        
        # Set up output video dimensions
        if debug:
            output_width = target_size * 3  # 3-up comparison
        else:
            output_width = target_size       # Generated only
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, target_size))
        
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and processed_count >= max_frames:
                break
            
            # Process frame with tiling
            edges_full, generated_full = self.process_frame_tiled(frame, target_size)
            
            # Convert generated to BGR for video output
            generated_bgr = cv2.cvtColor(generated_full, cv2.COLOR_RGB2BGR)
            
            if debug:
                # Create 3-up comparison: original | edges | generated
                original_resized = cv2.resize(frame, (target_size, target_size))
                combined = np.hstack([original_resized, edges_full, generated_bgr])
                out.write(combined)
            else:
                # Output only the generated video
                out.write(generated_bgr)
            
            processed_count += 1
            
            if processed_count % 10 == 0:  # More frequent updates due to slower processing
                tiles_total = (target_size // 256) ** 2
                print(f"Processed {processed_count}/{total_frames if not max_frames else max_frames} frames ({tiles_total} tiles per frame)")
        
        cap.release()
        out.release()
        
        print(f"Tiled video processing complete: {output_video}")
        print(f"Processed {processed_count} frames at {target_size}x{target_size}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process videos with trained pix2pix model')
    
    # Required arguments
    parser.add_argument('input_video', help='Input video file path')
    
    # Processing method
    parser.add_argument('--method', choices=['upscale', 'tiled', 'temporal'], default='tiled',
                       help='Processing method (default: tiled)')
    
    # Model configuration
    parser.add_argument('--model_path', default='./models',
                       help='Path to model checkpoints (default: ./models)')
    parser.add_argument('--model_name', default='cactus_clean',
                       help='Model name (default: cactus_clean)')
    parser.add_argument('--epoch', default='latest',
                       help='Epoch to load (default: latest)')
    
    # Output options
    parser.add_argument('--output', help='Output video file (auto-generated if not specified)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (3-up comparison)')
    
    # Processing parameters
    parser.add_argument('--size', type=int, default=1024,
                       help='Output size for upscaling, or grid size for tiled/temporal (default: 1024)')
    parser.add_argument('--grid', type=int, 
                       help='Grid size (2-8 for tiled/temporal methods, overrides --size)')
    parser.add_argument('--time-offset', type=int, default=30,
                       help='Time offset between tiles for temporal method (default: 30 frames)')
    
    # Performance options
    parser.add_argument('--max_frames', type=int, default=100,
                       help='Limit number of frames to process (for testing)')
    
    return parser.parse_args()

def main():
    import sys
    
    # If no command line arguments provided, use interactive mode
    if len(sys.argv) == 1:
        interactive_main()
        return
    
    # Parse command line arguments
    args = parse_args()
    
    print(f"Processing: {args.input_video}")
    print(f"Method: {args.method}")
    print(f"Model: {args.model_name} epoch {args.epoch}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"max_frames: {args.max_frames}")
    
    # Initialize processor
    processor = VideoProcessor(args.model_path, args.model_name, args.epoch)
    
    # Generate output filename if not provided
    if not args.output:
        base_name = args.input_video.rsplit('.', 1)[0]
        suffix = "_debug" if args.debug else "_clean"
        if args.method == 'upscale':
            args.output = f"{base_name}_upscaled_{args.size}{suffix}.mp4"
        elif args.method == 'tiled':
            grid_size = args.grid or (args.size // 256) if args.size >= 256 else 4
            target_size = grid_size * 256
            args.output = f"{base_name}_tiled_{target_size}{suffix}.mp4"
        elif args.method == 'temporal':
            grid_size = args.grid or (args.size // 256) if args.size >= 256 else 4
            target_size = grid_size * 256
            args.output = f"{base_name}_temporal_{target_size}_offset{args.time_offset}{suffix}.mp4"
    
    print(f"Output: {args.output}")
    
    # Execute processing
    if args.method == 'upscale':
        processor.process_video(args.input_video, args.output,
                               output_size=args.size, max_frames=args.max_frames, debug=args.debug)
    
    elif args.method == 'tiled':
        grid_size = args.grid or (args.size // 256) if args.size >= 256 else 4
        target_size = grid_size * 256
        tiles_total = grid_size * grid_size
        
        print(f"Tiled processing: {grid_size}x{grid_size} grid ({tiles_total} tiles)")
        processor.process_video_tiled(args.input_video, args.output,
                                     target_size=target_size, max_frames=args.max_frames, debug=args.debug)
    
    elif args.method == 'temporal':
        grid_size = args.grid or (args.size // 256) if args.size >= 256 else 4
        target_size = grid_size * 256
        tiles_total = grid_size * grid_size
        
        print(f"Temporal collage: {grid_size}x{grid_size} grid ({tiles_total} temporal fragments)")
        print(f"Time offset: {args.time_offset} frames")
        processor.process_temporal_collage(args.input_video, args.output,
                                         target_size=target_size, time_offset=args.time_offset,
                                         max_frames=args.max_frames, debug=args.debug)

def interactive_main():
    # Configuration
    MODEL_PATH = './models'
    MODEL_NAME = 'cactus_clean'  # Your model name
    EPOCH = 'latest'  # or specific epoch like '100'
    
    INPUT_VIDEO = 'hoodie.mp4'    # Your input video
    VIDNAME = INPUT_VIDEO.split('.')[0]
    
    # Different output methods
    OUTPUT_UPSCALED = f'{VIDNAME}_output_upscaled_512.mp4'  # Simple upscaling
    OUTPUT_TILED = f'{VIDNAME}_output_tiled_1024.mp4'       # Tiled processing
    
    # Optional: limit frames for testing
    MAX_FRAMES = 100  # Set to number like 100 for testing
    
    # Initialize processor
    processor = VideoProcessor(MODEL_PATH, MODEL_NAME, EPOCH)
    
    print("Choose processing method:")
    print("1. Simple upscaling (fast)")
    print("2. Tiled processing (higher quality, slower)")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        # Option 1: Simple upscaling
        processor.process_video(INPUT_VIDEO, OUTPUT_UPSCALED, output_size=512, max_frames=MAX_FRAMES)
        
    elif choice == "2":
        # Option 2: Tiled processing
        print("Available tiled configurations:")
        print("2x2 tiles (512x512) - Fast")
        print("3x3 tiles (768x768) - Medium") 
        print("4x4 tiles (1024x1024) - Slow")
        print("5x5 tiles (1280x1280) - Very Slow")
        print("6x6 tiles (1536x1536) - Ultra Slow")
        print("8x8 tiles (2048x2048) - Extremely Slow")
        
        grid_size = int(input("Enter grid size (2, 3, 4, 5, 6, or 8): "))
        
        if grid_size < 2 or grid_size > 8:
            print(f"Invalid grid size, using 4x4")
            grid_size = 4
            
        target_size = grid_size * 256
        tiles_total = grid_size * grid_size
        
        print(f"Processing at {target_size}x{target_size} using {tiles_total} tiles per frame")
        print(f"Estimated processing time: ~{tiles_total}x slower than single frame")
        
        # Warn for very large grids
        if tiles_total > 25:
            confirm = input(f"Warning: {tiles_total} tiles per frame will be very slow. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("Cancelled")
                return
        
        output_file = f'{VIDNAME}_output_tiled_{target_size}.mp4'
        processor.process_video_tiled(INPUT_VIDEO, output_file, target_size=target_size, max_frames=MAX_FRAMES)
        
    else:
        print("Invalid choice, using simple upscaling")
        processor.process_video(INPUT_VIDEO, OUTPUT_UPSCALED, output_size=512, max_frames=MAX_FRAMES)

if __name__ == "__main__":
    main()
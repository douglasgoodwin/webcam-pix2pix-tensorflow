import cv2
import os
import numpy as np
from glob import glob
import argparse

def process_cactus_frames(frames_dir, output_dir):
    """
    Process your 3580 cactus frames into pix2pix training pairs
    """
    
    # Create output structure
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    
    # Get all frame files
    frame_files = sorted(glob(f"{frames_dir}/*.png"))
    print(f"Found {len(frame_files)} frames")
    
    if len(frame_files) != 3580:
        print(f"⚠️  Expected 3580 frames, found {len(frame_files)}")
    
    processed = 0
    
    for i, frame_file in enumerate(frame_files):
        img = cv2.imread(frame_file)
        if img is None:
            continue
        
        # Should already be 256x256, but ensure it
        if img.shape[:2] != (256, 256):
            img = cv2.resize(img, (256, 256))
        
        # Enhanced edge detection for cactus/desert scenes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale Canny optimized for organic shapes + spines
        # Lower thresholds to catch fine cactus spines
        edges1 = cv2.Canny(gray, 30, 120, apertureSize=3)  # Fine details
        edges2 = cv2.Canny(gray, 80, 160, apertureSize=5)  # Main structures
        
        # Combine for rich edge information
        edges_combined = cv2.bitwise_or(edges1, edges2)
        
        # Slight morphological enhancement for natural textures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_final = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
        
        # Convert to RGB
        edges_rgb = cv2.cvtColor(edges_final, cv2.COLOR_GRAY2RGB)
        
        # Create pix2pix pair: edges | original
        combined = np.hstack([edges_rgb, img])
        
        # Train/val split: 90% train (3222), 10% val (358)
        subset = "train" if i < len(frame_files) * 0.9 else "val"
        output_path = f"{output_dir}/{subset}/{os.path.basename(frame_file)}"
        
        cv2.imwrite(output_path, combined)
        processed += 1
        
        # Progress updates
        if processed % 200 == 0:
            print(f"✓ Processed {processed}/{len(frame_files)} frames ({processed/len(frame_files)*100:.1f}%)")
    
    train_count = len(glob(f"{output_dir}/train/*.png"))
    val_count = len(glob(f"{output_dir}/val/*.png"))
    
    print(f"\n🌵 Desert garden dataset ready!")
    print(f"📁 Dataset: {output_dir}")
    print(f"🚂 Training: {train_count} images")
    print(f"🧪 Validation: {val_count} images")
    print(f"📊 Total processed: {processed}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Process desert/cactus frames into pix2pix training dataset"
    )
    
    parser.add_argument(
        "input_dir",
        help="Input directory containing the extracted frames"
    )
    
    parser.add_argument(
        "output_dir",
        help="Output directory for the pix2pix dataset"
    )
    
    parser.add_argument(
        "--file-pattern",
        default="*.png",
        help="File pattern for frame files (default: *.png)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist!")
        return 1
    
    # Check if input directory contains frames
    frame_files = glob(os.path.join(args.input_dir, args.file_pattern))
    if not frame_files:
        print(f"❌ Error: No files matching '{args.file_pattern}' found in '{args.input_dir}'")
        return 1
    
    print(f"Processing frames from: {args.input_dir}")
    print(f"Output dataset to: {args.output_dir}")
    print(f"Looking for files: {args.file_pattern}")
    
    try:
        # Process all frames
        dataset_path = process_cactus_frames(args.input_dir, args.output_dir)
        
        print(f"\nNext steps:")
#         print(f"1. cd pix2pix-tensorflow")
#         print(f"2. Run training:")
#         print(f"   python pix2pix.py --mode train --output_dir ./cactus_model --max_epochs 200 --input_dir ../{os.path.basename(dataset_path)}/train --which_direction AtoB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing frames: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

# Example usage:
# python process_desert_frames.py /path/to/desert_frames desert_cactus_dataset
# python process_desert_frames.py ./frames ./training_dataset --file-pattern "*.jpg"
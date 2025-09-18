import cv2
import os
import numpy as np
from glob import glob

def create_clean_cactus_edges(raw_frames_dir, output_dir):
    """
    Create MUCH cleaner edge detection for cactus scenes
    """
    
    # Clean up and recreate directories
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(f"{output_dir}/train", exist_ok=True)   # Combined aligned images
    os.makedirs(f"{output_dir}/val", exist_ok=True)     # Combined aligned images
    
    frame_files = sorted(glob(f"{raw_frames_dir}/*.png"))
    print(f"Reprocessing {len(frame_files)} frames with MUCH cleaner edges...")
    
    processed = 0
    
    for i, frame_file in enumerate(frame_files):
        img = cv2.imread(frame_file)
        if img is None:
            continue
        
        # Ensure 256x256
        img = cv2.resize(img, (256, 256))
        
        # Balanced edge detection - capture cactus structure without noise
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Light pre-blur to reduce noise but keep detail
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        
        # Multi-scale Canny - back to original approach but balanced
        edges1 = cv2.Canny(blurred, 40, 120, apertureSize=3)  # Fine details (spines)
        edges2 = cv2.Canny(blurred, 80, 160, apertureSize=5)  # Main structure
        
        # Combine both scales
        edges_combined = cv2.bitwise_or(edges1, edges2)
        
        # Light cleanup to connect broken lines, reduce isolated pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_cleaned = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Alternative: Try adaptive threshold instead of Canny
        # (uncomment these lines and comment out the Canny section above to test)
        # adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                cv2.THRESH_BINARY, 15, 8)
        # edges_cleaned = cv2.bitwise_not(adaptive)
        
        # Convert to RGB
        edges_rgb = cv2.cvtColor(edges_cleaned, cv2.COLOR_GRAY2RGB)
        
        # Create side-by-side aligned format: edges | original
        combined = np.hstack([edges_rgb, img])  # 512x256 image
        
        # 90% train, 10% val
        if i < len(frame_files) * 0.9:
            subset = "train"
        else:
            subset = "val"
        
        filename = os.path.basename(frame_file)
        
        # Save as single aligned image (edges|original)
        cv2.imwrite(f"{output_dir}/{subset}/{filename}", combined)
        
        processed += 1
        
        if processed % 200 == 0:
            print(f"✓ Clean edges: {processed}/{len(frame_files)}")
    
    # Test visualization - save a few comparison images
    print("\n📸 Creating comparison samples...")
    for i in [0, 100, 500]:
        if i < len(frame_files):
            frame_file = frame_files[i]
            img = cv2.imread(frame_file)
            img = cv2.resize(img, (256, 256))
            
            # Create the same edge processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
            edges = cv2.Canny(blurred, 100, 200, apertureSize=3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges_cleaned = cv2.morphologyEx(edges_cleaned, cv2.MORPH_OPEN, kernel)
            edges_rgb = cv2.cvtColor(edges_cleaned, cv2.COLOR_GRAY2RGB)
            
            # Side-by-side comparison
            comparison = np.hstack([edges_rgb, img])
            cv2.imwrite(f"edge_comparison_{i:03d}.png", comparison)
    
    train_count = len(glob(f"{output_dir}/train/*.png"))
    val_count = len(glob(f"{output_dir}/val/*.png"))
    
    print(f"\n🌵 CLEAN aligned cactus dataset created!")
    print(f"📁 Dataset: {output_dir}")
    print(f"🚂 Training: {train_count} aligned images")
    print(f"🧪 Validation: {val_count} aligned images")
    print(f"\n📸 Check comparison images: edge_comparison_*.png")
    print(f"\nIf edges look good, restart training:")
    print(f"python train.py --dataroot {output_dir} --name cactus_clean --model pix2pix --direction AtoB")

if __name__ == "__main__":
    raw_frames_folder = "raw_cactus_frames"  # Your raw frames
    clean_dataset = "../cactus_clean_dataset"
    
    create_clean_cactus_edges(raw_frames_folder, clean_dataset)
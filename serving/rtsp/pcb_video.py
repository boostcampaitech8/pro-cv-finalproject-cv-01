import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def create_premium_pcb_video():
    # --- [1. CONFIGURATION] ---
    WIDTH, HEIGHT = 1920, 1080 # Full HD for premium feel
    FPS = 30                   # Smooth 60fps
    MOVE_SPEED = 12            # Pixels per frame
    GAP = 500                  # Gap between PCBs
    
    # Visual Effects
    NOISE_LEVEL = 3            # Subtle ISO noise
    SHADOW_BLUR = 45          # Drop shadow spread
    SHADOW_OPACITY = 0.6       # Shadow intensity
    VIGNETTE_STRENGTH = 0.25   # Reduced for a cleaner look
    BOTTOM_SHADING_STRENGTH = 0.15 # Subtle darkening at the bottom
    
    ROOT_DIR = '/data/ephemeral/home/ss/serving/rtsp'
    OUTPUT_NAME = os.path.join(ROOT_DIR, '30fps_dark.mp4')
    
    # --- [2. SEARCH IMAGES] ---
    # Only look inside the 'images' folder to avoid picking up temp files or outputs
    IMAGE_DIR = os.path.join(ROOT_DIR, 'images')
    img_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.[jJ][pP][gG]")))
    
    if not img_paths:
        print(f"❌ No images found in {IMAGE_DIR}")
        return
    print(f"📦 Found {len(img_paths)} images. Processing...")

    # --- [3. PRE-PROCESS IMAGES] ---
    processed_imgs = []
    target_h = int(HEIGHT * 0.7) # PCBs take 70% of height
    
    for path in img_paths:
        img = cv2.imread(path)
        if img is None: continue
        
        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = target_h / h
        new_w, new_h = int(w * scale), target_h
        img_res = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        processed_imgs.append(img_res)

    # --- [4. DESIGN THE BELT] ---
    # Calculate dimensions
    # total_belt_width is one full cycle of images + gaps
    # To make it look like a continuous empty belt after the last image, 
    # we ensure there's a GAP after the last image before the first one repeats.
    total_belt_width = sum([img.shape[1] for img in processed_imgs]) + (GAP * len(processed_imgs))
    
    # We need to tile the belt to ensure smooth scrolling. 
    # For a perfect seamless loop that includes empty space, we just use total_belt_width.
    # The canvas needs to be total_belt_width + WIDTH to handle the transition.
    canvas_width = total_belt_width + WIDTH
    belt = np.full((HEIGHT, canvas_width, 3), 15, dtype=np.uint8) # Dark charcoal belt

    # Create a subtle texture for the conveyor belt
    noise_tex = np.random.randint(0, 10, (HEIGHT, canvas_width, 3), dtype=np.uint8)
    belt = cv2.add(belt, noise_tex)

    # Function to draw drop shadow
    def draw_shadow(canvas, x, y, w, h):
        # Even more subtle shadow as requested
        SUBTLE_SHADOW_OPACITY = 0.25 
        SUBTLE_SHADOW_BLUR = 61
        
        shadow_mask = np.zeros((HEIGHT, canvas_width), dtype=np.uint8)
        cv2.rectangle(shadow_mask, (x+10, y+10), (x+w-10, y+h-10), 255, -1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (SUBTLE_SHADOW_BLUR, SUBTLE_SHADOW_BLUR), 0)
        
        # Apply shadow using linear blending
        shadow_area = (shadow_mask.astype(float) / 255.0) * SUBTLE_SHADOW_OPACITY
        for c in range(3):
            canvas[:, :, c] = (canvas[:, :, c].astype(float) * (1.0 - shadow_area)).astype(np.uint8)

    # Place images on belt
    curr_x = 0
    # Add images to the belt
    for img in processed_imgs:
        h_i, w_i = img.shape[:2]
        y_off = (HEIGHT - h_i) // 2
        
        # Draw shadow
        draw_shadow(belt, curr_x, y_off, w_i, h_i)
        
        # Place image
        belt[y_off:y_off+h_i, curr_x:curr_x+w_i] = img
        curr_x += (w_i + GAP)

    # To make the loop seamless, we need to draw the beginning of the belt again at the end
    # of the 'total_belt_width' position.
    temp_x = total_belt_width
    idx = 0
    while temp_x < canvas_width:
        img = processed_imgs[idx % len(processed_imgs)]
        h_i, w_i = img.shape[:2]
        y_off = (HEIGHT - h_i) // 2
        
        # Calculate maximum possible width to draw without exceeding canvas
        draw_w = min(w_i, canvas_width - temp_x)
        if draw_w <= 0: break
        
        draw_shadow(belt, temp_x, y_off, w_i, h_i)
        belt[y_off:y_off+h_i, temp_x:temp_x+draw_w] = img[:, :draw_w]
        
        temp_x += (w_i + GAP)
        idx += 1

    # --- [5. VIDEO RENDERING] ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_NAME, fourcc, FPS, (WIDTH, HEIGHT))

    print(f"🎬 Rendering {OUTPUT_NAME} for Seamless Loop...")

    # Vignette mask
    X_resultant_kernel = cv2.getGaussianKernel(WIDTH, WIDTH/2)
    Y_resultant_kernel = cv2.getGaussianKernel(HEIGHT, HEIGHT/2)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    vignette = (1.0 - VIGNETTE_STRENGTH) + VIGNETTE_STRENGTH * mask
    vignette = cv2.merge([vignette, vignette, vignette])

    # Number of frames to complete one cycle
    total_frames = total_belt_width // MOVE_SPEED

    for f in tqdm(range(total_frames)):
        start_x = (f * MOVE_SPEED)
        
        # Extract frame
        frame = belt[:, start_x : start_x + WIDTH].copy()
        
        # Aesthetic: Vignette
        frame = (frame.astype(float) * vignette).astype(np.uint8)
        
        # Aesthetic: Lighting (Smooth Top Glow)
        top_glow_h = HEIGHT // 2
        y_top = np.linspace(1.0, 0.0, top_glow_h).reshape(top_glow_h, 1, 1)
        top_glow = (y_top * [30, 30, 35]).astype(np.uint8)
        # Use numpy addition with clipping for saturation
        frame[:top_glow_h, :] = np.clip(frame[:top_glow_h, :].astype(np.int16) + top_glow, 0, 255).astype(np.uint8)
        
        # Aesthetic: Soft Bottom Shading (Center to Bottom)
        # Natural transition from center (no shading) to bottom
        shading_h = HEIGHT // 2
        y_bottom = np.linspace(0.0, 1.0, shading_h).reshape(shading_h, 1, 1)
        shading_mask = 1.0 - (y_bottom * BOTTOM_SHADING_STRENGTH)
        frame[HEIGHT//2:, :] = (frame[HEIGHT//2:, :].astype(float) * shading_mask).astype(np.uint8)
        
        # Aesthetic: Low-level noise for film grain feel
        grain = np.random.normal(0, NOISE_LEVEL, (HEIGHT, WIDTH, 3)).astype(np.int8)
        frame = cv2.add(frame, grain, dtype=cv2.CV_8U)
        
        out.write(frame)

    out.release()
    print(f"\n✨ SUCCESS: Premium video saved to {OUTPUT_NAME}")

if __name__ == "__main__":
    create_premium_pcb_video()

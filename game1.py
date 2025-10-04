import cv2
import numpy as np
import random
import time



# Load OpenCV's built-in Haar cascade for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Game variables
# Set window size to match screen resolution
# Get primary screen resolution
width, height = 640, 480  # Default fallback values
try:
    # Try to get screen resolution if running in a desktop environment
    import ctypes
    user32 = ctypes.windll.user32
    width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    # Use 80% of screen size for the game window
    width = int(width * 0.8)
    height = int(height * 0.8)
except:
    # Fallback to default values if we can't get screen resolution
    width, height = 640, 320
bird_y = height // 2
bird_v = 0
gravity = 0.3
short_jump = -3
long_jump = -6
pipe_gap = 180   # Increased vertical gap between top and bottom pipes
pipe_width = 80
pipe_speed = 6  # Increased from 4 to 6 for faster movement (50% increase)
pipes = []
score = 0
game_over = False
is_blinking = False
blink_start = 0

# Bird animation
wing_up = True
wing_counter = 0

# Clouds
clouds = [{"x": random.randint(0, width), "y": random.randint(0, height//2), "size": random.randint(40, 80)} for _ in range(5)]
cloud_speed = 1

# Open camera
cap = cv2.VideoCapture(0)

# Create resizable window
cv2.namedWindow("Blink Chicken - With Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blink Chicken - With Webcam", width, height)

def reset_game():
    global bird_y, bird_v, pipes, score, game_over
    bird_y = height // 2
    bird_v = 0
    pipes = []
    score = 0
    game_over = False
    pipes.append({"x": width, "h": random.randint(50, height - pipe_gap - 50)})

reset_game()

def draw_bird(canvas, x, y, wing_up):
    # White chicken body (slightly larger oval)
    cv2.ellipse(canvas, (x, int(y)), (18, 15), 0, 0, 360, (255, 255, 255), -1)
    
    # Tail feathers
    tail_pts = np.array([[x - 15, int(y) - 8], [x - 25, int(y) - 12], [x - 25, int(y)], [x - 25, int(y) + 12], [x - 15, int(y) + 8]], np.int32)
    cv2.fillPoly(canvas, [tail_pts], (220, 220, 220))  # Light gray tail
    
    # Wings
    if wing_up:
        wing_pts = np.array([[x - 5, int(y) - 5], [x - 20, int(y) - 20], [x - 10, int(y) - 5]], np.int32)
    else:
        wing_pts = np.array([[x - 5, int(y) + 5], [x - 20, int(y) + 20], [x - 10, int(y) + 5]], np.int32)
    cv2.fillPoly(canvas, [wing_pts], (240, 240, 240))  # Slightly darker white for wings
    
    # Head (slightly smaller than body)
    cv2.circle(canvas, (x + 12, int(y) - 5), 10, (255, 255, 255), -1)
    
    # Eye
    cv2.circle(canvas, (x + 15, int(y) - 7), 2, (0, 0, 0), -1)
    
    # Beak
    beak_pts = np.array([[x + 20, int(y) - 7], [x + 28, int(y) - 5], [x + 20, int(y) - 3]], np.int32)
    cv2.fillPoly(canvas, [beak_pts], (255, 165, 0))  # Orange beak
    
    # Comb on head (red crest)
    comb_pts = np.array([[x + 10, int(y) - 15], [x + 13, int(y) - 20], [x + 16, int(y) - 15]], np.int32)
    cv2.fillPoly(canvas, [comb_pts], (0, 0, 255))  # Red comb

def draw_clouds(canvas, clouds):
    for cloud in clouds:
        x, y, size = cloud["x"], cloud["y"], cloud["size"]
        cv2.circle(canvas, (x, y), size//2, (255, 255, 255), -1)
        cv2.circle(canvas, (x + size//3, y + size//4), size//3, (255, 255, 255), -1)
        cv2.circle(canvas, (x - size//3, y + size//4), size//3, (255, 255, 255), -1)
        # Move cloud
        cloud["x"] -= cloud_speed
        if cloud["x"] + size < 0:
            cloud["x"] = width + size
            cloud["y"] = random.randint(0, height//2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        exit()
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    blink_detected = False
    blink_type = None
    
    if len(eyes) == 0:  # No eyes detected → blink
        if not is_blinking:
            is_blinking = True
            blink_start = time.time()
    else:
        if is_blinking:
            blink_duration = time.time() - blink_start
            blink_detected = True
            blink_type = "long" if blink_duration > 0.25 else "short"
            is_blinking = False

    if not game_over:
        # Bird physics (adjusted for faster game speed)
        bird_v += gravity * 1.2  # Increased gravity by 20% to match faster game speed
        bird_y += bird_v

        # Blink jump
        if blink_detected:
            if blink_type == "long":
                bird_v = long_jump
            else:
                bird_v = short_jump

        # Update pipes
        for pipe in pipes:
            pipe["x"] -= pipe_speed

        if pipes and pipes[0]["x"] + pipe_width < 0:
            pipes.pop(0)
            score += 1

        # Increase spacing between pipes (doubled the original distance)
        if len(pipes) == 0 or pipes[-1]["x"] < width - 600:  # Increased from 350 to 600
            pipes.append({"x": width, "h": random.randint(50, height - pipe_gap - 50)})

        # Collision check
        for pipe in pipes:
            # Adjusted collision box for the chicken (slightly larger)
            bird_x = 50
            bird_width = 35  # Increased from 30 to account for chicken design
            bird_height = 30
            if bird_x < pipe["x"] + pipe_width and bird_x + bird_width > pipe["x"]:  # bird within pipe X
                if bird_y - bird_height/2 < pipe["h"] or bird_y + bird_height/2 > pipe["h"] + pipe_gap:
                    game_over = True

        if bird_y <= 0 or bird_y >= height:
            game_over = True

    # Draw game canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (135, 206, 250)  # Sky blue background
    
    # Resize webcam frame to fit in a corner of the game
    if ret:
        # Calculate webcam display size (25% of game width, maintaining aspect ratio)
        webcam_width = width // 4
        webcam_height = int(frame.shape[0] * (webcam_width / frame.shape[1]))
        webcam_frame = cv2.resize(frame, (webcam_width, webcam_height))
        
        # Position in top-right corner with padding
        padding = 10
        y_offset = padding  # Top position
        x_offset = width - webcam_width - padding  # Right position
        
        # Create a semi-transparent overlay for the webcam
        overlay = canvas.copy()
        overlay[y_offset:y_offset+webcam_height, x_offset:x_offset+webcam_width] = webcam_frame
        cv2.rectangle(overlay, (x_offset-2, y_offset-2), 
                     (x_offset+webcam_width+2, y_offset+webcam_height+2), (255, 255, 255), 2)
        
        # Blend the webcam overlay with the game canvas
        alpha = 0.8  # Transparency factor
        cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0, canvas)

    # Draw clouds
    draw_clouds(canvas, clouds)

    # Draw pipes
    for pipe in pipes:
        cv2.rectangle(canvas, (pipe["x"], 0), (pipe["x"] + pipe_width, pipe["h"]), (0, 255, 0), -1)
        cv2.rectangle(canvas, (pipe["x"], pipe["h"] + pipe_gap), 
                      (pipe["x"] + pipe_width, height), (0, 255, 0), -1)

    # Draw bird with flapping wings
    wing_counter += 1
    if wing_counter % 5 == 0:
        wing_up = not wing_up
    draw_bird(canvas, 50, bird_y, wing_up)

    # Draw score
    cv2.putText(canvas, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if game_over:
        cv2.putText(canvas, "GAME OVER", (width//2 - 100, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(canvas, "Short Blink or Press R to Restart", (width//2 - 180, height//2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, "Long Blink or Press Q to Exit", (width//2 - 180, height//2 + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # Handle blink actions when game is over
        if blink_detected:
            if blink_type == "short":
                reset_game()
            elif blink_type == "long":
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Show game
    cv2.imshow("Blink Chicken - With Webcam", canvas)

    key = cv2.waitKey(15) & 0xFF  # Reduced from 20 to 15 for faster frame rate
    if key == 27 or key == ord('q'):  # ESC or Q → exit
        break
    if key == ord('r'):
        reset_game()

cap.release()
cv2.destroyAllWindows()



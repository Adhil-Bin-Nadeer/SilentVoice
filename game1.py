import cv2
import numpy as np
import random
import time

# Load OpenCV's built-in Haar cascade for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Game variables
width, height = 640, 480
bird_y = height // 2
bird_v = 0
gravity = 0.3
short_jump = -3
long_jump = -6
pipe_gap = 180   # Increased vertical gap between top and bottom pipes
pipe_width = 80
pipe_speed = 2
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
    # Body
    cv2.ellipse(canvas, (x, int(y)), (15, 12), 0, 0, 360, (0, 0, 255), -1)
    # Wings
    if wing_up:
        pts = np.array([[x - 5, int(y)], [x - 20, int(y - 15)], [x - 5, int(y)]], np.int32)
    else:
        pts = np.array([[x - 5, int(y)], [x - 20, int(y + 15)], [x - 5, int(y)]], np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
    # Eye
    cv2.circle(canvas, (x + 5, int(y - 2)), 2, (255, 255, 255), -1)

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
        break
    
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
        # Bird physics
        bird_v += gravity
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

        # Increase spacing between pipes (was 200 → now 300)
        if len(pipes) == 0 or pipes[-1]["x"] < width - 300:
            pipes.append({"x": width, "h": random.randint(50, height - pipe_gap - 50)})

        # Collision check
        for pipe in pipes:
            if 50 < pipe["x"] + pipe_width and 50 + 30 > pipe["x"]:  # bird within pipe X
                if bird_y < pipe["h"] or bird_y > pipe["h"] + pipe_gap:
                    game_over = True

        if bird_y <= 0 or bird_y >= height:
            game_over = True

    # Draw game canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (135, 206, 250)  # Sky blue background

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
        cv2.putText(canvas, "Press R to Restart | Q to Exit", (width//2 - 180, height//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show game
    cv2.imshow("Blink Bird", canvas)

    key = cv2.waitKey(20) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or Q → exit
        break
    if key == ord('r'):
        reset_game()

cap.release()
cv2.destroyAllWindows()

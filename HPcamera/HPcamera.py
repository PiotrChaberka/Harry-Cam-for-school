import cv2
import pygame
import time
import random

# Initialize Pygame
pygame.mixer.init()

# Load the music file
music_file = "music.mp3"  # Replace with the path to the music file
pygame.mixer.music.load(music_file)
pygame.mixer.music.play(-1)  # Start playing the music in a loop

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use the default camera (change the index if you have multiple cameras)

# Load the hat model image with transparent background
hat_image = cv2.imread("model.png", cv2.IMREAD_UNCHANGED)  # Replace with the path to the transparent hat image

# Load the pre-trained Haar cascade classifier for face detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Variables for hat position smoothing
smooth_factor = 0.75  # Smoothing factor (0.0 - 1.0)
smooth_hat_positions = []  # List to store the smoothed hat positions

# Maximum number of hats to display
max_hats = 2

# Function to detect a person and overlay the hat on the face
def detect_person(frame):
    # Convert the frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Shuffle the faces to randomize the hat placement
    random.shuffle(faces)

    # Overlay the hat on each detected face up to the maximum limit
    for (x, y, w, h) in faces[:max_hats]:
        # Resize the hat image to be 2x bigger than the face width
        hat_width = int(w * 2)
        hat_height = int(h * 2)
        hat_resized = cv2.resize(hat_image, (hat_width, hat_height))

        # Calculate the position to place the hat on the face
        hat_x = x - int((hat_width - w) / 2)
        hat_y = y - h - 85

        # Smooth the hat position using averaging
        smooth_hat_x, smooth_hat_y = smooth_hat_position(hat_x, hat_y)

        # Ensure that the hat does not go beyond the frame boundaries
        if (
            smooth_hat_x >= 0
            and smooth_hat_y >= 0
            and smooth_hat_x + hat_width <= frame.shape[1]
            and smooth_hat_y + hat_height <= frame.shape[0]
        ):
            # Create a mask for the hat image
            mask = hat_resized[:, :, 3] / 255.0  # Normalize the alpha channel (transparency) to the range [0, 1]

            # Apply the mask to the hat region
            hat_region = frame[smooth_hat_y : smooth_hat_y + hat_height, smooth_hat_x : smooth_hat_x + hat_width]
            hat_region[:, :, 0] = hat_region[:, :, 0] * (1 - mask) + hat_resized[:, :, 0] * mask
            hat_region[:, :, 1] = hat_region[:, :, 1] * (1 - mask) + hat_resized[:, :, 1] * mask
            hat_region[:, :, 2] = hat_region[:, :, 2] * (1 - mask) + hat_resized[:, :, 2] * mask

            # Store the smoothed hat position
            smooth_hat_positions.append((smooth_hat_x, smooth_hat_y))

    # Remove old hat positions beyond the maximum limit
    if len(smooth_hat_positions) > max_hats:
        smooth_hat_positions.pop(0)

    # Display the frame with the hats
    cv2.imshow("Camera", frame)

# Function to smooth the hat position using averaging
def smooth_hat_position(hat_x, hat_y):
    smooth_hat_x = hat_x
    smooth_hat_y = hat_y

    if len(smooth_hat_positions) > 0:
        prev_hat_x, prev_hat_y = smooth_hat_positions[-1]
        smooth_hat_x = int(prev_hat_x * smooth_factor + hat_x * (1 - smooth_factor))
        smooth_hat_y = int(prev_hat_y * smooth_factor + hat_y * (1 - smooth_factor))

    return smooth_hat_x, smooth_hat_y

# Start the main loop
while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Check if a person is in the camera's view and overlay the hat
    detect_person(frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the music
pygame.mixer.music.stop()

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()


import cv2
import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the Pygame window
window_width, window_height = 800, 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Harry Potter Hat")

# Load the hat model image
hat_image = pygame.image.load("model.png")  # Replace with the actual path to the hat model image

# Load the Harry Potter music
pygame.mixer.music.load("music.mp3")  # Replace with the actual path to the Harry Potter music file

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use the default camera (change the index if you have multiple cameras)

# Define the position for placing the hat
hat_position = (300, 100)  # Adjust the position based on your desired placement

# Function to detect a person using OpenCV
def detect_person(frame):
    # Convert the frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a pre-trained Haar cascade classifier for person detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Detect persons in the frame
    persons = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return True if a person is detected, False otherwise
    return len(persons) > 0

# Start the main loop
running = True
person_detected = False
animation_started = False

while running:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Mirror the frame horizontally for a more intuitive display
    frame = cv2.flip(frame, 1)

    # Check if a person is in the camera's view
    if detect_person(frame):
        person_detected = True
        # Display the hat on the person's head
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = pygame.surfarray.make_surface(frame)
        window.blit(frame_pil, (0, 0))
        window.blit(hat_image, hat_position)
    else:
        person_detected = False

    # Update the Pygame window
    pygame.display.update()

    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check if the animation should start
    if person_detected and not animation_started:
        command = input("Enter 'start' to begin the animation: ")
        if command == "start":
            # Play the Harry Potter music
            pygame.mixer.music.play()

            # Display one of the four possible lines from Harry Potter
            lines = ["Line 1", "Line 2", "Line 3", "Line 4"]
            random_line = random.choice(lines)
            print(random_line)

            animation_started = True

# Release the camera and quit Pygame
camera.release()
pygame.quit()
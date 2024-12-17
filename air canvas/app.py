import cv2
import numpy as np

# Define color ranges for marker (e.g., blue color in HSV)
lower_bound = np.array([100, 150, 50])  # Adjust these values for blue
upper_bound = np.array([140, 255, 255])

# Initialize variables
canvas = None
drawing = False
brush_color = (255, 0, 0)  # Default: Blue
brush_size = 5

# Brush colors dictionary
colors = {
    'b': (255, 0, 0),  # Blue
    'g': (0, 255, 0),  # Green
    'r': (0, 0, 255),  # Red
    'y': (0, 255, 255)  # Yellow
}

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect

    # Initialize canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get the marker color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500:  # Minimum area to avoid noise
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            center = (int(x), int(y))

            # Debugging: Show contour area and center
            print(f"Contour area: {cv2.contourArea(max_contour)}")
            print(f"Drawing center: {center}")

            if drawing:
                # Draw on the canvas with brush size and color
                cv2.circle(canvas, center, brush_size, brush_color, -1)

            # Draw a circle around the marker in the frame
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
        else:
            drawing = False  # Disable drawing if the contour is too small
    else:
        drawing = False  # Disable drawing if no contour is found

    # Combine the frame and canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show the outputs
    cv2.imshow('Air Canvas', combined)
    cv2.imshow('Mask', mask)

    # Key bindings for functionalities
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear the canvas
        canvas = np.zeros_like(frame)
        print("Canvas cleared.")
    elif key == ord('s'):  # Save the canvas
        cv2.imwrite('air_canvas_output.png', canvas)
        print("Canvas saved as 'air_canvas_output.png'")
    elif key in colors:  # Change brush color
        brush_color = colors[chr(key)]
        print(f"Brush color changed to {chr(key).upper()}")
    elif key == ord('+'):  # Increase brush size
        brush_size = min(20, brush_size + 1)
        print(f"Brush size increased to {brush_size}")
    elif key == ord('-'):  # Decrease brush size
        brush_size = max(1, brush_size - 1)
        print(f"Brush size decreased to {brush_size}")
    elif key == ord('d'):  # Toggle drawing mode
        drawing = not drawing
        print(f"Drawing mode {'enabled' if drawing else 'disabled'}.")

# Release resources
cap.release()
cv2.destroyAllWindows()

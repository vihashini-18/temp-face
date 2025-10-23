import cv2
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

CAMERAS = config["CAMERA_SOURCES"]

# Open all cameras (including DroidCam)
caps = []
for cam in CAMERAS:
    cap = cv2.VideoCapture(cam["source"])
    if not cap.isOpened():
        print(f"Error: Cannot open {cam['name']}")
    else:
        print(f"{cam['name']} connected.")
        caps.append((cam["name"], cap))

# Main loop
while True:
    for cam_name, cap in caps:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow(cam_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for _, cap in caps:
    cap.release()
cv2.destroyAllWindows()

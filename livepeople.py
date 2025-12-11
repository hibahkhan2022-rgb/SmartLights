import os
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

# ========= GPIO VIA SYSFS =========
# REPLACE THIS WITH THE REAL GPIO NUMBER FOR YOUR PIN
GPIO_NUM = 15  # <-- EXAMPLE. Put your actual GPIO number here.

GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_NUM}"

def export_gpio():
    # export if not already exported
    if not os.path.exists(GPIO_PATH):
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(GPIO_NUM))
    # set direction to out
    with open(os.path.join(GPIO_PATH, "direction"), "w") as f:
        f.write("out")

def set_gpio(value: bool):
    with open(os.path.join(GPIO_PATH, "value"), "w") as f:
        f.write("1" if value else "0")

def cleanup_gpio():
    if os.path.exists(GPIO_PATH):
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(GPIO_NUM))
        except Exception:
            # ignore if already unexported
            pass

# export + init GPIO
export_gpio()
set_gpio(False)  # LED off initially

# ========= JETSON INFERENCE SETUP =========
net = detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = videoSource("csi://0")      # or "v4l2:///dev/video0"
display = videoOutput("display://0")

try:
    while display.IsStreaming():
        img = camera.Capture()
        if img is None:
            continue

        detections = net.Detect(img)

        # PERSON = classID 1 (COCO)
        person_detected = any(d.ClassID == 1 for d in detections)

        # LED control
        set_gpio(person_detected)

        # Display
        display.Render(img)
        display.SetStatus(
            "Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())
        )

finally:
    # on exit, turn LED off and clean up
    set_gpio(False)
    cleanup_gpio()


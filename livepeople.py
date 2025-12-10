from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput
import Jetson.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
LED_PIN = 18
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = videoSource("csi://0")      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
    img = camera.Capture()

    if img is None:
        continue

    detections = net.Detect(img)
    
    # ----- PERSON CHECK -----
    person_detected = False
    for d in detections:
        if d.ClassID == 1:   # 1 = person
            person_detected = True
            break

    # ----- LED CONTROL -----
    if person_detected:
        GPIO.output(LED_PIN, GPIO.HIGH)
    else:
        GPIO.output(LED_PIN, GPIO.LOW)

    # ----- DISPLAY -----
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

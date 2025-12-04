#===================
# README... FOR NOW
#===================
# hand gestures you can do: 
# 1. peace sign
# 2. middle finger
# 3. closed fist
# 4. open palm: if you close your fist and turn, then it changes the color that you draw with
# 5. hang loose
# 6. thumbs down
# 7. rock and roll
# 8. i love you
# 9. ok
# 10. pinch
# 11. pointer: used to draw
# 12. tight (or the promise sign): used as an eraser for when you draw
# 13. "i" in ASL erases the whole drawing


# import modules and make it easier to refer to submodules 
import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

#===================
# GESTURE FUNCTIONS
#===================
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def what_finger_be_doing(hand_landmarks):
    fingers = {}
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    wrist = hand_landmarks.landmark[0]
    is_right_hand = thumb_tip.x < wrist.x

    if is_right_hand:
        fingers["thumb"] = thumb_tip.x < thumb_ip.x
    else:
        fingers["thumb"] = thumb_tip.x > thumb_ip.x
        
    fingers["index"] = hand_landmarks.landmark[8].y  < hand_landmarks.landmark[6].y
    fingers["middle"] = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    fingers["ring"] = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    fingers["pinky"] = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    return fingers


def curled_fingers(tip, pip, mcp):
    dist_tip_mcp = distance(tip, mcp)
    dist_tip_pip = distance(tip, pip)
    return dist_tip_mcp < dist_tip_pip


def classify_gesture(fingers, hand_landmarks):
    # landmark showing where each fingertip and wrist is
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip  = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # distance for certain gestures
    pinch_dist = distance(thumb_tip, index_tip)
    index_middle_dist = distance(index_tip, middle_tip)

    # actual fingers if they are extended
    pointer = fingers["index"]
    middle  = fingers["middle"]
    ring    = fingers["ring"]
    pinky   = fingers["pinky"]
    thumb   = fingers["thumb"]

    # curled fingers (or toes...)
    curled_index  = curled_fingers(index_tip, hand_landmarks.landmark[6], hand_landmarks.landmark[5])
    curled_middle = curled_fingers(middle_tip, hand_landmarks.landmark[10], hand_landmarks.landmark[9])
    curled_ring   = curled_fingers(ring_tip, hand_landmarks.landmark[14], hand_landmarks.landmark[13])
    curled_pinky  = curled_fingers(pinky_tip, hand_landmarks.landmark[18], hand_landmarks.landmark[17])

    #================================================
    # actual gestures
    #================================================
    
    # "i" or complete eraser
    if not pointer and not middle and not ring and pinky and not thumb:
        return "i in ASL"
    
    # hang loooosseee
    if thumb and pinky and curled_index and curled_middle and curled_ring:
        return "Waaazzzaaapppp"
    
    # thumbs down
    if thumb and curled_index and curled_middle and curled_ring and curled_pinky:
        if thumb_tip.y > wrist.y:
            return "Thumbs down. Nuh uh!"

    # rock and rollll || <3
    if pointer and not middle and not ring and pinky:
        if thumb:
            return "I love you!"
        else:
            return "Rock and Rollllll!!!!!"

    # ok!
    thumb_tip  = hand_landmarks.landmark[4]
    index_tip  = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip   = hand_landmarks.landmark[16]
    pinky_tip  = hand_landmarks.landmark[20]

    pinch_dist = distance(thumb_tip, index_tip)
    other_fingers = middle and ring and pinky

    if pinch_dist < 0.03 and other_fingers:
        return "OK"

    # closed fist
    curled_index  = curled_fingers(index_tip, hand_landmarks.landmark[6], hand_landmarks.landmark[5])
    curled_middle = curled_fingers(middle_tip, hand_landmarks.landmark[10], hand_landmarks.landmark[9])
    curled_ring   = curled_fingers(ring_tip, hand_landmarks.landmark[14], hand_landmarks.landmark[13])
    curled_pinky  = curled_fingers(pinky_tip, hand_landmarks.landmark[18], hand_landmarks.landmark[17])
    curled_thumb  = distance(thumb_tip, hand_landmarks.landmark[2]) < distance(thumb_tip, hand_landmarks.landmark[1])

    if curled_index and curled_middle and curled_ring and curled_pinky and curled_thumb:
        return "Closed Fist"

    # pinch
    curled_index = curled_fingers(index_tip, hand_landmarks.landmark[6], hand_landmarks.landmark[5])
    if pinch_dist < 0.035 and not (curled_index and other_fingers):
        return "Pinch"

    # eraser (or 2) || peace
    if pointer and middle and not ring and not pinky:
        if index_middle_dist < 0.04:
            return "Eraser"
        else:
            return "Peaceee!!!!"

    # pointer
    if pointer and not middle and not ring and not pinky:
        return "Pointing/Drawing"

    # middle finger
    if middle and not pointer and not ring and not pinky:
        return "Middle finger"

    # open palm
    if pointer and middle and ring and pinky:
        return "Open Palm"

    return None


def hand_roll_angle(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]
    pinky_mcp = hand_landmarks.landmark[17]

    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y
    angle = math.degrees(math.atan2(dy, dx))

    if angle > 180:
        angle -= 360
    if angle < -180:
        angle += 360

    return angle

#===================
# TRACKING VIDEO  
#===================
def run_tracking():
    cam = cv2.VideoCapture(0)
    canvas = None
    prev_point = None

    # testing color turns
    colors = [(255, 255, 255), # white
              (255,0,0), # red
              (255, 151, 0), # orange
              (255,255,0), # yellow
              (0,255,0), # green
              (0,0,255), # blue
              (151, 0 ,255), # purple
              (255,0,255)] # pink

    color_index = 0
    current_color = 0

    # init MediaPipe Hands model
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        # open cam
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                continue

            # create canvas for drawing
            if canvas is None:
                canvas = np.zeros_like(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # change to rbg
            hand_results = hands.process(frame_rgb) # detect hands

            gesture_text = ""
            draw_point = None
            # detect each hand
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:

                    # draw hands
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # classify the gesture
                    fingers = what_finger_be_doing(hand_landmarks)
                    gesture = classify_gesture(fingers, hand_landmarks)
                    if gesture:
                        gesture_text = gesture

                    # fingertip
                    frame_w = frame.shape[1]
                    frame_h = frame.shape[0]
                    # find position
                    x = int(hand_landmarks.landmark[8].x * frame_w)
                    y = int(hand_landmarks.landmark[8].y * frame_h)

                    # flip text
                    x = frame_w - x  
                    draw_point = (x, y)

                    if gesture == "Open Palm":
                        # compute simple roll using wrist which is middle MCP
                        wrist = hand_landmarks.landmark[0]
                        middle_tip = hand_landmarks.landmark[12]
                        dx = middle_tip.x - wrist.x
                        # if open hand is tilted enough, switch color
                        if dx < -0.10:
                            color_index = 0
                        elif -0.10 <= dx < -0.05:
                            color_index = 1
                        elif -0.05 <= dx < 0.00:
                            color_index = 2
                        elif 0.00 <= dx < 0.05:
                            color_index = 3
                        elif 0.05 <= dx < 0.10:
                            color_index = 4
                        elif 0.10 <= dx < 0.15:
                            color_index = 5
                        elif 0.15 <= dx < 0.20:
                            color_index = 6
                        else:
                            color_index = 7
                        current_color = colors[color_index]


               # white, red, orange, yellow, green, blue, purple, pink 

            # flip frame
            frame = cv2.flip(frame, 1)
            canvas = cv2.flip(canvas, 1)

            # Draw text
            if gesture_text:
                cv2.putText(frame, gesture_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), # Don't forgest to change color!!!!
                    3)

            if gesture_text == "Eraser" and draw_point is not None:
                if prev_point is not None:
                    cv2.line(canvas, prev_point, draw_point, (0, 0, 0), 20)
                prev_point = draw_point
            elif gesture_text == "Pointing/Drawing" and draw_point is not None:
                if prev_point is not None:
                    cv2.line(canvas, prev_point, draw_point, current_color, 5)
                prev_point = draw_point
            elif gesture_text == "i in ASL":
                canvas[:] = 0
                prev_point = None
            else:
                prev_point = None

            # combine canvas and camera
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
            frame[mask] = canvas[mask]
            
            # make color wheel
            circle_radius = 20
            circle_x = frame.shape[1] - 40
            circle_y = 40
            cv2.circle(frame, (circle_x, circle_y), circle_radius, current_color, -1)
            cv2.rectangle(frame, (circle_x-circle_radius-2, circle_y-circle_radius-2), (circle_x+circle_radius+2, circle_y+circle_radius+2), (255,255,255), 1)
            
            # title
            cv2.imshow("Hand Tracking", frame)

            # unflip canvas so drawing maps onto the finger
            canvas = cv2.flip(canvas, 1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()

#===================
# MAIN FUNCTION
#===================
if __name__ == "__main__":
    import numpy as np
    run_tracking()

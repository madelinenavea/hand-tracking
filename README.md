# CyberHand Tracking

A real-time **hand-gesture-controlled drawing interface** built with **OpenCV** and **MediaPipe Hands**, styled with a cyberpunk HUD aesthetic. Draw, erase, pan, and change colors!

---

## Features

- Gesture-based drawing & erasing  
- Color selection by hand rotation  
- 13 supported hand gestures  
- Canvas clearing gesture  
- Cyberpunk UI with grids, glow effects, scanlines, and HUD overlays  
- Real-time webcam tracking  
- Multi-hand support  

---

## Demo Controls (Gestures)

| Gesture | Action |
|------|------|
| ☝️ **Pointer (index finger)** | Draw on canvas |
| ✌️ **Peace sign** | Peace gesture (visual feedback) |
| ✊ **Closed fist** | Pan / drag the canvas |
| ✋ **Open palm** | Change drawing color by tilting hand |
| 🤙 **Hang loose** | Gesture recognition feedback |
| 👎 **Thumbs down** | Gesture recognition feedback |
| 🤘 **Rock & Roll** | Gesture recognition feedback |
| 🤟 **I Love You** | Gesture recognition feedback |
| 👌 **OK sign** | Gesture recognition feedback |
| 🤏 **Pinch** | Gesture recognition feedback |
| ✌️ (tight fingers) | **Eraser (draw to erase)** |
| 🖕 **Middle finger** | Gesture recognition feedback |
| 🤙 *(ASL “I”)* | **Clear entire canvas** |

---

## Run the Script
```bash
git clone https://github.com/yourusername/cyberhand-interface.git
cd cyberhand-interface
pip install opencv-python mediapipe numpy
python main.py
```

## How It Works

- MediaPipe Hands tracks 21 hand landmarks per hand
- Finger extension, curl, and landmark distances are analyzed to classify gestures
- A transparent drawing canvas is layered on top of the live camera feed
- Cyberpunk UI elements (grids, glow, scanlines) are rendered in real time using OpenCV
- Hand orientation controls color selection dynamically

## Color Selection

- Hold an open palm
- Tilt your hand left or right
- The selected color updates in the COLOR MATRIX panel

## Tech Stack

- Python
- OpenCV
- MediaPipe Hands
- NumPy

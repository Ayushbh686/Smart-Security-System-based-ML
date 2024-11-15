import cv2
import face_recognition
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import telegram
import httpx
import io
import asyncio
import numpy as np
import threading
import os
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Load YOLO model for spoof detection
model = YOLO(r"train-final\weights\best.pt").to(device)
ip_address = ''

# Load face encodings for all users
def load_face_encodings(users_folder="Users"):
    known_face_encodings = []
    known_face_names = []

    for user_folder in os.listdir(users_folder):
        user_path = os.path.join(users_folder, user_folder)
        if os.path.isdir(user_path):
            for image_file in os.listdir(user_path):
                image_path = os.path.join(user_path, image_file)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(user_folder)

    return known_face_encodings, known_face_names

# Telegram bot details
bot_token = ""
chat_id = ""

client = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
bot = telegram.Bot(token=bot_token)

# Initialize door status
door_locked = True

# Event loop for asynchronous tasks
loop = asyncio.get_event_loop()

# Function to send alerts to Telegram
async def send_alert(image, message):
    try:
        image_pil = Image.fromarray(image)
        bio = io.BytesIO()
        image_pil.save(bio, format='JPEG')
        bio.seek(0)
        await bot.send_photo(chat_id=chat_id, photo=bio, caption=message)
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Function to listen for Telegram commands
async def listen_for_commands():
    global door_locked
    offset = 0
    while True:
        try:
            updates = await bot.get_updates(offset=offset)
            for update in updates:
                if update.message:
                    text = update.message.text.lower()
                    if "unlock" in text and door_locked:
                        door_locked = False
                        status_label.config(text="Status: Unlocked")
                        await bot.send_message(chat_id=chat_id, text="Door is now unlocked.")
                    elif "lock" in text and not door_locked:
                        door_locked = True
                        status_label.config(text="Status: Locked")
                        await bot.send_message(chat_id=chat_id, text="Door is now locked.")
                offset = update.update_id + 1
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in Telegram listener: {e}")
            await asyncio.sleep(5)

# Function to start the Telegram listener in a separate thread
def start_telegram_listener():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen_for_commands())

# Tkinter setup
root = tk.Tk()
root.title("Face Verification System")
cap = cv2.VideoCapture(ip_address)
cap.set(cv2.CAP_PROP_FPS, 20)  # Adjust frame rate if necessary

if not cap.isOpened():
    print("Error: Could not open the video stream.")
    exit()
status_label = tk.Label(root, text="Status: Locked", font=("Helvetica", 16))
status_label.pack()

# Function to show the video feed
def show_frame():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame, attempting to reconnect")
        cap.release()
        cap.open(ip_address)
        root.after(1000, show_frame)
        return
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    root.after(10, show_frame)


# Spoof detection function
# Spoof detection function
def spoof_detection(face_image):
    
    # Predict with YOLO model
    results = model.predict(face_image)

    # Access the probabilities object
    probs = results[0].probs

    # # Extract the actual probabilities using .item() method
    # real_prob = probs[0].item()  # Probability for 'real'
    # spoof_prob = probs[1].item()  # Probability for 'spoof'

    # # If the spoof probability is greater than 0.7, return an alert for spoof
    # if spoof_prob > real_prob:
    #     return spoof_prob, "spoof"

    # return real_prob, "real"
    top_prediction = probs.top1  # Index of top prediction (0: 'real', 1: 'spoof')
    top_confidence = probs.top1conf.item()  # Confidence score of the top prediction

    # If the top prediction is 'spoof' (index 1) and confidence is > 0.7
    if top_prediction == 1:
        return top_confidence, "spoof"

    else :
        return top_confidence, "real"


# Face verification function
def verify_face():
    global known_face_encodings, known_face_names
    
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image.")
        return

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        messagebox.showinfo("Result", "No face found.")
        return

    if len(face_encodings) > 1:
        loop.create_task(send_alert(frame, "Multiple faces detected!"))  # Run alert in the event loop
        messagebox.showinfo("Verification", "Multiple faces detected! Alert sent.")
        return

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)  # Increased tolerance for better recognition
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index] and face_distances[best_match_index] < 0.5:  # Loosened threshold
            name = known_face_names[best_match_index]

            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            spoof_score, result = spoof_detection(face_image)
            if result == "spoof":
                messagebox.showwarning("Alert", f"Spoof detected with score {spoof_score:.2f}. Locking door.")
                lock_door()
                loop.create_task(send_alert(face_image, f"Spoof detected! Score: {spoof_score:.2f}"))
            else:
                messagebox.showinfo("Result", f"Access granted to {name}.")
        else:
            loop.create_task(send_alert(frame, "Unrecognized face detected!"))
            messagebox.showwarning("Alert", "Unauthorized face detected.")
            lock_door()


# Placeholder function to simulate door locking
def lock_door():
    global door_locked
    door_locked = True
    print("Door locked.")

# Label for displaying the video
label = tk.Label(root)
label.pack()

# Load face encodings at the start
known_face_encodings, known_face_names = load_face_encodings()

# Verify button
verify_button = tk.Button(root, text="Verify", command=verify_face)
verify_button.pack()

# Start video feed
show_frame()

# Start Telegram listener thread
telegram_thread = threading.Thread(target=start_telegram_listener)
telegram_thread.start()

# Start Tkinter main loop
root.mainloop()

# Release video capture object
cap.release()
cv2.destroyAllWindows()

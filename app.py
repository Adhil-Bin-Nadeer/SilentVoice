import os
import cv2
import numpy as np
import base64
import json
import time
from datetime import datetime
import mediapipe as mp
import dlib
from scipy.spatial import distance
import tensorflow as tf
import logging
import sys
import pickle
from collections import deque
import threading
import requests # --- THIS IMPORT HAS BEEN ADDED ---

import warnings

from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit

# --- Suppress TensorFlow and related logs for a cleaner console ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable CUDA for dlib/Mediapipe if not explicitly using GPU for TF

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=Warning)

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR) # Suppress MediaPipe's extensive logging

if not sys.warnoptions:
    sys.warnoptions = ['ignore']

# Import Keras components from tf_keras
from tf_keras.models import Sequential, load_model
from tf_keras.layers import Dense, Dropout
from tf_keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler # Added for BlinkClassifier

class BlinkDetector:
    # --- MODIFIED SECTION START ---
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Downloading dlib shape predictor model...")
            self._download_shape_predictor()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Correctly reference the MediaPipe solutions module.
        # The FaceMesh object will be created temporarily inside the detection method.
        self.mp_face_mesh = mp.solutions.face_mesh

        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        self.LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        self.base_ear_thresh = 0.21
        self.current_ear_thresh = self.base_ear_thresh
        self.ear_history = deque(maxlen=30)
        self.EYE_AR_CONSEC_FRAMES = 1
        self.counter = 0
        self.blink_detected = False
        self.blink_start_time = 0
        self.brightness_history = deque(maxlen=10)
        self.use_enhancement = False
    # --- MODIFIED SECTION END ---

    def _download_shape_predictor(self):
        import urllib.request
        import bz2
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        try:
            urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
            with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as f_in:
                with open("shape_predictor_68_face_landmarks.dat", 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove("shape_predictor_68_face_landmarks.dat.bz2")
            print("Shape predictor model downloaded successfully!")
        except Exception as e:
            print(f"Failed to download shape predictor: {e}. Please ensure internet connection or provide the file manually.")
            raise

    def enhance_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.brightness_history.append(brightness)
        avg_brightness = np.mean(self.brightness_history)
        self.use_enhancement = avg_brightness < 80
        if self.use_enhancement:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if avg_brightness < 50:
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=25)
            return enhanced
        return frame

    def eye_aspect_ratio_dlib(self, eye_landmarks):
        try:
            A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
            C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
            if C == 0:
                return 0
            ear = (A + B) / (2.0 * C)
            return ear
        except IndexError:
            return 0

    def eye_aspect_ratio_mediapipe(self, eye_landmarks):
        try:
            A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
            C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
            if C == 0:
                return 0
            ear = (A + B) / (2.0 * C)
            return ear
        except IndexError:
            return 0

    def get_eye_landmarks_mediapipe(self, landmarks, eye_indices, image_width, image_height):
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            eye_points.append([x, y])
        return np.array(eye_points)

    def adapt_threshold(self, current_ear):
        if current_ear is not None:
            self.ear_history.append(current_ear)
            if len(self.ear_history) >= 10:
                recent_ears = list(self.ear_history)[-10:]
                mean_ear = np.mean(recent_ears)
                std_ear = np.std(recent_ears)
                adaptive_thresh = max(0.15, min(0.25, mean_ear - 2*std_ear))
                self.current_ear_thresh = 0.7 * self.current_ear_thresh + 0.3 * adaptive_thresh

    def detect_blink_dlib(self, frame):
        try:
            enhanced_frame = self.enhance_frame(frame)
            gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if len(faces) > 0:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in self.LEFT_EYE_POINTS])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in self.RIGHT_EYE_POINTS])
                left_ear = self.eye_aspect_ratio_dlib(left_eye)
                right_ear = self.eye_aspect_ratio_dlib(right_eye)
                ear = (left_ear + right_ear) / 2.0
                return ear, True
            return None, False
        except Exception as e:
            # print(f"dlib blink detection failed: {e}")
            return None, False

    # --- MODIFIED SECTION START ---
    def detect_blink_mediapipe(self, frame):
        try:
            # Use the 'with' statement for correct initialization and resource management
            with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3) as face_mesh:

                enhanced_frame = self.enhance_frame(frame)
                rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        h, w = enhanced_frame.shape[:2]
                        left_eye = self.get_eye_landmarks_mediapipe(face_landmarks, self.LEFT_EYE_EAR_INDICES, w, h)
                        right_eye = self.get_eye_landmarks_mediapipe(face_landmarks, self.RIGHT_EYE_EAR_INDICES, w, h)
                        left_ear = self.eye_aspect_ratio_mediapipe(left_eye)
                        right_ear = self.eye_aspect_ratio_mediapipe(right_eye)
                        ear = (left_ear + right_ear) / 2.0
                        return ear, True
            return None, False
        except Exception as e:
            # print(f"MediaPipe blink detection failed: {e}")
            return None, False
    # --- MODIFIED SECTION END ---

    def detect_blink(self, frame):
        blink_info = None
        current_ear = None
        ear, dlib_success = self.detect_blink_dlib(frame)
        if not dlib_success:
            ear, mp_success = self.detect_blink_mediapipe(frame)
            if not mp_success:
                # print("Warning: Both dlib and MediaPipe face detection failed.")
                return None, None
        current_ear = ear
        self.adapt_threshold(current_ear)
        if ear is not None and ear < self.current_ear_thresh:
            self.counter += 1
            if not self.blink_detected:
                self.blink_start_time = time.time()
                self.blink_detected = True
        else:
            if self.counter >= self.EYE_AR_CONSEC_FRAMES and self.blink_detected:
                blink_duration = time.time() - self.blink_start_time
                if 0.05 < blink_duration < 3.0: # Valid blink duration
                    blink_info = {
                        'duration': blink_duration,
                        'intensity': max(0.01, self.current_ear_thresh - min(ear if ear else 0, self.current_ear_thresh)),
                        'timestamp': time.time(),
                        'min_ear': ear if ear else 0,
                        'enhanced': self.use_enhancement
                    }
            self.counter = 0
            self.blink_detected = False
        return blink_info, current_ear

class MorseCodeDecoder:
    def __init__(self):
        self.morse_code_dict = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z', '.----': '1', '..---': '2', '...--': '3',
            '....-': '4', '.....': '5', '-....': '6', '--...': '7',
            '---..': '8', '----.': '9', '-----': '0', '--..--': ',',
            '.-.-.-': '.', '..--..': '?', '.----.': "'", '-.-.--': '!',
            '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&',
            '---...': ':', '-.-.-.': ';', '-...-': '=', '.-.-.': '+',
            '-....-': '-', '..--.-': '_', '.-..-.': '"', '...-..-': '$',
            '.--.-.': '@', '....-...': 'SOS'
        }

    def decode(self, morse_sequence):
        return self.morse_code_dict.get(morse_sequence, '?')

class UserManager:
    def __init__(self):
        self.users_dir = "users"
        self.users_file = os.path.join(self.users_dir, "users.json")
        self.ensure_directories()
        self.users = self.load_users()

    def ensure_directories(self):
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)

    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {self.users_file} is corrupted. Starting with empty user list.")
                return {}
        return {}

    def save_users(self):
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")

    def add_user(self, username):
        if username not in self.users:
            self.users[username] = {
                'created_date': datetime.now().isoformat(),
                'model_path': os.path.join(self.users_dir, f"{username}_model"),
                'trained': False
            }
            self.save_users()
            return True
        return False

    def get_user(self, username):
        return self.users.get(username)

    def mark_user_trained(self, username):
        if username in self.users:
            self.users[username]['trained'] = True
            self.save_users()

    def delete_user_model(self, username):
        if username in self.users:
            model_path = self.users[username]['model_path']
            model_file = f"{model_path}_model.h5"
            data_file = f"{model_path}_data.pkl"
            try:
                if os.path.exists(model_file):
                    os.remove(model_file)
                    print(f"Deleted model file: {model_file}")
                if os.path.exists(data_file):
                    os.remove(data_file)
                    print(f"Deleted data file: {data_file}")
                self.users[username]['trained'] = False
                self.save_users()
                return True
            except PermissionError as e:
                print(f"Permission error deleting model for {username}: {e}")
                return False
            except Exception as e:
                print(f"Error deleting model for {username}: {e}")
                return False
        print(f"User {username} not found.")
        return False

    def list_users(self):
        return list(self.users.keys())

class BlinkClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.dot_threshold = 0.4 # Default backup threshold

    def create_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(4,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare_features(self, blink_data):
        features = []
        for blink in blink_data:
            feature = [
                blink['duration'],
                blink['intensity'],
                blink.get('min_ear', 0.2),
                1.0 / (blink['duration'] + 0.001)
            ]
            features.append(feature)
        return np.array(features)

    def train(self, dot_blinks, dash_blinks):
        print(f"Training with {len(dot_blinks)} dots and {len(dash_blinks)} dashes")

        dot_durations = [b['duration'] for b in dot_blinks]
        dash_durations = [b['duration'] for b in dash_blinks]

        if not dot_durations or not dash_durations:
            print("Insufficient data for training. Need both dot and dash blinks.")
            self.model = None
            self.scaler = None
            return None, None

        dot_mean = np.mean(dot_durations)
        dash_mean = np.mean(dash_durations)
        print(f"Dot mean duration: {dot_mean:.3f}s, Dash mean duration: {dash_mean:.3f}s")

        if dash_mean > dot_mean:
            self.dot_threshold = (max(dot_durations) + min(dash_durations)) / 2
        else:
            self.dot_threshold = 0.4
        print(f"Backup threshold set to: {self.dot_threshold:.3f}s")

        dot_features = self.prepare_features(dot_blinks)
        dash_features = self.prepare_features(dash_blinks)

        X = np.vstack([dot_features, dash_features])
        y = np.hstack([np.zeros(len(dot_features)), np.ones(len(dash_features))])

        if np.isnan(X).any():
            print("Warning: NaN values found in features, replacing with 0")
            X = np.nan_to_num(X)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = self.create_model()
        try:
            self.model.fit(
                X_scaled, y,
                epochs=50,
                batch_size=min(8, len(X)) if len(X) > 0 else 1,
                verbose=0,
                validation_split=0.2 if len(X) > 5 else 0
            )
            loss, accuracy = self.model.evaluate(X_scaled, y, verbose=0)
            print(f"Training completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            return loss, accuracy
        except Exception as e:
            print(f"Model training failed: {e}")
            self.model = None
            return 0, 0.5

    def predict(self, blink_data):
        if self.model is not None and self.scaler is not None:
            try:
                features = self.prepare_features([blink_data])
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled, verbose=0)[0][0]
                return 'dash' if prediction > 0.5 else 'dot'
            except Exception as e:
                print(f"Model prediction failed: {e}, using duration threshold as fallback.")
        return 'dash' if blink_data['duration'] > self.dot_threshold else 'dot'

    def save_model(self, filepath):
        try:
            if self.model is not None:
                self.model.save(f"{filepath}_model.h5")
            model_data = {
                'scaler': self.scaler,
                'dot_threshold': self.dot_threshold,
                'has_model': self.model is not None
            }
            with open(f"{filepath}_data.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model and scaler saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath):
        try:
            with open(f"{filepath}_data.pkl", 'rb') as f:
                model_data = pickle.load(f)

            if not isinstance(model_data, dict) or 'scaler' not in model_data:
                print(f"Error: {filepath}_data.pkl is corrupted or invalid.")
                return False

            self.scaler = model_data['scaler']
            self.dot_threshold = model_data['dot_threshold']

            if model_data.get('has_model', False):
                try:
                    self.model = load_model(f"{filepath}_model.h5")
                    print("Neural network model loaded successfully.")
                except Exception as keras_e:
                    print(f"Neural network model file not found or corrupted ({filepath}_model.h5): {keras_e}. Will use threshold method.")
                    self.model = None
            else:
                self.model = None
                print("No neural network model found, using threshold method for classification.")
            return True
        except FileNotFoundError:
            print(f"Model files not found for {filepath}. User needs training.")
            self.model = None
            self.scaler = None
            return False
        except Exception as e:
            print(f"Error loading model for {filepath}: {e}")
            self.model = None
            self.scaler = None
            return False

class MorseCodeCommunicator:
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.morse_decoder = MorseCodeDecoder()
        self.user_manager = UserManager()
        self.current_user = None
        self.classifier = BlinkClassifier()
        self.current_morse_sequence = ""
        self.blink_buffer = deque(maxlen=100)
        self.last_blink_time = 0
        self.last_letter_time = 0
        self.LETTER_PAUSE = 5.0
        self.SPACE_PAUSE = 5.0
        self.BLINK_COOLDOWN = 1.0

    def train_user(self, username):
        print(f"\n=== Training Model for {username} ===")
        print("Position yourself in front of the camera with good lighting.")
        cap = None
        try:
            for cam_index in [0, 1]:
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    print(f"Camera {cam_index} opened successfully.")
                    break
            if not cap or not cap.isOpened():
                raise Exception("Could not open any camera. Check connections or permissions.")

            print("\nTesting camera and face detection (press SPACE to continue, ESC to cancel)...")
            current_ear_test = None
            for i in range(100):
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Could not read from camera during test. Check camera feed.")

                temp_blink_info, temp_ear = self.blink_detector.detect_blink(frame)
                if temp_ear is not None:
                    current_ear_test = temp_ear

                display_frame = frame.copy()
                if current_ear_test is not None:
                    cv2.putText(display_frame, f"EAR: {current_ear_test:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face detected! Press SPACE to continue...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No face detected - adjust position", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, "Press SPACE to skip test, ESC to cancel.", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow('Camera Test - Training Setup', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == 27:
                    print("Camera test cancelled by user.")
                    return False

            if current_ear_test is None:
                print("Warning: Face not consistently detected during camera test. Proceeding anyway.")

            print("\n--- Collecting DOT blinks ---")
            print("Perform SHORT, QUICK blinks (aim for 0.1-0.4 seconds)")
            dot_blinks = []
            if not self.collect_training_data(cap, dot_blinks, "dot", 15):
                raise Exception("Dot blink collection cancelled or failed.")

            print("\n--- Collecting DASH blinks ---")
            print("Perform LONG, DELIBERATE blinks (aim for 0.5-3.0 seconds)")
            dash_blinks = []
            if not self.collect_training_data(cap, dash_blinks, "dash", 15):
                raise Exception("Dash blink collection cancelled or failed.")

            dot_durations = [b['duration'] for b in dot_blinks]
            dash_durations = [b['duration'] for b in dash_blinks]

            print(f"\nCollected {len(dot_blinks)} dots and {len(dash_blinks)} dashes")

            if dot_durations:
                print(f"Dot durations: min={min(dot_durations):.3f}s, max={max(dot_durations):.3f}s (avg: {np.mean(dot_durations):.3f}s)")
            if dash_durations:
                print(f"Dash durations: min={min(dash_durations):.3f}s, max={max(dash_durations):.3f}s (avg: {np.mean(dash_durations):.3f}s)")

            if len(dot_blinks) >= 8 and len(dash_blinks) >= 8:
                print("\nTraining model with collected data...")
                user_info = self.user_manager.get_user(username)

                loss, accuracy = self.classifier.train(dot_blinks, dash_blinks)

                if loss is not None and accuracy is not None and accuracy > 0.7:
                    self.classifier.save_model(user_info['model_path'])
                    self.user_manager.mark_user_trained(username)
                    avg_dot_duration = np.mean(dot_durations)
                    avg_dash_duration = np.mean(dash_durations)
                    self.LETTER_PAUSE = max(1.0, avg_dot_duration * 4)
                    self.SPACE_PAUSE = max(2.0, avg_dash_duration * 2)
                    self.BLINK_COOLDOWN = max(0.5, avg_dot_duration * 1.5)
                    print(f"Adaptive timings: LETTER_PAUSE={self.LETTER_PAUSE:.2f}s, SPACE_PAUSE={self.SPACE_PAUSE:.2f}s, BLINK_COOLDOWN={self.BLINK_COOLDOWN:.2f}s")

                    print(f"Training completed successfully! Accuracy: {accuracy:.2%}")
                    return True
                else:
                    print(f"Model training insufficient or low accuracy ({accuracy:.2%}). Please try again.")
                    return False
            else:
                print(f"Insufficient data: Collected {len(dot_blinks)} dots and {len(dash_blinks)} dashes.")
                return False
        except Exception as e:
            print(f"Training error: {e}")
            return False
        finally:
            if cap is not None:
                cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass

    def collect_training_data(self, cap, data_list, blink_type, target_count):
        collecting = False
        collected_count = 0
        cooldown_time = 0
        last_duration = None
        instruction_text = f"Perform {blink_type.upper()} blinks: {'SHORT (0.1-0.4s)' if blink_type == 'dot' else 'LONG (0.5-3.0s)'}"

        start_instruction_time = time.time()

        while collected_count < target_count:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera during data collection.")
                return False

            current_time = time.time()
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            cv2.putText(display_frame, f"Collecting {blink_type.upper()} blinks: {collected_count}/{target_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, instruction_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if not collecting and current_time > cooldown_time:
                cv2.putText(display_frame, "Press SPACE to start collecting next blink",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elif collecting:
                cv2.putText(display_frame, f"PERFORM BLINK NOW!",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press SPACE to cancel this attempt",
                            (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif current_time <= cooldown_time:
                remaining = int(cooldown_time - current_time) + 1
                cv2.putText(display_frame, f"Wait {remaining} seconds for cooldown...",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            if last_duration is not None:
                cv2.putText(display_frame, f"Last blink: {last_duration:.3f}s",
                            (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(display_frame, "Press ESC to cancel training",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if collecting:
                blink_info, current_ear = self.blink_detector.detect_blink(frame)
                if current_ear is not None:
                    cv2.putText(display_frame, f"EAR: {current_ear:.3f}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if blink_info:
                    duration = blink_info['duration']
                    last_duration = duration
                    valid = False
                    if blink_type == "dot" and 0.05 <= duration <= 0.4:
                        data_list.append(blink_info)
                        collected_count += 1
                        cooldown_time = current_time + 1.5
                        valid = True
                        print(f"Collected {blink_type.upper()} #{collected_count}: {duration:.3f}s")
                    elif blink_type == "dash" and 0.5 <= duration <= 3.0:
                        data_list.append(blink_info)
                        collected_count += 1
                        cooldown_time = current_time + 2.0
                        valid = True
                        print(f"Collected {blink_type.upper()} #{collected_count}: {duration:.3f}s")

                    if not valid:
                        print(f"Invalid duration for {blink_type.upper()} ({duration:.3f}s). Please try again.")
                        if blink_type == "dot":
                            if duration > 0.4: print("  Hint: Too long. Aim for a quicker blink (0.1-0.4s).")
                            elif duration < 0.05: print("  Hint: Too short. Aim for a slightly longer blink (0.1-0.4s).")
                        elif blink_type == "dash":
                            if duration < 0.5: print("  Hint: Too short. Hold eyes closed longer (0.5-3.0s).")
                            elif duration > 3.0: print("  Hint: Too long. Try a shorter, more deliberate blink (3.0s max).")
                        cooldown_time = max(cooldown_time, current_time + 1.0)

                    collecting = False

            cv2.imshow('Training Data Collection', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and current_time > cooldown_time:
                collecting = not collecting
                if collecting:
                    print(f"Ready to collect {blink_type} blink #{collected_count + 1}. PERFORM BLINK NOW!")
            elif key == 27:
                print("Training data collection cancelled by user.")
                return False
        return True

# --- Flask Application Setup ---
app = Flask(__name__,
            template_folder='.',
            static_folder='.',
            static_url_path='/'
           )
app.config['SECRET_KEY'] = 'a_very_secret_key_that_you_should_change_for_production'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 * 1024 * 1024)

# Global instance of the MorseCodeCommunicator to maintain state across requests/sockets
communicator = MorseCodeCommunicator()

# Thread-safe queue for incoming frames and flag for processing thread control
frame_queue = deque()
processing_active = False
processing_thread = None

# Global variable to track the current mode for backend processing
current_backend_mode = 'idle' # 'idle', 'navigation', 'morse_input'

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main communication page (index.html)."""
    return render_template('index.html')

@app.route('/quick_messages')
def quick_messages():
    """Renders the quick messages page (quick_messages.html)."""
    return render_template('quick_messages.html')

@app.route('/users')
def list_users_api():
    """API endpoint to list all registered users and their training status."""
    users = communicator.user_manager.list_users()
    user_data = {}
    for user in users:
        info = communicator.user_manager.get_user(user)
        user_data[user] = {'trained': info['trained']}
    return jsonify(user_data)

@app.route('/create_user/<username>')
def create_user_api(username):
    """API endpoint to create a new user."""
    if communicator.user_manager.add_user(username):
        print(f"User '{username}' created.")
        return jsonify({"status": "success", "message": f"User '{username}' created successfully!"})
    else:
        print(f"User '{username}' already exists.")
        return jsonify({"status": "error", "message": f"User '{username}' already exists."}), 409

@app.route('/flappy_bird')
def flappy_bird():
    """Renders the Flappy Bird game page (flappy_bird.html)."""
    return render_template('flappy_bird.html')

# --- Socket.IO Event Handlers ---

@socketio.on('connect')
def handle_connect():
    """Handles new WebSocket client connections."""
    print(f'Client connected: {request.sid}')
    global communicator
    communicator.current_morse_sequence = ""
    communicator.last_blink_time = 0
    communicator.last_letter_time = 0

@socketio.on('disconnect')
def handle_disconnect():
    """Handles WebSocket client disconnections."""
    print(f'Client disconnected: {request.sid}')
    global processing_active, current_backend_mode
    processing_active = False
    current_backend_mode = 'idle' # Reset mode on disconnect

@socketio.on('select_user')
def handle_select_user(data):
    """Socket.IO event to set the current active user and load their ML model."""
    username = data.get('username')
    if not username:
        return {'status': 'error', 'message': 'No username provided for selection.'}

    user_info = communicator.user_manager.get_user(username)
    if not user_info:
        print(f"User '{username}' not found on select_user event.")
        return {'status': 'error', 'message': f"User '{username}' not found."}

    communicator.current_user = username

    if user_info['trained']:
        if communicator.classifier.load_model(user_info['model_path']):
            print(f"User '{username}' selected and model loaded via Socket.IO.")
            return {'status': 'success', 'message': f"User '{username}' selected and model loaded."}
        else:
            print(f"Failed to load model for user '{username}'. User needs retraining.")
            communicator.user_manager.users[username]['trained'] = False
            communicator.user_manager.save_users()
            return {'status': 'error', 'message': f"Failed to load model for '{username}'. Needs training?"}
    else:
        print(f"User '{username}' selected but not trained.")
        communicator.classifier.model = None
        communicator.classifier.scaler = None
        return {'status': 'warning', 'message': f"User '{username}' selected but not trained. Please train the model."}

@socketio.on('start_stream')
def start_stream():
    """Starts the frame processing thread on the backend."""
    if communicator.current_user is None:
        emit('status', {'message': 'No user selected. Please select a user first.'}, room=request.sid)
        return

    user_info = communicator.user_manager.get_user(communicator.current_user)
    if user_info is None:
        emit('status', {'message': f"Error: User '{communicator.current_user}' info not found."}, room=request.sid)
        return

    if not user_info['trained'] or communicator.classifier.model is None or communicator.classifier.scaler is None:
        emit('status', {'message': f"User '{communicator.current_user}' is not trained or model not loaded. Please train the model first."}, room=request.sid)
        return

    print(f"Starting video stream processing for user '{communicator.current_user}'...")
    global processing_active, processing_thread
    processing_active = True

    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_frames, args=(request.sid,))
        processing_thread.daemon = True
        processing_thread.start()
    emit('stream_started', {'message': 'Processing started.'}, room=request.sid)

@socketio.on('stop_stream')
def stop_stream():
    """Stops the frame processing thread on the backend."""
    print("Stopping video stream processing...")
    global processing_active, current_backend_mode
    processing_active = False
    current_backend_mode = 'idle'
    emit('stream_stopped', {'message': 'Processing stopped.'}, room=request.sid)

@socketio.on('send_quick_message')
def handle_quick_message(data):
    """Handles quick messages sent from the frontend."""
    message = data.get('message')
    if message:
        print(f"Quick message received from client ({request.sid}): \"{message}\"")
        emit('status', {'message': f"Quick message sent: '{message}'"}, room=request.sid)
    else:
        emit('status', {'message': "No message provided for quick send."}, room=request.sid)

@socketio.on('frame')
def handle_frame(data):
    """Receives video frames from the frontend and adds them to a processing queue."""
    img_bytes = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame_queue.append(frame)

@socketio.on('set_mode')
def set_mode(data):
    """Sets the operating mode for the backend processing thread."""
    global current_backend_mode
    mode = data.get('mode')
    if mode in ['idle', 'navigation', 'morse_input']:
        current_backend_mode = mode
        print(f"Backend mode set to: {current_backend_mode}")
        if current_backend_mode == 'navigation' or current_backend_mode == 'idle':
            communicator.current_morse_sequence = ""
            communicator.last_blink_time = 0
            communicator.last_letter_time = 0
        emit('status', {'message': f'Mode changed to {current_backend_mode}'}, room=request.sid)
    else:
        emit('status', {'message': f'Invalid mode: {mode}'}, room=request.sid)

@socketio.on('room_command')
def handle_room_command(data):
    """Handles commands for room devices sent from the frontend."""
    device = data.get('device')
    action = data.get('action')
    print(f"Room command received: Control '{device}' to be '{action}'")

    # --- IMPORTANT: REPLACE WITH YOUR ESP32's IP ADDRESS ---
    ESP32_IP = "192.168.1.102"

    if device and action:
        try:
            requests.get(f"http://{ESP32_IP}/{device}/{action}", timeout=3)
            emit('status', {'message': f'Command sent: Turn {device} {action}'}, room=request.sid)
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ESP32: {e}")
            emit('status', {'message': 'Error: Could not connect to device.'}, room=request.sid)

# --- Core Frame Processing Thread ---
def process_frames(sid):
    """
    Runs in a separate thread to continuously process frames from the queue,
    detect blinks, classify them, decode Morse code, and send updates to the client.
    """
    global processing_active, current_backend_mode

    communicator.current_morse_sequence = ""
    communicator.last_blink_time = 0
    communicator.last_letter_time = 0
    blink_cooldown_end = 0

    print(f"Processing thread started for session {sid}...")

    while processing_active:
        if not frame_queue:
            time.sleep(0.01)
            continue

        frame = frame_queue.popleft()
        current_time = time.time()
        blink_info, current_ear = None, None

        if current_time >= blink_cooldown_end:
            blink_info, current_ear = communicator.blink_detector.detect_blink(frame)
            if blink_info:
                blink_type = communicator.classifier.predict(blink_info)
                if current_backend_mode != 'idle':
                    socketio.emit('blink_detected', {'type': blink_type}, room=sid)
                blink_cooldown_end = current_time + communicator.BLINK_COOLDOWN

        if current_backend_mode == 'morse_input':
            if blink_info:
                communicator.current_morse_sequence += ('.' if blink_type == 'dot' else '-')
                communicator.last_blink_time = current_time
                print(f"Backend detected {blink_type} ({blink_info['duration']:.3f}s): {communicator.current_morse_sequence}")

            if communicator.current_morse_sequence and \
               communicator.last_blink_time > 0 and \
               (current_time - communicator.last_blink_time) > communicator.LETTER_PAUSE:

                letter = communicator.morse_decoder.decode(communicator.current_morse_sequence)
                print(f"Backend decoded: {communicator.current_morse_sequence} -> {letter}")

                if 'message_accum' not in communicator.__dict__:
                    communicator.message_accum = ""
                communicator.message_accum += letter

                socketio.emit('update_ui', {
                    'message': communicator.message_accum,
                    'morse_sequence': "",
                    'status': f"Letter '{letter}' decoded.",
                    'cooldown_percent': 0,
                    'letter_timer': 0,
                    'space_timer': 0
                }, room=sid)

                communicator.current_morse_sequence = ""
                communicator.last_letter_time = current_time

            elif not communicator.current_morse_sequence and \
                 communicator.last_letter_time > 0 and \
                 current_time >= blink_cooldown_end and \
                 (current_time - communicator.last_letter_time) > communicator.SPACE_PAUSE:

                if 'message_accum' not in communicator.__dict__:
                    communicator.message_accum = ""
                if communicator.message_accum and not communicator.message_accum.endswith(' '):
                    communicator.message_accum += " "

                socketio.emit('update_ui', {
                    'message': communicator.message_accum,
                    'morse_sequence': "",
                    'status': "Space added.",
                    'cooldown_percent': 0,
                    'letter_timer': 0,
                    'space_timer': 0
                }, room=sid)

                print("Backend added space.")
                communicator.last_letter_time = 0
        else:
            communicator.current_morse_sequence = ""
            communicator.last_blink_time = 0
            communicator.last_letter_time = 0

        elapsed_time = int(current_time - communicator.last_blink_time) if communicator.last_blink_time > 0 else 0
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60
        timer_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        message_accum = getattr(communicator, 'message_accum', '')

        socketio.emit('update_ui', {
            'message': message_accum,
            'morse_sequence': communicator.current_morse_sequence,
            'status': f"Status: {current_backend_mode.replace('_', ' ').title()}",
            'cooldown_percent': 1 - (max(0, blink_cooldown_end - current_time) / communicator.BLINK_COOLDOWN) if current_backend_mode != 'idle' else 0,
            'letter_timer': max(0, communicator.LETTER_PAUSE - (current_time - communicator.last_blink_time)) if communicator.current_morse_sequence else 0,
            'space_timer': max(0, communicator.SPACE_PAUSE - (current_time - communicator.last_letter_time)) if communicator.last_letter_time > 0 and not communicator.current_morse_sequence else 0,
            'timer': timer_str
        }, room=sid)

    print("Processing thread stopped.")

# --- Main Application Entry Point ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")

    socketio.run(app, debug=False, allow_unsafe_werkzeug=True, host='127.0.0.1', port=5000)

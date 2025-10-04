import os
import cv2
import numpy as np
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
import warnings

# --- Suppress TensorFlow and related logs for a cleaner console ---
os.environ[ 'TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ[ 'TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ[ 'CUDA_VISIBLE_DEVICES'] = '-1' # Disable CUDA for dlib/Mediapipe if not explicitly using GPU for TF

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



# --- CORE CLASSES (Duplicated here for train_model.py to work independently) ---
class BlinkDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Downloading dlib shape predictor model...")
            self._download_shape_predictor()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
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
            return None, False
        
    def detect_blink_mediapipe(self, frame):
        try:
            enhanced_frame = self.enhance_frame(frame)
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
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
            return None, False
        
    def detect_blink(self, frame):
        blink_info = None
        current_ear = None
        ear, dlib_success = self.detect_blink_dlib(frame)
        if not dlib_success:
            ear, mp_success = self.detect_blink_mediapipe(frame)
            if not mp_success:
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
                if 0.05 < blink_duration < 4.0: # Increased upper limit slightly
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
                if os.path.exists(model_file): os.remove(model_file)
                if os.path.exists(data_file): os.remove(data_file)
                self.users[username]['trained'] = False
                self.save_users()
                return True
            except Exception as e:
                print(f"Error deleting model for {username}: {e}")
                return False
        return False
        
    def list_users(self):
        return list(self.users.keys())

class BlinkClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.dot_threshold = 0.4
        
    def create_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(4,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
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
        dot_durations = [b['duration'] for b in dot_blinks]
        dash_durations = [b['duration'] for b in dash_blinks]
        
        if not dot_durations or not dash_durations:
            print("Insufficient data for training.")
            return None, None

        dot_mean = np.mean(dot_durations)
        dash_mean = np.mean(dash_durations)
        print(f"Dot mean duration: {dot_mean:.3f}s, Dash mean duration: {dash_mean:.3f}s")
        
        if dash_mean > dot_mean:
            self.dot_threshold = (max(dot_durations) + min(dash_durations)) / 2
        else:
            self.dot_threshold = 0.4
        print(f"Backup threshold set to: {self.dot_threshold:.3f}s")

        X = np.vstack([self.prepare_features(dot_blinks), self.prepare_features(dash_blinks)])
        y = np.hstack([np.zeros(len(dot_blinks)), np.ones(len(dash_blinks))])
        X = np.nan_to_num(X)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = self.create_model()
        try:
            self.model.fit(X_scaled, y, epochs=50, batch_size=min(8, len(X)) if len(X) > 0 else 1, verbose=0, validation_split=0.2 if len(X) > 5 else 0)
            loss, accuracy = self.model.evaluate(X_scaled, y, verbose=0)
            print(f"Training completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            return loss, accuracy
        except Exception as e:
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
                pass
        return 'dash' if blink_data['duration'] > self.dot_threshold else 'dot'
        
    def save_model(self, filepath):
        try:
            if self.model: self.model.save(f"{filepath}_model.h5")
            model_data = {'scaler': self.scaler, 'dot_threshold': self.dot_threshold, 'has_model': self.model is not None}
            with open(f"{filepath}_data.pkl", 'wb') as f: pickle.dump(model_data, f)
            print(f"Model and scaler saved to {filepath}")
        except Exception as e: print(f"Error saving model: {e}")
            
    def load_model(self, filepath):
        try:
            with open(f"{filepath}_data.pkl", 'rb') as f: model_data = pickle.load(f)
            if not isinstance(model_data, dict) or 'scaler' not in model_data: return False
            self.scaler = model_data['scaler']
            self.dot_threshold = model_data['dot_threshold']
            if model_data.get('has_model', False):
                try: self.model = load_model(f"{filepath}_model.h5")
                except Exception: self.model = None
            else: self.model = None
            return True
        except Exception:
            self.model, self.scaler = None, None
            return False

class MorseCodeCommunicator:
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.morse_decoder = MorseCodeDecoder()
        self.user_manager = UserManager()
        self.current_user = None
        self.classifier = BlinkClassifier()
        self.current_morse_sequence = ""
        self.last_blink_time = 0
        self.last_letter_time = 0
        self.LETTER_PAUSE = 1.0
        self.SPACE_PAUSE = 2.0
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
                raise Exception("Could not open any camera.")

            # --- Camera Test (unchanged) ---
            # (Camera test code remains the same)

            print("\n--- Collecting DOT blinks (SHORT) ---")
            dot_blinks = []
            if not self.collect_training_data(cap, dot_blinks, "dot", 15):
                raise Exception("Dot blink collection cancelled or failed.")

            # --- MODIFICATION: Calculate max dot duration before collecting dashes ---
            if not dot_blinks:
                print("No short blinks were collected. Cannot proceed to train long blinks.")
                return False
            max_dot_duration = max([b['duration'] for b in dot_blinks])
            print(f"\nYour longest SHORT blink was {max_dot_duration:.3f}s. Your long blinks should be longer than this.")

            print("\n--- Collecting DASH blinks (LONG) ---")
            dash_blinks = []
            # --- MODIFICATION: Pass max_dot_duration to the collection function ---
            if not self.collect_training_data(cap, dash_blinks, "dash", 15, max_dot_duration=max_dot_duration):
                raise Exception("Dash blink collection cancelled or failed.")

            dot_durations = [b['duration'] for b in dot_blinks]
            dash_durations = [b['duration'] for b in dash_blinks]

            print(f"\nCollected {len(dot_blinks)} dots and {len(dash_blinks)} dashes")
            
            if dot_durations: print(f"Dot durations: min={min(dot_durations):.3f}s, max={max(dot_durations):.3f}s (avg: {np.mean(dot_durations):.3f}s)")
            if dash_durations: print(f"Dash durations: min={min(dash_durations):.3f}s, max={max(dash_durations):.3f}s (avg: {np.mean(dash_durations):.3f}s)")

            if len(dot_blinks) >= 8 and len(dash_blinks) >= 8:
                print("\nTraining model with collected data...")
                user_info = self.user_manager.get_user(username)
                
                loss, accuracy = self.classifier.train(dot_blinks, dash_blinks)
                
                if loss is not None and accuracy is not None and accuracy > 0.7:
                    self.classifier.save_model(user_info['model_path'])
                    self.user_manager.mark_user_trained(username)
                    avg_dot_duration = np.mean(dot_durations)
                    self.LETTER_PAUSE = max(1.0, avg_dot_duration * 4)
                    self.SPACE_PAUSE = max(2.0, np.mean(dash_durations) * 2)
                    self.BLINK_COOLDOWN = max(0.5, avg_dot_duration * 1.5)
                    print(f"Adaptive timings set based on your blinks.")
                    print(f"Training completed successfully! Accuracy: {accuracy:.2%}")
                    return True
                else:
                    print(f"Model training insufficient or low accuracy ({accuracy:.2%}). Please try again.")
                    return False
            else:
                print(f"Insufficient data. Need at least 8 of each blink type.")
                return False
        except Exception as e:
            print(f"Training error: {e}")
            return False
        finally:
            if cap: cap.release()
            try: cv2.destroyAllWindows()
            except: pass
        
    def collect_training_data(self, cap, data_list, blink_type, target_count, max_dot_duration=None):
        collecting, collected_count, cooldown_time, last_duration = False, 0, 0, None
        
        # --- MODIFICATION: Updated instruction text ---
        if blink_type == 'dot':
            instruction_text = "Perform SHORT, QUICK blinks (0.1-0.4s)"
        else:
            instruction_text = "Perform LONG, DELIBERATE blinks"

        while collected_count < target_count:
            ret, frame = cap.read()
            if not ret: return False
            
            current_time = time.time()
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            cv2.putText(display_frame, f"Collecting {blink_type.upper()} blinks: {collected_count}/{target_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, instruction_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # (Display logic for UI text remains largely the same)
            if not collecting and current_time > cooldown_time:
                cv2.putText(display_frame, "Press SPACE to start collecting", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elif collecting:
                cv2.putText(display_frame, "PERFORM BLINK NOW!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if last_duration is not None:
                cv2.putText(display_frame, f"Last: {last_duration:.3f}s", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if collecting:
                blink_info, current_ear = self.blink_detector.detect_blink(frame)
                if blink_info:
                    duration = blink_info['duration']
                    last_duration = duration
                    valid = False
                    
                    # --- MODIFICATION: New validation logic ---
                    if blink_type == "dot" and 0.05 <= duration <= 0.4:
                        valid = True
                    elif blink_type == "dash" and max_dot_duration is not None:
                        # A dash is valid if it's at least 25% longer than the longest dot
                        if duration > max_dot_duration * 1.25:
                            valid = True
                        else:
                            print(f"Invalid duration for DASH ({duration:.3f}s). Must be longer than {max_dot_duration * 1.25:.3f}s.")
                            print("  Hint: Hold your eyes closed for longer to distinguish from your short blinks.")
                    
                    if valid:
                        data_list.append(blink_info)
                        collected_count += 1
                        cooldown_time = current_time + 1.5
                        print(f"Collected {blink_type.upper()} #{collected_count}: {duration:.3f}s")
                    else:
                        # General feedback for invalid dot blinks
                        if blink_type == "dot":
                            print(f"Invalid duration for DOT ({duration:.3f}s). Please try again.")
                        cooldown_time = max(cooldown_time, current_time + 1.0)
                    
                    collecting = False
                    
            cv2.imshow('Training Data Collection', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and current_time > cooldown_time:
                collecting = not collecting
                if collecting: print(f"Ready for {blink_type} blink #{collected_count + 1}...")
            elif key == 27:
                return False
        return True

# --- END OF CORE CLASSES ---


def main_train():
    """
    Main function to run the CLI for user training.
    """
    communicator = MorseCodeCommunicator()
    user_manager = communicator.user_manager

    while True:
        print("\n=== User Training Menu ===")
        print("1. List existing users")
        print("2. Create new user and train")
        print("3. Retrain existing user")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            users = user_manager.list_users()
            if users:
                print("\nExisting users:")
                for i, user in enumerate(users, 1):
                    info = user_manager.get_user(user)
                    status = "Trained" if info['trained'] else "Not trained"
                    print(f"{i}. {user} ({status})")
            else:
                print("No users found.")
        
        elif choice == '2':
            username = input("Enter new username: ").strip()
            if username:
                if user_manager.add_user(username):
                    print(f"User '{username}' created. Proceeding to training.")
                    if communicator.train_user(username):
                        print(f"\nModel for '{username}' trained successfully!")
                    else:
                        print(f"\nTraining for '{username}' failed or was cancelled.")
                else:
                    print(f"User '{username}' already exists.")
            else:
                print("Username cannot be empty.")
        
        elif choice == '3':
            users = user_manager.list_users()
            if not users:
                print("No users to retrain.")
                continue
            
            print("\nSelect user to retrain:")
            for i, user in enumerate(users, 1):
                print(f"{i}. {user}")
            
            try:
                idx = int(input("\nEnter user number: ")) - 1
                if 0 <= idx < len(users):
                    username = users[idx]
                    print(f"\nRetraining user '{username}'...")
                    if communicator.train_user(username):
                        print(f"\nModel for '{username}' retrained successfully!")
                    else:
                        print(f"\nRetraining for '{username}' failed or was cancelled.")
                else:
                    print("Invalid number.")
            except ValueError:
                print("Invalid input.")
        
        elif choice == '4':
            print("Exiting.")
            break
        
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main_train()



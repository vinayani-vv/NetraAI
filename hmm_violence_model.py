"""
HMM-based Violence Detection Model
Uses Hidden Markov Models to capture temporal patterns in video sequences
"""

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
import os
from pathlib import Path

class HMMBasedViolenceDetector:
    """
    Hidden Markov Model for violence detection in video sequences.
    Models temporal patterns of violent vs peaceful behavior.
    """
    
    def __init__(self, n_components=4, covariance_type='full', n_iter=100):
        """
        Initialize HMM-based violence detector.
        
        Args:
            n_components: Number of hidden states in HMM
            covariance_type: Type of covariance matrix ('full', 'diag', 'spherical', 'tied')
            n_iter: Maximum number of iterations for EM algorithm
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        
        # HMM models for each class
        self.violent_model = None
        self.peaceful_model = None
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features_from_frame(self, frame_result, yolo_model):
        """
        Extract features from YOLO frame prediction.
        
        Args:
            frame_result: YOLO prediction result for a single frame
            yolo_model: YOLO model instance
            
        Returns:
            numpy array of features
        """
        features = []
        
        # Detection-based features
        if hasattr(frame_result, 'boxes') and len(frame_result.boxes) > 0:
            boxes = frame_result.boxes.xyxy.cpu().numpy()
            confs = frame_result.boxes.conf.cpu().numpy()
            classes = frame_result.boxes.cls.cpu().numpy()
            
            # Number of detections
            features.append(len(boxes))
            
            # Average confidence
            features.append(np.mean(confs) if len(confs) > 0 else 0)
            
            # Max confidence
            features.append(np.max(confs) if len(confs) > 0 else 0)
            
            # Violence-related class detection (assuming class 0 is violent)
            violent_count = sum(1 for c in classes if c == 0)
            peaceful_count = sum(1 for c in classes if c == 1)
            features.append(violent_count)
            features.append(peaceful_count)
            
            # Bounding box statistics
            if len(boxes) > 0:
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                areas = widths * heights
                
                features.append(np.mean(areas))
                features.append(np.std(areas))
                features.append(np.mean(widths / heights))  # Aspect ratio
                
                # Total area coverage
                total_area = sum(areas)
                features.append(total_area)
            else:
                features.extend([0, 0, 1.0, 0, 0])  # No detections
                
            # Motion intensity (if available from previous frames)
            features.append(0)  # Placeholder for motion
        else:
            # No detections
            features.extend([0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0])
            
        # Speed features (optical flow approximation from frame differences)
        features.append(0)  # Placeholder for motion magnitude
        features.append(0)  # Placeholder for motion direction
        
        return np.array(features)
    
    def extract_motion_features(self, prev_frame, curr_frame):
        """
        Extract motion features between consecutive frames.
        
        Args:
            prev_frame: Previous frame (numpy array)
            curr_frame: Current frame (numpy array)
            
        Returns:
            numpy array of motion features
        """
        # Simple motion estimation using frame difference
        diff = np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))
        motion_magnitude = np.mean(diff)
        
        # Motion in different regions (simplified)
        h, w = diff.shape[:2]
        regions = [
            diff[:h//2, :w//2],  # Top-left
            diff[:h//2, w//2:],  # Top-right
            diff[h//2:, :w//2],  # Bottom-left
            diff[h//2:, w//2:],  # Bottom-right
        ]
        
        motion_features = [motion_magnitude]
        motion_features.extend([np.mean(r) for r in regions])
        
        return np.array(motion_features)
    
    def extract_sequence_features(self, frame_results, yolo_model):
        """
        Extract features from a sequence of frames.
        
        Args:
            frame_results: List of YOLO prediction results
            yolo_model: YOLO model instance
            
        Returns:
            numpy array of sequence features
        """
        frame_features = []
        
        for i, frame_result in enumerate(frame_results):
            features = self.extract_features_from_frame(frame_result, yolo_model)
            frame_features.append(features)
            
        # Add motion features between consecutive frames
        for i in range(1, len(frame_features)):
            motion_feat = self.extract_motion_features(
                frame_results[i-1].orig_img,
                frame_results[i].orig_img
            )
            frame_features[i] = np.concatenate([frame_features[i], motion_feat])
            
        return np.array(frame_features)
    
    def prepare_training_data(self, video_features, labels):
        """
        Prepare data for HMM training.
        
        Args:
            video_features: List of feature arrays for each video
            labels: List of labels (1 for violent, 0 for peaceful)
            
        Returns:
            X: Concatenated features
            lengths: Lengths of each sequence
        """
        # Filter by class
        violent_features = [f for f, l in zip(video_features, labels) if l == 1]
        peaceful_features = [f for f, l in zip(video_features, labels) if l == 0]
        
        return violent_features, peaceful_features
    
    def train(self, violent_sequences, peaceful_sequences):
        """
        Train HMM models for both classes.
        
        Args:
            violent_sequences: List of feature arrays for violent videos
            peaceful_sequences: List of feature arrays for peaceful videos
        """
        # Combine all sequences for scaling
        all_features = np.vstack(violent_sequences + peaceful_sequences)
        self.scaler.fit(all_features)
        
        # Scale sequences
        violent_scaled = [self.scaler.transform(seq) for seq in violent_sequences]
        peaceful_scaled = [self.scaler.transform(seq) for seq in peaceful_sequences]
        
        # Train violent HMM
        print("Training HMM for violent class...")
        self.violent_model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )
        
        # Combine violent sequences
        X_violent = np.vstack(violent_scaled)
        lengths_violent = [len(seq) for seq in violent_scaled]
        self.violent_model.fit(X_violent, lengths_violent)
        
        # Train peaceful HMM
        print("Training HMM for peaceful class...")
        self.peaceful_model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )
        
        # Combine peaceful sequences
        X_peaceful = np.vstack(peaceful_scaled)
        lengths_peaceful = [len(seq) for seq in peaceful_scaled]
        self.peaceful_model.fit(X_peaceful, lengths_peaceful)
        
        self.is_fitted = True
        print("HMM training complete!")
    
    def predict_sequence(self, frame_results, yolo_model):
        """
        Predict violence for a sequence of frames.
        
        Args:
            frame_results: List of YOLO prediction results
            yolo_model: YOLO model instance
            
        Returns:
            prediction: 1 for violent, 0 for peaceful
            probabilities: Probability scores for each class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted! Call train() first.")
        
        # Extract features
        features = self.extract_sequence_features(frame_results, yolo_model)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Get log-likelihood scores
        log_violent = self.violent_model.score(scaled_features)
        log_peaceful = self.peaceful_model.score(scaled_features)
        
        # Convert to probabilities (softmax)
        exp_violent = np.exp(log_violent - max(log_violent, log_peaceful))
        exp_peaceful = np.exp(log_peaceful - max(log_violent, log_peaceful))
        
        total = exp_violent + exp_peaceful
        prob_violent = exp_violent / total
        prob_peaceful = exp_peaceful / total
        
        prediction = 1 if prob_violent > prob_peaceful else 0
        
        return prediction, {'violent': prob_violent, 'peaceful': prob_peaceful}
    
    def predict_frame_probabilities(self, frame_results, yolo_model):
        """
        Get probability for each frame in sequence.
        
        Args:
            frame_results: List of YOLO prediction results
            yolo_model: YOLO model instance
            
        Returns:
            List of probabilities for each frame
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted! Call train() first.")
        
        frame_probs = []
        
        for i, frame_result in enumerate(frame_results[:i+1]):
            features = self.extract_features_from_frame(frame_result, yolo_model)
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            
            # Get emission probability for this frame
            log_violent = self.violent_model.score(scaled_features)
            log_peaceful = self.peaceful_model.score(scaled_features)
            
            exp_violent = np.exp(log_violent)
            exp_peaceful = np.exp(log_peaceful)
            total = exp_violent + exp_peaceful
            
            prob_violent = exp_violent / total if total > 0 else 0.5
            frame_probs.append({'violent': prob_violent, 'peaceful': 1 - prob_violent})
            
        return frame_probs
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        model_data = {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'violent_model': self.violent_model,
            'peaceful_model': self.peaceful_model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_components = model_data['n_components']
        self.covariance_type = model_data['covariance_type']
        self.n_iter = model_data['n_iter']
        self.violent_model = model_data['violent_model']
        self.peaceful_model = model_data['peaceful_model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")


class ViolenceDetectionPipeline:
    """
    Complete pipeline combining YOLO detection with HMM temporal modeling.
    """
    
    def __init__(self, yolo_model_path='runs/detect/violence_detection/weights/best.pt'):
        """
        Initialize the violence detection pipeline.
        
        Args:
            yolo_model_path: Path to trained YOLO model
        """
        from ultralytics import YOLO
        
        self.yolo_model = YOLO(yolo_model_path)
        self.hmm_detector = HMMBasedViolenceDetector()
        
    def process_video(self, video_path, frame_skip=5):
        """
        Process a video file for violence detection.
        
        Args:
            video_path: Path to video file
            frame_skip: Number of frames to skip between detections
            
        Returns:
            prediction: Violence detection result
            probabilities: Class probabilities
            frame_results: Detection results for each processed frame
        """
        # Run YOLO detection on video
        results = self.yolo_model.predict(
            source=video_path,
            stream=True,
            conf=0.5,
            verbose=False
        )
        
        # Collect frame results
        frame_results = []
        frame_count = 0
        
        for result in results:
            if frame_count % frame_skip == 0:
                frame_results.append(result)
            frame_count += 1
        
        # If we have enough frames, use HMM
        if len(frame_results) >= 3 and self.hmm_detector.is_fitted:
            prediction, probabilities = self.hmm_detector.predict_sequence(
                frame_results, self.yolo_model
            )
        else:
            # Fall back to majority voting from frame predictions
            predictions = []
            for frame_result in frame_results:
                if hasattr(frame_result, 'boxes') and len(frame_result.boxes) > 0:
                    classes = frame_result.boxes.cls.cpu().numpy()
                    most_common = Counter(classes).most_common(1)[0][0]
                    predictions.append(most_common)
                else:
                    predictions.append(1)  # Default to peaceful
            
            # Majority vote
            if predictions:
                most_common = Counter(predictions).most_common(1)[0][0]
                prediction = most_common
                probabilities = {'violent': prediction, 'peaceful': 1 - prediction}
            else:
                prediction = 1
                probabilities = {'violent': 0, 'peaceful': 1}
        
        return prediction, probabilities, frame_results
    
    def train_hmm(self, training_data_dir):
        """
        Train the HMM component using labeled video data.
        
        Args:
            training_data_dir: Directory containing training videos
        """
        violent_dir = Path(training_data_dir) / 'violent'
        peaceful_dir = Path(training_data_dir) / 'peaceful'
        
        violent_sequences = []
        peaceful_sequences = []
        
        # Process violent videos
        if violent_dir.exists():
            for video_path in violent_dir.glob('*.[ava][vmi][p]'):
                print(f"Processing violent video: {video_path}")
                _, probs, frames = self.process_video(str(video_path))
                if len(frames) >= 3:
                    seq_features = self.hmm_detector.extract_sequence_features(
                        frames, self.yolo_model
                    )
                    violent_sequences.append(seq_features)
        
        # Process peaceful videos
        if peaceful_dir.exists():
            for video_path in peaceful_dir.glob('*.[ava][vmi][p]'):
                print(f"Processing peaceful video: {video_path}")
                _, probs, frames = self.process_video(str(video_path))
                if len(frames) >= 3:
                    seq_features = self.hmm_detector.extract_sequence_features(
                        frames, self.yolo_model
                    )
                    peaceful_sequences.append(seq_features)
        
        # Train HMM
        if violent_sequences and peaceful_sequences:
            self.hmm_detector.train(violent_sequences, peaceful_sequences)
            
            # Save model
            self.hmm_detector.save_model('hmm_violence_model.pkl')
        else:
            print("Insufficient training data!")
    
    def load_hmm_model(self, model_path='hmm_violence_model.pkl'):
        """Load a pre-trained HMM model."""
        if os.path.exists(model_path):
            self.hmm_detector.load_model(model_path)
        else:
            print(f"Model file {model_path} not found!")


def create_hmm_from_yolo_predictions(yolo_model, video_path, max_frames=30):
    """
    Create HMM training data from YOLO predictions on videos.
    
    Args:
        yolo_model: Trained YOLO model
        video_path: Path to video file
        max_frames: Maximum frames to process per video
        
    Returns:
        Feature sequence for HMM training
    """
    detector = HMMBasedViolenceDetector()
    
    # Run YOLO on video
    results = yolo_model.predict(
        source=video_path,
        stream=True,
        conf=0.5,
        verbose=False
    )
    
    # Collect frames
    frame_results = []
    for i, result in enumerate(results):
        if i >= max_frames:
            break
        frame_results.append(result)
    
    # Extract features
    features = detector.extract_sequence_features(frame_results, yolo_model)
    
    return features


if __name__ == '__main__':
    # Example usage
    from ultralytics import YOLO
    
    # Check if YOLO model exists
    model_path = 'runs/detect/violence_detection/weights/best.pt'
    if not os.path.exists(model_path):
        print("Please train YOLO model first!")
        print(f"Expected model at: {model_path}")
    else:
        # Initialize pipeline
        pipeline = ViolenceDetectionPipeline(model_path)
        
        # Check if HMM model exists
        if os.path.exists('hmm_violence_model.pkl'):
            pipeline.load_hmm_model()
            print("HMM model loaded!")
        else:
            print("HMM model not found. Train with training data directory.")

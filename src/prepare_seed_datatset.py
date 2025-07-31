import os
import numpy as np
import scipy.io as sio
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ss_emerge.utils.data_helpers import calculate_de

RAW_DATA_ROOT = './data/SEED/ExtractedFeatures/'
OUTPUT_DATA_DIR = './data/SEED/prepared_data'

NUM_SUBJECTS_TOTAL = 15 
NUM_TRIALS_PER_SUBJECT = 15 
NUM_SESSIONS_PER_SUBJECT = 3 
NUM_CHANNELS = 62
NUM_BANDS = 5
DE_TIME_WINDOWS = 265
RAW_EEG_SFREQ = 200

NUM_CLASSES = 3

BANDS_FOR_DE_CALC = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 31), 'gamma': (31, 50)
}

def normalize_data(data):
    """Normalizes data per channel/feature using mean and std."""
    original_shape = data.shape
    data_flat = data.reshape(original_shape[0], -1)
    
    mean = np.mean(data_flat, axis=0, keepdims=True)
    std = np.std(data_flat, axis=0, keepdims=True)
    data_normalized_flat = (data_flat - mean) / (std + 1e-7)
    
    return data_normalized_flat.reshape(original_shape)

def process_seed_mat_files(raw_data_path):
    """
    Loads DE features from raw SEED .mat files, organizing them by video clip.
    """
    print(f"Loading raw .mat files from: {raw_data_path}")
    
    labels_raw = sio.loadmat(os.path.join(raw_data_path, 'label.mat'))['label'].squeeze()
    labels_adjusted = (labels_raw + 1).tolist()

    video_data_collection = {i: {} for i in range(NUM_TRIALS_PER_SUBJECT)}
    
    mat_files = sorted(glob.glob(os.path.join(raw_data_path, '*.mat')))
    mat_files = [f for f in mat_files if 'label.mat' not in f and 'readme.txt' not in f]

    subject_sessions = {}
    for f_path in mat_files:
        filename = os.path.basename(f_path)
        subject_id_raw = filename.split('_')[0]
        subject_sessions.setdefault(subject_id_raw, []).append(f_path)
    
    sorted_subject_ids = sorted(list(subject_sessions.keys()))

    # Determine actual number of subjects found in the provided files
    actual_num_subjects_found = len(sorted_subject_ids)
    print(f"Found {actual_num_subjects_found} unique subjects in provided .mat files.")


    for sub_id_raw in tqdm(sorted_subject_ids, desc="Processing Subjects"):
        session_files = sorted(subject_sessions[sub_id_raw])
        
        for session_file in session_files:
            data_mat = sio.loadmat(session_file, verify_compressed_data_integrity=False)
            
            for video_idx_1_based in range(1, NUM_TRIALS_PER_SUBJECT + 1):
                de_key = f'de_movingAve{video_idx_1_based}'
                
                processed_trial_data = None

                if de_key in data_mat:
                    temp_data = data_mat[de_key]
                    
                    if temp_data.ndim == 3 and \
                       temp_data.shape[0] == NUM_CHANNELS and \
                       temp_data.shape[2] == NUM_BANDS:
                        
                        processed_trial_data = temp_data.transpose(2, 0, 1) # (bands, channels, time_windows)
                    else:
                        print(f"Warning: Unexpected DE shape {temp_data.shape} for {de_key} in {session_file}. Skipping this trial.")
                        continue 

                    current_time_windows = processed_trial_data.shape[2]
                    if current_time_windows < DE_TIME_WINDOWS:
                        padded_data = np.zeros((NUM_BANDS, NUM_CHANNELS, DE_TIME_WINDOWS), dtype=processed_trial_data.dtype)
                        padded_data[:, :, :current_time_windows] = processed_trial_data
                        processed_trial_data = padded_data
                    elif current_time_windows > DE_TIME_WINDOWS:
                        processed_trial_data = processed_trial_data[:, :, :DE_TIME_WINDOWS]
                    
                if processed_trial_data is None:
                    print(f"Warning: No valid 'de_movingAve' data found for video {video_idx_1_based} in {session_file}. Skipping this trial.")
                    continue
                
                video_data_collection[video_idx_1_based - 1].setdefault(sub_id_raw, []).append(processed_trial_data)
    
    X_video_clips = [] # Will store (num_samples_found_for_video, F, C, T_de) for each video
    Y_video_labels = [] # Will store scalar label for each video

    X_flat_segments = [] # Will store (F, C, T_de) for individual segments
    Y_flat_segment_labels = [] # Will store scalar label for each individual segment

    for video_idx in tqdm(range(NUM_TRIALS_PER_SUBJECT), desc="Structuring Video Data"):
        video_specific_data_list = [] # Collect all (F, C, T_de) for this video across subjects/sessions
        
        for sub_id_str in sorted_subject_ids:
            if sub_id_str in video_data_collection[video_idx]:
                for session_data in video_data_collection[video_idx][sub_id_str]:
                    video_specific_data_list.append(session_data)
        
        # If no data for this video from any available subject/session, skip this video clip
        if not video_specific_data_list:
            print(f"Info: Video {video_idx} has no data from any available subject/session. Skipping this video.")
            continue 
        
        video_specific_data_np = np.array(video_specific_data_list) # (num_samples_found, F, C, T_de)
        
        # Apply normalization to this video clip's data (across its samples)
        video_specific_data_np = normalize_data(video_specific_data_np)

        # Append to video-level data
        X_video_clips.append(video_specific_data_np)
        Y_video_labels.append(labels_adjusted[video_idx]) # Label for this video clip

        # Append to flattened segment-level data
        for segment_data in video_specific_data_np:
            X_flat_segments.append(segment_data)
            Y_flat_segment_labels.append(labels_adjusted[video_idx]) # Each segment gets the video's label
    
    X_video_clips = np.array(X_video_clips, dtype=object) # Use dtype=object to handle potential ragged arrays
    Y_video_labels = np.array(Y_video_labels)

    X_flat_segments = np.array(X_flat_segments) # (total_segments, F, C, T_de)
    Y_flat_segment_labels = np.array(Y_flat_segment_labels) # (total_segments,)

    return X_video_clips, Y_video_labels, X_flat_segments, Y_flat_segment_labels

def main():
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    X_video_clips, Y_video_labels, X_flat_segments, Y_flat_segment_labels = process_seed_mat_files(RAW_DATA_ROOT)

    # Only proceed if any data was processed
    if X_video_clips.size == 0:
        print("No video clips were successfully processed. Skipping train/test split and saving.")
        return 

    X_train_videos, X_test_videos, Y_train_videos, Y_test_videos = train_test_split(
        X_video_clips, Y_video_labels, test_size=0.2, random_state=42, stratify=Y_video_labels
    )

    # Save video-level data for pretraining
    np.save(os.path.join(OUTPUT_DATA_DIR, 'x_train_SEED_videos.npy'), X_train_videos)
    np.save(os.path.join(OUTPUT_DATA_DIR, 'y_train_SEED_videos.npy'), Y_train_videos)
    np.save(os.path.join(OUTPUT_DATA_DIR, 'x_test_SEED_videos.npy'), X_test_videos)
    np.save(os.path.join(OUTPUT_DATA_DIR, 'y_test_SEED_videos.npy'), Y_test_videos)
    print(f"\nProcessed SEED video-level data saved to {OUTPUT_DATA_DIR}")
    print(f"Train video clips shape: {X_train_videos.shape}, Labels: {Y_train_videos.shape}")
    print(f"Test video clips shape: {X_test_videos.shape}, Labels: {Y_test_videos.shape}")

    # --- Split for Finetuning/Evaluation (Segment-level) ---
    if X_flat_segments.size == 0:
        print("No flat segments available after processing. Skipping train/test split and saving for segments.")
        return

    X_train_segments, X_test_segments, Y_train_segments, Y_test_segments = train_test_split(
        X_flat_segments, Y_flat_segment_labels, test_size=0.2, random_state=42, stratify=Y_flat_segment_labels
    )

    # Save segment-level data for finetuning/evaluation
    np.save(os.path.join(OUTPUT_DATA_DIR, 'x_train_SEED_segments.npy'), X_train_segments)
    np.save(os.path.join(OUTPUT_DATA_DIR, 'y_train_SEED_segments.npy'), Y_train_segments)
    np.save(os.path.join(OUTPUT_DATA_DIR, 'x_test_SEED_segments.npy'), X_test_segments)
    np.save(os.path.join(OUTPUT_DATA_DIR, 'y_test_SEED_segments.npy'), Y_test_segments)
    print(f"\nProcessed SEED segment-level data saved to {OUTPUT_DATA_DIR}")
    print(f"Train segments shape: {X_train_segments.shape}, Labels: {Y_train_segments.shape}")
    print(f"Test segments shape: {X_test_segments.shape}, Labels: {Y_test_segments.shape}")


if __name__ == "__main__":
    main()
import librosa
import numpy as np

def split_into_intervals(audio, sr=16000, threshold_length_segment_max=45, threshold_length_segment_min=0.256):
    intervals = []
    for top_db in range(30, 5, -5):
        intervals = librosa.effects.split(
            audio, top_db=top_db, frame_length=4096, hop_length=1024)
        if len(intervals) != 0 and max((intervals[:, 1] - intervals[:, 0]) / sr) <= threshold_length_segment_max:
            break
    return np.array([i for i in intervals if threshold_length_segment_min < (i[1] - i[0]) / sr <= threshold_length_segment_max])
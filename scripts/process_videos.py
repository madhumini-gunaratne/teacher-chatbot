import os
from pathlib import Path
from moviepy.editor import VideoFileClip
import whisper
import json
from tqdm import tqdm
from datetime import timedelta

# Configuration
VIDEOS_DIR = Path("data/media/videos")
TRANSCRIPTS_DIR = Path("data/media/transcripts")
AUDIO_MODEL = "base"

def extract_audio(video_path):
    """Extract audio from video file and analyze voice characteristics"""
    print(f"Extracting audio from {video_path}")
    video = VideoFileClip(str(video_path))
    audio = video.audio
    
    # Save a sample for voice analysis
    temp_audio = "temp_voice_analysis.wav"
    audio.write_audiofile(temp_audio)
    
    # Here we'll analyze voice characteristics like:
    # - Speech rate (words per minute)
    # - Pitch patterns
    # - Pauses and emphasis
    # - Voice tone and energy
    voice_characteristics = analyze_voice_patterns(temp_audio)
    
    # Clean up
    os.remove(temp_audio)
    
    return audio, voice_characteristics

def analyze_voice_patterns(audio_file):
    """Analyze voice characteristics from audio file"""
    from pydub import AudioSegment
    import numpy as np
    from scipy import signal
    
    # Load audio
    audio = AudioSegment.from_wav(audio_file)
    
    # Convert to numpy array for analysis
    samples = np.array(audio.get_array_of_samples())
    
    # Calculate voice features
    # 1. Amplitude and energy
    amplitude = np.abs(samples)
    energy = amplitude ** 2
    
    # 2. Spectral features using spectrogram
    frequencies, times, spectrogram = signal.spectrogram(samples, fs=audio.frame_rate)
    spectral_centroid = np.sum(frequencies[:, np.newaxis] * spectrogram, axis=0) / np.sum(spectrogram, axis=0)
    
    # 3. Pitch variation using autocorrelation
    frame_length = int(audio.frame_rate * 0.025)  # 25ms frames
    hop_length = frame_length // 2
    n_frames = (len(samples) - frame_length) // hop_length
    pitches = []
    
    for i in range(n_frames):
        frame = samples[i*hop_length:i*hop_length + frame_length]
        correlation = signal.correlate(frame, frame, mode='full')
        correlation = correlation[len(correlation)//2:]
        peaks = signal.find_peaks(correlation)[0]
        if len(peaks) > 0:
            fundamental_freq = audio.frame_rate / peaks[0] if peaks[0] != 0 else 0
            pitches.append(fundamental_freq)
    
    # Comprehensive analysis results
    characteristics = {
        "amplitude": {
            "average": float(np.mean(amplitude)),
            "variation": float(np.std(amplitude)),
            "max": float(np.max(amplitude)),
            "dynamic_range": float(np.percentile(amplitude, 95) - np.percentile(amplitude, 5))
        },
        "energy": {
            "average": float(np.mean(energy)),
            "variation": float(np.std(energy))
        },
        "spectral": {
            "centroid_mean": float(np.mean(spectral_centroid)),
            "centroid_std": float(np.std(spectral_centroid))
        },
        "pitch": {
            "average": float(np.mean(pitches)) if pitches else 0,
            "variation": float(np.std(pitches)) if pitches else 0,
            "range": float(np.percentile(pitches, 95) - np.percentile(pitches, 5)) if pitches else 0
        },
        "temporal": {
            "pace": calculate_speech_pace(samples, audio.frame_rate),
            "pauses": detect_pauses(samples, audio.frame_rate)
        }
    }
    
    return characteristics

def calculate_speech_pace(samples, frame_rate):
    """Estimate speech pace based on energy variations and patterns"""
    # Use multiple window sizes for better accuracy
    window_sizes = [
        int(frame_rate * 0.05),  # 50ms for short sounds
        int(frame_rate * 0.1),   # 100ms for typical syllables
        int(frame_rate * 0.2)    # 200ms for longer sounds
    ]
    
    energies = []
    for window_size in window_sizes:
        energy = np.array([np.sum(np.abs(samples[i:i+window_size])) 
                         for i in range(0, len(samples), window_size)])
        energies.append(energy)
    
    # Combine peaks from different window sizes
    all_peaks = []
    for energy in energies:
        peaks = detect_peaks(energy)
        all_peaks.extend(peaks)
    
    # Remove duplicates and sort
    unique_peaks = sorted(set(all_peaks))
    
    # Calculate speech rate
    duration_minutes = len(samples) / frame_rate / 60
    estimated_words = len(unique_peaks) / len(window_sizes)  # Average across window sizes
    return estimated_words / duration_minutes

def detect_peaks(energy, threshold=0.5):
    """Detect peaks in energy that likely correspond to words"""
    normalized = energy / np.max(energy)
    return np.where(normalized > threshold)[0]

def detect_pauses(samples, frame_rate):
    """Detect and analyze significant pauses in speech"""
    from scipy import signal
    
    # Use multiple window sizes to detect different types of pauses
    window_sizes = [
        int(frame_rate * 0.2),  # 200ms for short pauses
        int(frame_rate * 0.5),  # 500ms for medium pauses
        int(frame_rate * 1.0)   # 1s for long pauses
    ]
    
    pause_analysis = {}
    
    for window_size in window_sizes:
        # Calculate energy in windows
        energy = np.array([np.sum(np.abs(samples[i:i+window_size])) 
                          for i in range(0, len(samples), window_size)])
        
        # Normalize energy
        normalized = energy / np.max(energy)
        
        # Find pause segments
        pause_mask = normalized < 0.2
        pause_segments = signal.find_peaks(pause_mask.astype(int))[0]
        
        # Calculate pause durations
        pause_durations = []
        current_pause = 0
        
        for i in range(len(pause_mask)):
            if pause_mask[i]:
                current_pause += 1
            elif current_pause > 0:
                pause_durations.append(current_pause * (window_size / frame_rate))
                current_pause = 0
        
        # Add final pause if exists
        if current_pause > 0:
            pause_durations.append(current_pause * (window_size / frame_rate))
        
        # Analyze pause patterns
        pause_analysis[f"window_{int(window_size/frame_rate*1000)}ms"] = {
            "count": len(pause_durations),
            "frequency": len(pause_durations) / (len(samples) / frame_rate),
            "avg_duration": float(np.mean(pause_durations)) if pause_durations else 0,
            "max_duration": float(np.max(pause_durations)) if pause_durations else 0,
            "total_pause_time": float(sum(pause_durations)) if pause_durations else 0
        }
    
    return pause_analysis

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))

def analyze_teaching_style(text):
    """Analyze teaching style and personality traits from transcript"""
    # Keywords and phrases that indicate different teaching styles
    style_indicators = {
        "interactive": ["let's try", "what do you think", "can anyone", "who can", "?", "let's", "can you", "try this"],
        "encouraging": ["great question", "good point", "excellent", "well done", "that's right", "good job", "interesting", "amazing", "fascinating"],
        "methodical": ["first", "second", "next", "then", "finally", "step", "therefore", "because", "means", "this is"],
        "analogies": ["like", "similar to", "imagine", "think of it as", "it's as if", "for example", "just as", "compare"],
        "humor": ["laugh", "joke", "funny", "😊", "😄", "haha", "here's a fun", "let me tell you"],
        "empathetic": ["understand", "might be confused", "don't worry", "take your time", "it's okay", "common mistake", "many students"]
    }
    
    # Analyze frequency of style indicators
    style_scores = {}
    text_lower = text.lower()
    
    for style, indicators in style_indicators.items():
        count = sum(text_lower.count(indicator.lower()) for indicator in indicators)
        style_scores[style] = count
    
    # Normalize scores
    total = sum(style_scores.values()) or 1
    style_scores = {k: v/total for k, v in style_scores.items()}
    
    return style_scores

def transcribe_audio(audio_clip, output_path):
    """Transcribe audio using Whisper and analyze teaching style"""
    print("2. Loading Whisper model...")
    # Load Whisper model
    model = whisper.load_model(AUDIO_MODEL)
    
    print("3. Preparing audio for transcription...")
    # Save audio temporarily
    temp_audio = "temp_audio.mp3"
    audio_clip.write_audiofile(temp_audio)
    
    # Transcribe
    print(f"Transcribing audio...")
    result = model.transcribe(temp_audio)
    
    # Analyze teaching style
    teaching_style = analyze_teaching_style(result["text"])
    
    # Format transcript with timestamps and style analysis
    transcript = {
        "text": result["text"],
        "segments": [{
            "start": format_timestamp(seg["start"]),
            "end": format_timestamp(seg["end"]),
            "text": seg["text"]
        } for seg in result["segments"]],
        "teaching_style": teaching_style
    }
    
    # Save transcript
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    # Clean up
    os.remove(temp_audio)
    return transcript

def process_videos():
    """Process all video files in the videos directory"""
    if not VIDEOS_DIR.exists():
        print(f"Videos directory not found: {VIDEOS_DIR}")
        return
    
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        print("No .mp4 files found in videos directory")
        return
        
    print(f"\nFound {len(video_files)} videos to process:")
    for video in video_files:
        print(f"- {video.name}")
    
    for video_path in video_files:
        print(f"\nProcessing video: {video_path.name}")
        # Generate output path for transcript
        transcript_path = TRANSCRIPTS_DIR / f"{video_path.stem}_transcript.json"
        
        # Skip if already processed
        if transcript_path.exists():
            print(f"Skipping {video_path.name} - transcript already exists")
            continue
            
        print("1. Extracting audio...")
        
        try:
            # Extract audio
            audio = extract_audio(video_path)
            
            # Transcribe
            transcript = transcribe_audio(audio, transcript_path)
            print(f"Successfully processed {video_path.name}")
            
        except Exception as e:
            print(f"Error processing {video_path.name}: {str(e)}")
            continue
        
        finally:
            # Clean up
            if 'audio' in locals():
                audio.close()

if __name__ == "__main__":
    process_videos()
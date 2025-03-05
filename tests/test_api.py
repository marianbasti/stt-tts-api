import os
import io
import json
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def create_test_audio(duration=3, sample_rate=22050):
    """Create a test sine wave audio file"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio, sample_rate

def test_main_page():
    """Test the main HTML page endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_transcription_endpoint():
    """Test the transcription endpoint"""
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Save to BytesIO object
    audio_io = io.BytesIO()
    sf.write(audio_io, audio_data, sample_rate, format='WAV')
    audio_io.seek(0)
    
    # Create test request
    files = {
        'file': ('test.wav', audio_io, 'audio/wav')
    }
    data = {
        'model': 'openai/whisper-large-v3-turbo',
        'response_format': 'json'
    }
    
    response = client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200
    
    # Check response format
    result = response.json()
    assert 'text' in result

def test_tts_endpoint():
    """Test the text-to-speech endpoint"""
    # Create test audio for speaker reference
    audio_data, sample_rate = create_test_audio()
    
    # Save to BytesIO object
    audio_io = io.BytesIO()
    sf.write(audio_io, audio_data, sample_rate, format='WAV')
    audio_io.seek(0)
    
    # Create test request
    files = {
        'speaker_wav': ('speaker.wav', audio_io, 'audio/wav')
    }
    data = {
        'text': 'Hello, this is a test.'
    }
    
    response = client.post("/v1/audio/tts", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

def test_audio_to_blendshapes():
    """Test the audio to blendshapes endpoint"""
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Save to BytesIO object
    audio_io = io.BytesIO()
    sf.write(audio_io, audio_data, sample_rate, format='WAV')
    audio_io.seek(0)
    
    # Create test request
    files = {
        'audio': ('test.wav', audio_io, 'audio/wav')
    }
    
    response = client.post("/v1/audio_to_blendshapes", files=files)
    assert response.status_code == 200
    
    # Check response format
    result = response.json()
    assert 'status' in result
    assert 'frames' in result
    assert 'blendshapes' in result
    assert result['status'] == 'success'
    assert isinstance(result['frames'], int)
    assert isinstance(result['blendshapes'], list)
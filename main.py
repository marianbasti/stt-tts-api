import os
import sys
import io
import shutil
import time
import numpy as np
from datetime import timedelta
from functools import lru_cache
from typing import Any, List, Union, Optional

import torch
import soundfile as sf
from fastapi import FastAPI, Form, UploadFile, File
from fastapi import HTTPException, status, Response
from fastapi.responses import Response, JSONResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

# Add models directory to Python path
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
NEUROSYNC_PATH = os.path.join(MODELS_PATH, 'NEUROSYNC_Audio_To_Face_Blendshape')
if MODELS_PATH not in sys.path:
    sys.path.append(MODELS_PATH)
if NEUROSYNC_PATH not in sys.path:
    sys.path.append(NEUROSYNC_PATH)

from NEUROSYNC_Audio_To_Face_Blendshape.utils.model.model import load_model as load_neurosync_model
from NEUROSYNC_Audio_To_Face_Blendshape.utils.generate_face_shapes import generate_facial_data_from_bytes
from NEUROSYNC_Audio_To_Face_Blendshape.utils.config import config as neurosync_config

# Set tts model path
TTS_MODEL = os.getenv('TTS_MODEL', "./models/XTTS-v2-argentinian-spanish")
WHISPER_MODEL = os.getenv('WHISPER_MODEL', "openai/whisper-large-v3-turbo")

UPLOAD_DIR="/tmp"

app = FastAPI()
"""
curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F model="whisper-1" \
  -F file="@/path/to/file.mp3"


response=$(curl -X POST -F "text="Hola, como estas?"" -F "speaker_wav=@"path/to/speaker/audio.wav"" "http://BASE_URL:PORT/v1/audio/tts")
"""



# Whisper transcription functions
# ----------------
@lru_cache(maxsize=1)
def get_whisper_model_faster(whisper_model: str):
    """Get a whisper model"""
    model = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        torch_dtype=torch.float16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        model_kwargs={"use_flash_attention_2": False},  # Disable Flash Attention 2
    )
    return model

def get_whisper_model(whisper_model: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(whisper_model)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

def get_tts_model(model_dir: str):
    """Get a TTS model and ensure tokenizer is initialized"""
    config = XttsConfig()
    config.load_json(f"{model_dir}/config.json")
    
    # Set tokenizer file path
    tokenizer_file = os.path.join(model_dir, "vocab.json")
    if not os.path.exists(tokenizer_file):
        raise RuntimeError(f"Tokenizer file not found at {tokenizer_file}")
    config.model_args['tokenizer_file'] = tokenizer_file
    
    # Create model instance
    model = Xtts.init_from_config(config)
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint not found at {checkpoint_path}")
    
    model.load_checkpoint(config=config, checkpoint_path=checkpoint_path)
    
    # Move model to device
    if torch.cuda.is_available():
        model.cuda()
    
    # Initialize tokenizer explicitly
    if model.tokenizer is None:
        raise RuntimeError("Tokenizer failed to initialize")
        
    return model, config

def transcribe_faster(audio_path: str, whisper_model: str, **whisper_args):
    """Transcribe the audio file using whisper"""
    start_time = time.time()
    transcriber = get_whisper_model_faster(whisper_model)

    transcript = transcriber(
        audio_path,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=False
    )
    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds")
    return transcript

def transcribe(audio_path: str, whisper_model: str, **whisper_args):
    """Transcribe the audio file using whisper"""
    start_time = time.time()
    
    transcriber = get_whisper_model(whisper_model)

    transcript = transcriber(
        audio_path
    )

    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds")
    return transcript

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.environ.get("NO_TTS", False):
    model, config = get_tts_model(TTS_MODEL)

# Add NeuroSync model loading
neurosync_model_path = 'models/NEUROSYNC_Audio_To_Face_Blendshape/model.pth'
try:
    blendshape_model = load_neurosync_model(neurosync_model_path, neurosync_config, device)
except RuntimeError as e:
    print(f"Warning: Failed to load NeuroSync model: {e}")
    blendshape_model = None

WHISPER_DEFAULT_SETTINGS = {
    "whisper_model": WHISPER_MODEL,
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "verbose": False,
    "task": "transcribe",
    "language": "es",
}

@app.post('/v1/audio/transcriptions')
async def transcriptions(model: str = Form(...),
                         file: UploadFile = File(...),
                         response_format: Optional[str] = Form(None),
                         prompt: Optional[str] = Form(None),
                         temperature: Optional[float] = Form(None),
                         language: Optional[str] = Form(None)):

    assert model == WHISPER_MODEL
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad file"
            )
    if response_format is None:
        response_format = 'json'
    if response_format not in ['json',
                           'text',
                           'srt',
                           'verbose_json',
                           'vtt']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad response_format"
            )
    if temperature is None:
        temperature = 0.0
    if temperature < 0.0 or temperature > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad temperature"
            )

    filename = file.filename
    fileobj = file.file
    upload_name = os.path.join(UPLOAD_DIR, filename)
    upload_file = open(upload_name, 'wb+')
    shutil.copyfileobj(fileobj, upload_file)
    upload_file.close()

    transcript = transcribe_faster(audio_path=upload_name, **WHISPER_DEFAULT_SETTINGS)


    if response_format in ['text']:
        return transcript['text']

    if response_format in ['srt']:
        ret = ""
        for seg in transcript['segments']:
            
            td_s = timedelta(milliseconds=seg["start"]*1000)
            td_e = timedelta(milliseconds=seg["end"]*1000)

            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'

            ret += '{}\n{} --> {}\n{}\n\n'.format(seg["id"], t_s, t_e, seg["text"])
        ret += '\n'
        return ret

    if response_format in ['vtt']:
        ret = "WEBVTT\n\n"
        for seg in transcript['segments']:
            td_s = timedelta(milliseconds=seg["start"]*1000)
            td_e = timedelta(milliseconds=seg["end"]*1000)

            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'

            ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"])
        return ret

    if response_format in ['verbose_json']:
        transcript.setdefault('task', WHISPER_DEFAULT_SETTINGS['task'])
        transcript.setdefault('duration', transcript['segments'][-1]['end'])
        return transcript

    return {'text': transcript['text']}

@app.post("/v1/audio/tts")
async def generate_audio(text: str = Form(...), speaker_wav: UploadFile = File(...)):
    if not model or not hasattr(model, 'tokenizer') or model.tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail="TTS model or tokenizer not properly initialized"
        )

    def tts():
        try:
            t0 = time.time()
            output = model.synthesize(
                text,
                config,
                speaker_wav.file,
                language="es"
            )

            inference_time = time.time() - t0
            print(f"Time to generate audio: {round(inference_time*1000)} milliseconds")
            
            with io.BytesIO() as wav_io:
                sf.write(wav_io, output['wav'], samplerate=22050, format='WAV')
                wav_io.seek(0)
                wav_bytes = wav_io.read()
            return wav_bytes
            
        except Exception as e:
            print(f"Error in TTS generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"TTS generation failed: {str(e)}"
            )

    return Response(tts(), media_type="audio/wav")

@app.post("/v1/audio_to_blendshapes")
async def audio_to_blendshapes(audio: UploadFile = File(...)):
    """Convert audio to facial blendshapes using NeuroSync model"""
    if blendshape_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NeuroSync model is not available"
        )
    
    # Validate file type
    if not audio.filename.endswith('.wav'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only WAV files are supported"
        )
    
    try:
        # Read audio bytes
        print("Reading audio file...")
        audio_bytes = await audio.read()
        
        # Generate facial data
        print("Generating facial blendshapes...")
        generated_facial_data = generate_facial_data_from_bytes(
            audio_bytes, 
            blendshape_model, 
            device, 
            neurosync_config
        )
        
        if generated_facial_data is None or len(generated_facial_data) == 0:
            raise ValueError("Failed to generate facial data")
            
        # Convert numpy array to list for JSON serialization
        if isinstance(generated_facial_data, np.ndarray):
            generated_facial_data_list = generated_facial_data.tolist()
        else:
            generated_facial_data_list = generated_facial_data
            
        print(f"Successfully generated {len(generated_facial_data_list)} frames of blendshape data")
        
        return JSONResponse({
            'status': 'success',
            'frames': len(generated_facial_data_list),
            'blendshapes': generated_facial_data_list
        })
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process audio: {str(e)}"
        )

# Serve test.html webpage
@app.get("/")
async def main():
    content = """<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XTTS & Whisper Inference API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .container {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px;
            width: 300px;
            text-align: center;
        }

        h1 {
            color: #333;
            text-align: center;
        }

        input[type="text"],
        input[type="file"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            overflow: hidden;
        }

        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #45a049;
        }

        #audio-player {
            width: 100%;
            margin-top: 15px;
        }

        #transcribed {
            display: block;
            margin-top: 10px;
            color: #555;
            font-style: italic;
            word-wrap: break-word;
        }

        #llm-response {
            margin-top: 10px;
            color: #555;
            font-style: italic;
        }

        .row {
            display: flex;
            flex-direction: row;
        }
    </style>
</head>

<body>
    <div class="row">
        <!-- Left Column -->
        <div class="container">
            <h1>XTTS Inference API</h1>
            <input type="text" id="text-input" placeholder="Enter text to synthesize">
            <input type="file" id="speaker-file" accept="audio/wav">
            <button id="say-button" onclick="measureTime(() => say(document.getElementById('text-input').value))">Say</button>
        </div>

        <div class="container">
            <h1>Whisper Inference API</h1>
            <input type="file" id="audio-transcribe" accept="audio/wav">
            <button id="transcribe-button" onclick="transcribe()">Transcribe</button>
            <a id="transcribed"></a>
        </div>

        <!-- Right Column for LLM interaction -->
        <div class="container">
            <h1>LLM Interaction</h1>
            <input type="text" id="openai-base-url" placeholder="OpenAI Base URL">
            <input type="text" id="openai-api-key" placeholder="OpenAI API Key">
            <input type="text" id="system-message" placeholder="System message to contextualize LLM">
            <input type="file" id="audio-file" accept="audio/wav">
            <button id="process-audio" onclick="measureTime(processAudio)">Process Audio</button>
        </div>
        <div class="container">
            <a id="transcribed-llm"></a>
            <p id="llm-response"></p>
            <a id="time"></a>
            <audio id="audio-player" controls></audio>
        </div>
    </div>

    <script>
        function measureTime(func, ...args) {
            const startTime = performance.now();
            func(...args).then(() => {
                const endTime = performance.now();
                document.getElementById("time").innerText = `Time taken: ${endTime - startTime} milliseconds`;
            });
        }
        // Post the text and speaker audio to the API and play the streaming response
        async function say(text) {
            const speakerFile = document.getElementById("speaker-file").files[0];
            const formData = new FormData();
            formData.append("text", text);
            formData.append("speaker_wav", speakerFile);

            try {
                const response = await fetch("/v1/audio/tts", {
                    method: "POST",
                    body: formData,
                });
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                document.getElementById("audio-player").src = audioUrl;
            } catch (error) {
                console.error("Error:", error);
            }
        }

        // Post the audio to the API and show transcribed text
        async function transcribe() {
            const audioFile = document.getElementById("audio-transcribe").files[0];
            const formData = new FormData();
            formData.append("model", "openai/whisper-large-v3-turbo");
            formData.append("file", audioFile);

            try {
                const response = await fetch("/v1/audio/transcriptions", {
                    method: "POST",
                    body: formData,
                });
                const transcribed = await response.text();
                document.getElementById("transcribed").innerText = transcribed;
            } catch (error) {
                console.error("Error:", error);
            }
        }

        // Handle the entire LLM interaction flow
        async function processAudio() {
            const audioFile = document.getElementById("audio-file").files[0];
            const baseUrl = document.getElementById("openai-base-url").value;
            const apiKey = document.getElementById("openai-api-key").value;
            const systemMessage = document.getElementById("system-message").value;
            const formData = new FormData();
            formData.append("model", "openai/whisper-large-v3-turbo");
            formData.append("file", audioFile);

            try {
                // Step 1: Transcribe audio
                const transcriptionResponse = await fetch("/v1/audio/transcriptions", {
                    method: "POST",
                    body: formData,
                });
                const transcription = await transcriptionResponse.text();
                document.getElementById("transcribed-llm").innerText = transcription;

                // Step 2: Send transcription to LLM
                const llmResponse = await fetch(`${baseUrl}/chat/completions`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${apiKey}`
                    },
                    body: JSON.stringify({
                        model: "gpt-3.5-turbo",
                        messages: [{role:"system", content:systemMessage},{ role: "user", content: transcription }]
                    })
                });

                const llmData = await llmResponse.json();
                const llmText = llmData.choices[0].message.content.replace('"text":"', '').replace('"', '');
                document.getElementById("llm-response").innerText = llmText;

                // Step 3: Send LLM response to XTTS for synthesis
                say(llmText)
            } catch (error) {
                console.error("Error:", error);
            }
        }
    </script>
</body>

</html>

    """
    return Response(content, media_type="text/html")

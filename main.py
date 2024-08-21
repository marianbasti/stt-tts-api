from fastapi import FastAPI, Form, UploadFile, File
from fastapi import HTTPException, status, Response
from fastapi.responses import Response
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import os, io, wave
import shutil
from functools import lru_cache
from typing import Any, List, Union, Optional

from datetime import timedelta
import time
import soundfile as sf

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

app = FastAPI()
"""
curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F model="whisper-1" \
  -F file="@/path/to/file.mp3"


response=$(curl -X POST -F "text="Hola, como estas?"" -F "speaker_wav=@"path/to/speaker/audio.wav"" "http://localhost:8000/v1/audio/tts")
"""



# Whisper transcription functions
# ----------------
@lru_cache(maxsize=1)
def get_whisper_model(whisper_model: str):
    """Get a whisper model"""
    model = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        torch_dtype=torch.float16,
        device="cuda:0", # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )
    return model

def get_tts_model(model_dir: str):
    """Get a TTS model"""
    config = XttsConfig()
    config.load_json(f"{model_dir}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    model.cuda()
    return model, config

def transcribe(audio_path: str, whisper_model: str, **whisper_args):
    """Transcribe the audio file using whisper"""
    import time
    start_time = time.time()
    transcriber = get_whisper_model(whisper_model)

    transcript = transcriber(
        audio_path,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )
    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds")
    return transcript

WHISPER_DEFAULT_SETTINGS = {
    "whisper_model": "marianbasti/distil-whisper-large-v3-es",
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

UPLOAD_DIR="/tmp"

TTS_MODEL="/home/marian/TTS/XTTS-v2-argentinian-spanish"
model, config = get_tts_model(TTS_MODEL)

@app.post('/v1/audio/transcriptions')
async def transcriptions(model: str = Form(...),
                         file: UploadFile = File(...),
                         response_format: Optional[str] = Form(None),
                         prompt: Optional[str] = Form(None),
                         temperature: Optional[float] = Form(None),
                         language: Optional[str] = Form(None)):

    assert model == "marianbasti/distil-whisper-large-v3-es"
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

    transcript = transcribe(audio_path=upload_name, **WHISPER_DEFAULT_SETTINGS)


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

    def tts():
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

    return Response(tts(), media_type="audio/wav")


# Serve test.html webpage
@app.get("/")
async def main():
    content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XTTS Inference API</title>
<style>
    /* Same styles as before */
</style>
</head>
<body>
<div class="container">
    <h1>XTTS Inference API</h1>
    <input type="text" id="text-input" placeholder="Enter text to synthesize">
    <input type="file" id="speaker-file" accept="audio/wav">
    <button id="say-button" onclick="say()">Say</button>
    <audio id="audio-player" controls></audio>
</div>

<script>
    // Post the text and speaker audio to the API and play the streaming response
    async function say() {
        const text = document.getElementById("text-input").value;
        const speakerFile = document.getElementById("speaker-file").files[0];
        const formData = new FormData();
        formData.append("text", text);
        formData.append("speaker_wav", speakerFile);
        const response = await fetch("/v1/audio/tts", {
            method: "POST",
            body: formData,
        });
        console.log(response)
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audioPlayer = document.getElementById("audio-player");
        audioPlayer.src = audioUrl;
    }
</script>
</body>
</html>
    """
    return Response(content, media_type="text/html")

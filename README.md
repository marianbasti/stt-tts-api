# stt-tts-api

FastAPI server to serve whisper transcription models, xtts speech synthesis and audio-to-blendshape face animation models, for an end-to-end voice conversational agent.

Some of code has been copied from [whisper-ui](https://github.com/hayabhay/whisper-ui) and [NeuroSync](https://github.com/AnimaVR/NeuroSync_Local_API)

## Setup
This was built & tested on Python 3.12.3, Ubutu 24.
We recommend using a python virtual environment.

```bash
git clone https://github.com/marianbasti/stt-tts-api
cd stt-tts-api
python3 -m venv .venv
source env/bin/activate
pip install -r requirements.txt
```

Download and place the XTTS-v2 model from [🤗HuggingFace](https://huggingface.co/marianbasti/XTTS-v2-argentinian-spanish) in `/models/`:
```bash
cd models
git clone https://huggingface.co/marianbasti/XTTS-v2-argentinian-spanish
cd ..
# Optionally, set TTS model path and whisper model HF ID as environment variables. These are the default values
export TTS_MODEL=./models/XTTS-v2-argentinian-spanish
export WHISPER_MODEL=marianbasti/distil-whisper-large-v3-es
```

Download and place the NeuroSync model from [🤗HuggingFace](https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape/resolve/main/model.pth?download=true) in `/models/`, and rename it as `neurosync.pth`:
```bash
cd models
wget https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape/resolve/main/model.pth
mv model.pth neurosync.pth
cd ..
```
## Usage

### Run server
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```


### Simple web interface
A minimalistic web GUI runs on the base URL. It enables
- Interacting with the TTS model.
- Interacting with the transcription model.
- Testing the full pipeline with an external OpenAI-compatible API LLM. Just set the backend URL, API key and an optional system prompt.

### Endpoint: `/v1/audio/transcriptions`

**Description:** This endpoint accepts an audio file and returns its transcription in the specified format.

**Parameters:**

- `model` (str, required): The model to be used for transcription.
- `file` (UploadFile, required): The audio file to be transcribed.
- `response_format` (str, optional): The format of the transcription response. Can be one of `['json', 'text', 'srt', 'verbose_json', 'vtt']`. Default is `json`.
- `prompt` (str, optional): An optional prompt to guide the transcription.
- `temperature` (float, optional): A value between 0.0 and 1.0 to control the randomness of the transcription. Default is `0.0`.
- `language` (str, optional): The language of the transcription.

**Responses:**

- **200 OK**: Returns the transcription in the specified format.
- **400 Bad Request**: Returns an error if the request parameters are invalid.

**Example Request:**

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
-F "model=marianbasti/distil-whisper-large-v3-es" \
-F "file=@path_to_your_audio_file.wav" \
-F "response_format=json"
```

**Response:**

```json
{
  "text": "Transcribed text here"
}
```

### Endpoint: `/v1/audio/tts`

**Description:** This endpoint accepts text and a speaker audio file, and returns a synthesized audio file.

**Parameters:**

- `text` (str, required): The text to be converted to speech.
- `speaker_wav` (UploadFile, required): An audio file of the speaker's voice.

**Responses:**

- **200 OK**: Returns the synthesized audio file in WAV format.
- **400 Bad Request**: Returns an error if the request parameters are invalid.

**Example Request:**

```bash
curl -X POST "http://localhost:8080/v1/audio/tts" \
-F "text=Hello, world!" \
-F "speaker_wav=@path_to_speaker_audio_file.wav"
```

**Response:**

The response will be a binary WAV file containing the synthesized speech.

### Endpoint: `/v1/audio_to_blendshapes`

**Description:** This endpoint accepts an audio file and returns blendshape coefficients for facial animation.

**Parameters:**

- `file` (UploadFile, required): The audio file to be analyzed.
- `model` (str, optional): The model to be used for generating blendshapes. Default is `default_blendshape_model`.

**Responses:**

- **200 OK**: Returns the blendshape coefficients in JSON format.
- **400 Bad Request**: Returns an error if the request parameters are invalid.

**Example Request:**

```bash
curl -X POST "http://localhost:8080/v1/audio/blendshapes" \
-F "file=@path_to_your_audio_file.wav" \
-F "model=default_blendshape_model"
```

## License

MIT

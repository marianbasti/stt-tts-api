# stt-tts-api

FastAPI server to serve whisper transcription models and xtts speech synthesis models.

Some of code has been copied from [whisper-ui](https://github.com/hayabhay/whisper-ui)

## Setup
This was built & tested on Python 3.12.3, Ubutu 24

```bash
pip install -r requirements.txt
```

## Usage

### Run server
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --tts_model ./path/to/tts/model
```


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

## License

MIT

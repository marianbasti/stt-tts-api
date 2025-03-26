from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import aiohttp
import os
import logging
import time
from openai import AsyncOpenAI
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

router = APIRouter()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "")

# Validate API key at startup
if not OPENAI_API_KEY:
    OPENAI_API_KEY='none'

logger.debug(f"Initializing router with BACKEND_URL: {BACKEND_URL}")
client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL if OPENAI_API_URL else None)

# Initialize with dummy values
client = AsyncOpenAI(api_key="none", base_url="http://localhost:8080")

async def update_openai_client(api_url: str):
    global client
    client = AsyncOpenAI(api_key="none", base_url=api_url)
    logger.info(f"OpenAI client updated with base URL: {api_url}")

@router.post("/api/config")
async def update_config(api_url: str = Form(...)):
    try:
        await update_openai_client(api_url)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache for the available model
_cached_model: Optional[str] = None

# Store chat histories by session
chat_histories: Dict[str, List[dict]] = {}

async def get_first_available_model() -> str:
    global _cached_model
    
    if _cached_model:
        logger.debug(f"Using cached model: {_cached_model}")
        return _cached_model
        
    try:
        logger.debug("Fetching available models from OpenAI")
        models = await client.models.list()
        if not models.data:
            logger.error("No models available from OpenAI")
            raise HTTPException(status_code=500, detail="No models available")
        _cached_model = models.data[0].id
        logger.info(f"Selected model: {_cached_model}")
        return _cached_model
    except Exception as e:
        logger.error(f"Failed to fetch models: {str(e)}")
        if "authentication" in str(e).lower():
            raise HTTPException(status_code=500, detail="Invalid or missing API key")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    start_time = time.time()
    logger.debug(f"Received transcription request for file: {file.filename}")
    try:
        form_data = aiohttp.FormData()
        form_data.add_field('file', file.file.read(), 
                           filename=file.filename,
                           content_type='audio/wav')
        form_data.add_field('model', 'openai/whisper-large-v3-turbo')
        
        logger.debug("Sending transcription request to backend")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{BACKEND_URL}/v1/audio/transcriptions", 
                                  data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed_time = time.time() - start_time
                    logger.info(f"Transcription successful in {elapsed_time:.2f}s: {result.get('text', '')[:50]}...")
                    result['elapsed_time'] = elapsed_time
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Transcription failed with status {response.status}: {error_text}")
                    raise HTTPException(status_code=response.status, 
                                      detail="Transcription failed")
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/models")
async def get_models():
    try:
        models = await client.models.list()
        return {"models": [model.id for model in models.data]}
    except Exception as e:
        logger.error(f"Failed to fetch models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat")
async def chat_completion(request: Request, message: str = Form(...), model: str = Form(None)):
    start_time = time.time()
    logger.debug(f"Received chat request with message: {message[:50]}... using model: {model}")
    try:
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Get or create session-specific chat history
        session_id = str(hash(request.client.host))
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        chat_histories[session_id].append({"role": "user", "content": message})
        
        # Use specified model or get first available
        selected_model = model if model else await get_first_available_model()
        logger.debug(f"Using model: {selected_model}")
        
        try:
            response = await client.chat.completions.create(
                model=selected_model,
                messages=chat_histories[session_id],
                max_tokens=150
            )
            reply = response.choices[0].message.content
            
            chat_histories[session_id].append({"role": "assistant", "content": reply})
            
            if len(chat_histories[session_id]) > 10:
                chat_histories[session_id] = chat_histories[session_id][-10:]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Chat completion successful in {elapsed_time:.2f}s. Response: {reply[:50]}...")
            
            return JSONResponse(content={
                "response": reply,
                "elapsed_time": elapsed_time
            })
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            if "authentication" in str(e).lower():
                raise HTTPException(status_code=500, detail="Invalid or missing API key")
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/tts")
async def text_to_speech(text: str = Form(...), speaker_wav: UploadFile = File(...)):
    start_time = time.time()
    logger.debug(f"Received TTS request - Text: {text[:50]}..., Speaker file: {speaker_wav.filename}")
    try:
        form_data = aiohttp.FormData()
        form_data.add_field('text', text)
        form_data.add_field('speaker_wav', speaker_wav.file.read(),
                           filename=speaker_wav.filename,
                           content_type='audio/wav')
        
        logger.debug("Sending TTS request to backend")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{BACKEND_URL}/v1/audio/tts",
                                  data=form_data) as response:
                if response.status == 200:
                    elapsed_time = time.time() - start_time
                    logger.info(f"TTS generation successful in {elapsed_time:.2f}s")
                    audio_content = await response.read()
                    headers = {'X-Processing-Time': f'{elapsed_time:.2f}'}
                    return Response(
                        content=audio_content,
                        media_type="audio/wav",
                        headers=headers
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"TTS failed with status {response.status}: {error_text}")
                    raise HTTPException(status_code=response.status,
                                      detail="Text-to-speech failed")
    except Exception as e:
        logger.error(f"Error in TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    try:
        models = await client.models.list()
        if models and models.data:
            logger.info(f"Health check successful, found {len(models.data)} models")
            return {"status": "healthy", "models_available": True, "models_count": len(models.data)}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
    return {"status": "unhealthy", "models_available": False, "models_count": 0}
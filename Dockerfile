FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    git-lfs \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Copy only necessary files (excluding models directory)
COPY requirements.txt start_server.sh main.py ./
COPY routers ./routers/
COPY utils ./utils/
COPY templates ./templates/
COPY static ./static/
COPY NeuroSync_Player ./NeuroSync_Player/

# Create models directory
RUN mkdir -p /app/models

# Install Python dependencies
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install --no-build-isolation flash-attn
RUN pip3 install huggingface_hub

# Set environment variables
ENV WHISPER_MODEL=openai/whisper-large-v3-turbo
ENV BLENDSHAPE_MODEL=AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape
ENV TTS_MODEL=marianbasti/XTTS-v2-argentinian-spanish

# Create download script
RUN echo '#!/bin/bash\n\
mkdir -p /app/models\n\
TTS_DIR_NAME=$(basename ${TTS_MODEL})\n\
BLENDSHAPE_DIR_NAME=$(basename ${BLENDSHAPE_MODEL})\n\
\n\
# Set local directory paths\n\
export LOCAL_TTS_DIR=/app/models/${TTS_DIR_NAME}\n\
export LOCAL_BLENDSHAPE_DIR=/app/models/${BLENDSHAPE_DIR_NAME}\n\
\n\
if [ ! -d "/app/models/${TTS_DIR_NAME}" ] || [ ! -d "/app/models/${BLENDSHAPE_DIR_NAME}" ]; then\n\
    echo "Downloading models from Hugging Face..."\n\
    python3 -c "\
from huggingface_hub import snapshot_download\n\
import os\n\
\n\
tts_model = os.environ['\''TTS_MODEL'\''].split('\''/'\'')\n\
blendshape_model = os.environ['\''BLENDSHAPE_MODEL'\''].split('\''/'\'')\n\
\n\
# Download TTS model\n\
if not os.path.exists(f\"/app/models/{tts_model[-1]}\"):\n\
    snapshot_download(repo_id=os.environ['\''TTS_MODEL'\''], local_dir=f\"/app/models/{tts_model[-1]}\")\n\
\n\
# Download Blendshape model\n\
if not os.path.exists(f\"/app/models/{blendshape_model[-1]}\"):\n\
    snapshot_download(repo_id=os.environ['\''BLENDSHAPE_MODEL'\''], local_dir=f\"/app/models/{blendshape_model[-1]}\")\n\
"\n\
fi\n\
./start_server.sh' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Make scripts executable
RUN chmod +x start_server.sh

# Expose port
EXPOSE 8080

# Set the entrypoint
CMD ["/app/entrypoint.sh"]
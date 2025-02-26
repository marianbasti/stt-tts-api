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
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Copy local files to container
COPY . /app/

# Install Python dependencies
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install --no-build-isolation flash-attn

# Make scripts executable
RUN chmod +x start_server.sh

# Expose port
EXPOSE 8080

# Set environment variables
ENV TTS_MODEL=/app/models/XTTS-v2-argentinian-spanish
ENV WHISPER_MODEL=openai/whisper-large-v3-turbo

# Run the FastAPI server
CMD ["./start_server.sh"]
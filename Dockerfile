FROM python:3.10

WORKDIR /code

COPY ./app /code

# Clone BitNet with submodules directly into /code (ensures all files and submodules are present)
RUN git clone --recursive https://github.com/microsoft/BitNet.git /tmp/BitNet && \
    cp -r /tmp/BitNet/* /code && \
    rm -rf /tmp/BitNet

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    cmake \
    clang \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Patch const-correctness error in BitNet source (clang is stricter than gcc)
RUN sed -i 's/int8_t \* y_col = y + col \* by;/const int8_t * y_col = y + col * by;/' /code/src/ggml-bitnet-mad.cpp

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    pip install "fastapi[standard]" "uvicorn[standard]" httpx fastapi-mcp psutil

# Download pre-built GGUF model (skips the broken HF-to-GGUF conversion)
# Use the exact model name "BitNet-b1.58-2B-4T" so setup_env.py recognizes it
RUN huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf --local-dir /code/models/BitNet-b1.58-2B-4T

# Run setup (compile + codegen, model already has the gguf so conversion is skipped)
RUN python /code/setup_env.py -md /code/models/BitNet-b1.58-2B-4T -q i2_s 2>&1 \
    || (cat /code/logs/*.log 2>/dev/null; exit 1)

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
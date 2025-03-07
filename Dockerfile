FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# 나머지 설정은 동일
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# CUDA 관련 패키지 제외한 일반 패키지 설치
COPY requirements.txt .

RUN grep -Ev 'torch|torchvision|torchaudio' requirements.txt > requirements_general.txt \
    && pip install -r requirements_general.txt
CMD ["bash"]
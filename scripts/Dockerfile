# 1. NVIDIA 공식 이미지 가져오기
FROM nvcr.io/nvidia/pytorch:26.01-py3

# 2. 작업 디렉토리 설정
WORKDIR /workspace

# 3. 추가로 필요한 시스템 패키지 설치
RUN apt update && apt install -y git curl

# 4. 추가로 필요한 파이썬 패키지 설치
# 예: open-r1에 필요한 추가 라이브러리들
RUN pip install --no-cache-dir datasets transformers math_verify latex2sympy2_extended accelerate wandb liger-kernel dotenv
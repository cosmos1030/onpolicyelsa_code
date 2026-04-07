#!/bin/bash

# 1. 실행 시간을 초 단위로 가져와서 고유한 ID 생성
ID=$(date +%H%M%S)

# 2. 할당받은 GPU의 UUID 리스트 추출
UUIDLIST=$(nvidia-smi -L | grep UUID | cut -d '(' -f 2 | awk '{print $2}' | tr -d ")" | paste -s -d, -)

# 3. 할당받은 CPU 코어 리스트 추출 
CPULIST=$(grep "Cpus_allowed_list" /proc/self/status | awk '{print $2}')

# 4. Docker 실행
# 컨테이너 이름 뒤에 $ID를 붙여서 중복을 방지합니다.
docker run -it --rm \
    --name ${USER}_open_r1_${ID} \
    --gpus '"device='${UUIDLIST}'"' \
    --cpuset-cpus "${CPULIST}" \
    --shm-size=64gb \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -v $(pwd):/workspace \
    -w /workspace \
    dyk6208/open-r1 \
    /bin/bash
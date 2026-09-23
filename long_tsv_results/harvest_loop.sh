#!/bin/bash
# 30분 주기 long_tsv_results 재수확.
#
# 왜 주기 실행인가: 이전에는 "s3_* 잡이 끝나면 수확"하는 감지형 모니터였는데, 그
# 프로세스가 죽은 걸 아무도 몰랐고 3일 걸린 1.7B s70 ALPS+retrain(965350)의 seed 42가
# 한 시간 넘게 TSV에 반영되지 않았다. wandb에서 전부 다시 읽는 구조라 감지가 필요 없고,
# 한 주기가 실패해도 다음 주기에 저절로 복구된다.
#
# 왜 Monitor가 아니라 nohup 루프인가: 하네스 모니터는 30분에 만료되므로 30분 주기를
# 담지 못한다. 이 루프는 세션과 독립적으로 돈다.
#
# 사용: nohup bash long_tsv_results/harvest_loop.sh > /dev/null 2>&1 & disown
H=/home1/doyoonkim/projects/long_tsv_results
LOG=$H/harvest_cron.log
PY=/home1/doyoonkim/miniconda3/envs/rac/bin/python
INTERVAL=${INTERVAL:-1800}

# 이 루프가 중복 실행되지 않게 잠금
exec 9>"$H/.harvest_loop.lock"
flock -n 9 || { echo "[$(date +%F' '%T)] 이미 다른 루프가 돌고 있음, 종료" >> "$LOG"; exit 0; }

echo "[$(date +%F' '%T)] harvest 루프 시작 (주기 ${INTERVAL}s, pid $$)" >> "$LOG"
while true; do
    # 실제 python 프로세스만 센다. 그냥 `pgrep -f harvest_long_tsv.py`는 이 이름을
    # 명령줄 어딘가에 품은 셸(이 스크립트를 띄운 bash -c, 하네스 모니터 셸 등)까지
    # 잡아서, 아무것도 안 돌 때도 매 주기를 skip 했다(2026-09-21 23:19, 23:49).
    if pgrep -f "python[^ ]* harvest_long_tsv\.py" > /dev/null; then
        # 수동 실행 중이면 TSV 동시 쓰기를 피해 이번 주기는 건너뛴다
        echo "[$(date +%F' '%T)] 다른 harvest 실행 중, skip" >> "$LOG"
    else
        BEFORE=$(md5sum $H/*/*.tsv 2>/dev/null | md5sum | cut -c1-8)
        cd "$H" && timeout 1500 $PY harvest_long_tsv.py >> "$LOG" 2>&1
        RC=$?
        # strict 표(잘렸는데 정답 = 오답)도 같이 갱신. 같은 wandb 데이터를 한 번 더
        # 읽을 뿐이라 비용은 두 배지만, 두 표가 갈라지면 어느 쪽이 최신인지 알 수 없다.
        if [ $RC -eq 0 ]; then
            cd "$H" && HARVEST_STRICT=1 timeout 1500 $PY harvest_long_tsv.py >> "$LOG" 2>&1
            [ $? -ne 0 ] && echo "[$(date +%F' '%T)] STRICT-HARVEST-FAIL" >> "$LOG"
        fi
        AFTER=$(md5sum $H/*/*.tsv 2>/dev/null | md5sum | cut -c1-8)
        if [ $RC -ne 0 ]; then
            echo "[$(date +%F' '%T)] HARVEST-FAIL exit=$RC" >> "$LOG"
        elif [ "$BEFORE" != "$AFTER" ]; then
            echo "[$(date +%F' '%T)] HARVEST-CHANGED" >> "$LOG"
        else
            echo "[$(date +%F' '%T)] harvest ok (변화 없음)" >> "$LOG"
        fi
    fi
    sleep "$INTERVAL"
done

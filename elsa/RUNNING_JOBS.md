# 현재 실행 중인 잡 현황 (OT80/FW20 재실행)

마지막 업데이트: 2026-08-02 (진행 중 — 완료/HF URL은 각 wandb run summary의 `hub_model_id`/`hub_model_url`에서 최종 확인)

데이터: `/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl` (OT80/FW20, Qwen3 토크나이저)

## Dense baseline eval (push_to_hub 없음, 원본 HF 모델 그대로 평가만)
| Job | 모델 | wandb run |
|---|---|---|
| 689471 | Qwen3-1.7B dense | [reasoning_qwen3_1.7b/kfp4aubb](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/kfp4aubb) |
| 689472 | Qwen3-4B dense | [reasoning_qwen3_4b/udim0r15](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/udim0r15) |
| 689473 | Qwen3-8B dense | [reasoning_qwen3_8b/mz8h2gpc](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b/runs/mz8h2gpc) |

## SparseGPT (one-shot, push_to_hub=true) — 전부 완료, HF 업로드 완료
| 모델 | sparsity | wandb run | HF Hub |
|---|---|---|---|
| 1.7B | s50 | [runs/e7bnbwl8](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/e7bnbwl8) | [qwen3-1.7b-sgpt-s50pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-1.7b-sgpt-s50pct-ot80fw20) |
| 1.7B | s60 | [runs/ruvmq3dz](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/ruvmq3dz) | [qwen3-1.7b-sgpt-s60pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-1.7b-sgpt-s60pct-ot80fw20) |
| 1.7B | s70 | [runs/y90smdu6](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/y90smdu6) | [qwen3-1.7b-sgpt-s70pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-1.7b-sgpt-s70pct-ot80fw20) |
| 1.7B | 2:4 | [runs/2h03xjeh](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/2h03xjeh) | [qwen3-1.7b-sgpt-2to4-ot80fw20](https://huggingface.co/cosmos1030/qwen3-1.7b-sgpt-2to4-ot80fw20) |
| 4B | s50 | [runs/oyk1bskx](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/oyk1bskx) | [qwen3-4b-sgpt-s50pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-4b-sgpt-s50pct-ot80fw20) |
| 4B | s60 | [runs/o06kdz7a](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/o06kdz7a) | [qwen3-4b-sgpt-s60pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-4b-sgpt-s60pct-ot80fw20) |
| 4B | s70 | [runs/neofreir](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/neofreir) | [qwen3-4b-sgpt-s70pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-4b-sgpt-s70pct-ot80fw20) |
| 4B | 2:4 | [runs/07tjr91i](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/07tjr91i) | [qwen3-4b-sgpt-2to4-ot80fw20](https://huggingface.co/cosmos1030/qwen3-4b-sgpt-2to4-ot80fw20) |
| 8B | s50 | [runs/8bbx14g2](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b/runs/8bbx14g2) | [qwen3-8b-sgpt-s50pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-8b-sgpt-s50pct-ot80fw20) |
| 8B | s60 | [runs/96wx909u](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b/runs/96wx909u) | [qwen3-8b-sgpt-s60pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-8b-sgpt-s60pct-ot80fw20) |
| 8B | s70 | [runs/2wb6flzu](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b/runs/2wb6flzu) | [qwen3-8b-sgpt-s70pct-ot80fw20](https://huggingface.co/cosmos1030/qwen3-8b-sgpt-s70pct-ot80fw20) |
| 8B | 2:4 | [runs/1923gyhb](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b/runs/1923gyhb) | [qwen3-8b-sgpt-2to4-ot80fw20](https://huggingface.co/cosmos1030/qwen3-8b-sgpt-2to4-ot80fw20) |

*(HF 업로드는 슬럼 스크립트에 `--push_to_hub`가 없어서 처음엔 안 됐던 걸 CPU 잡(689952)으로 소급 업로드함 — 모델 저장 디렉토리에 섞여있던 lighteval 캐시 잡동사니는 allow_patterns로 필터링해서 제외.)
| 689506 | 8B | s60 | [runs/96wx909u](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b/runs/96wx909u) | 〃 |
| 689507 | 8B | s70 | [runs/2wb6flzu](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b/runs/2wb6flzu) | 〃 |
| 689508 | 8B | 2:4 | (PENDING, 리소스 대기) | 〃 |

## ELSA NTP-ADMM plain (push_to_hub=true, steps=4096/global batch=16/lr_warmup=256)
| Job | 모델 | sparsity | lr | lmda | wandb run | HF Hub |
|---|---|---|---|---|---|---|
| 689581 | 1.7B | s50 | 1e-4 | 1e-3 | [runs/gem7785d](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/gem7785d) | run summary의 `hub_model_url` 참고 |
| 689582 | 1.7B | s60 | 1e-4 | 1e-3 | [runs/913p7aab](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/913p7aab) | 〃 |
| 689583 | 1.7B | s70 | 1e-4 | 5e-3 | [runs/k37udbse](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/k37udbse) | 〃 |
| 689584 | 1.7B | 2:4 | 1e-4 | 1e-3 | [runs/i7nfll9a](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/runs/i7nfll9a) | 〃 |
| 689587 | 4B | s50 | 5e-5 | 1e-3 | [runs/27ozhjv6](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/27ozhjv6) | 〃 |
| 689588 | 4B | s60 | 5e-5 | 5e-3 | [runs/jz6u74b7](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/jz6u74b7) | 〃 |
| 689589 | 4B | s70 | 1e-4 | 5e-3 | [runs/dcipudsw](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b/runs/dcipudsw) | 〃 |
| 689590 | 4B | 2:4 | 5e-5 | 5e-3 | (PENDING, 리소스 대기) | 〃 |

## ALPS (one-shot, push_to_hub=true)
| Job | 모델 | sparsity | wandb run |
|---|---|---|---|
| 689591 | 1.7B | s50 | (PENDING) |
| 689592 | 1.7B | s60 | (PENDING) |
| 689593 | 1.7B | s70 | (PENDING) |
| 689594 | 1.7B | 2:4 | (PENDING) |
| 689595 | 4B | s50 | RUNNING (wandb run id 확인 전) |
| 689596 | 4B | s60 | RUNNING (wandb run id 확인 전) |
| 689597 | 4B | s70 | RUNNING (wandb run id 확인 전) |
| 689598 | 4B | 2:4 | RUNNING (wandb run id 확인 전) |
| 689603 | 8B | s50 | (PENDING, online wandb로 재제출) |
| 689604 | 8B | s60 | (PENDING, online wandb로 재제출) |
| 689605 | 8B | s70 | (PENDING, online wandb로 재제출) |
| 689606 | 8B | 2:4 | (PENDING, online wandb로 재제출) |

## SparseLLM (one-shot, push_to_hub=true)
| Job | 모델 | sparsity |
|---|---|---|
| 689938 | 1.7B | s50 |
| 689939 | 1.7B | s60 |
| 689940 | 1.7B | s70 |
| 689941 | 1.7B | 2:4 |
| 689942 | 4B | s50 |
| 689943 | 4B | s60 |
| 689944 | 4B | s70 |
| 689945 | 4B | 2:4 |
| 689946 | 8B | s50 |
| 689947 | 8B | s60 |
| 689948 | 8B | s70 |
| 689949 | 8B | 2:4 |

## 스모크테스트 (검증용, 실제 실험 아님)
| Job | 모델 | 목적 | wandb project | HF Hub |
|---|---|---|---|---|
| 689519 | Qwen3-0.6B SparseGPT | gsm8k/mmlu_redux/push_to_hub 신규 기능 파이프라인 검증 (완료, 성공) | `smoketest_qwen3_0.6b` | https://huggingface.co/cosmos1030/smoketest-qwen3-0.6b-sgpt-s50 |

## 참고
- 각 잡의 최종 HF Hub URL은 학습/프루닝이 끝난 뒤 해당 wandb run의 summary에 `hub_model_id`/`hub_model_url`로 자동 기록됨 (push_to_hub 실패 시에도 평가는 계속 진행되도록 try/except 처리됨).
- ELSA plain 잡들은 현재 데이터셋 토크나이징 단계(최초 1회, 이후 `/home1/doyoonkim/projects/elsa/.cache/datasets`에 캐시되어 같은 모델+데이터 조합이면 재사용됨).
- **MMLU-Redux는 벤치마크 suite에서 제거됨** (2026-08-02): Qwen3 기본 thinking mode 때문에 `generation_size=1`로는 `<think>` 태그 여는 순간 잘려서 57개 서브셋 전부 0점 나옴 — lighteval이 `enable_thinking` kwarg를 넘겨줄 방법이 없어서 원천 차단, non-reasoning probe라 그냥 제거하기로 결정. 이미 완료된 dense 1.7B eval(689471) 등 **과거 결과의 mmlu_redux=0.000은 무시**할 것 — 나머지 4B/8B dense를 포함해 이 수정 이전에 이미 실행 중이던 잡들은 옛 코드가 메모리에 로드된 채로 계속 돌기 때문에 여전히 mmlu_redux=0을 기록함. 이 수정 이후 새로 제출하는 잡부터는 5개 벤치마크(math500/gpqa/ifeval/lcb/gsm8k)만 나옴.

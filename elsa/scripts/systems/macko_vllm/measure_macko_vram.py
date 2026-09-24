"""MACKO 가중치 VRAM: dense / 압축본 / dense 사본을 뺀 경우.

왜 이렇게 재는가: drop_dense=True 로 띄우면 vLLM 이 기동 중 profile_run 으로
prefill 을 돌리다 죽는다(가중치가 없으니 vec(0)). 그래서 정상 로드한 뒤,
압축에 성공한 층의 dense 가중치 바이트만 빼서 계산한다. 스킵 판정은 vLLM 이
qkv/gate_up 을 융합한 뒤의 층(144개) 기준이어야 실제와 같으므로, 체크포인트를
직접 세는 방식(252개)으로는 대체할 수 없다.

    python measure_macko_vram.py --dense <dir> --sparse <dir> [--out x.json]
"""
import argparse
import json

import torch


def _compressed_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_compressed_bytes(o) for o in obj)
    if hasattr(obj, "__dict__"):
        return sum(_compressed_bytes(v) for v in vars(obj).values())
    return 0


def probe(model_path, quant=None, gpu_util=0.80):
    import macko_linear  # noqa: F401  -- "macko" 양자화 방식 등록
    from vllm import LLM
    kw = dict(model=model_path, dtype="float16", max_model_len=2048,
              gpu_memory_utilization=gpu_util, enforce_eager=True,
              max_num_seqs=1, disable_log_stats=True)
    if quant:
        kw["quantization"] = quant
    llm = LLM(**kw)
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    params = sum(p.numel() * p.element_size() for p in m.parameters())
    comp = freed = 0
    seen = compressed = 0
    for mod in m.modules():
        if not hasattr(mod, "macko_density"):
            continue
        seen += 1
        c = getattr(mod, "macko_compressed", None)
        if c is None:
            continue
        compressed += 1
        comp += _compressed_bytes(c)
        # drop_dense=True 였다면 이 층의 dense 가중치가 해제된다.
        freed += mod.weight.numel() * mod.weight.element_size()
    G = 2 ** 30
    return dict(params_gib=params / G, compressed_gib=comp / G, freed_gib=freed / G,
                keep_dense_gib=(params + comp) / G,
                drop_dense_gib=(params - freed + comp) / G,
                linears_seen=seen, linears_compressed=compressed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--tag", default="sparse")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"dense": probe(a.dense), a.tag: probe(a.sparse, "macko")}
    d = res["dense"]["params_gib"]
    print(f"\n  dense                 {d:6.2f} GiB")
    r = res[a.tag]
    print(f"  {a.tag} 압축층          {r['linears_compressed']}/{r['linears_seen']}")
    print(f"  {a.tag} 압축본          {r['compressed_gib']:6.2f} GiB")
    print(f"  {a.tag} dense 사본 유지 {r['keep_dense_gib']:6.2f} GiB  ({r['keep_dense_gib']/d:.2f}x)")
    print(f"  {a.tag} dense 사본 제외 {r['drop_dense_gib']:6.2f} GiB  ({r['drop_dense_gib']/d:.2f}x)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

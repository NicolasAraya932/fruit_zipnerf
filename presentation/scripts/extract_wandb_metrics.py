import json, sys, os
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

RUNS = {
 "fruit_nerf / SIFT":      "/workspace/Desktop/DATASETS/OUTPUTS/cherry_default_fruitnerf/cherry_sift_ds2/fruit_nerf/run_100k/wandb/run-20260802_201553-jf9cflq2/run-jf9cflq2.wandb",
 "fruit_zipnerf / SIFT":   "/workspace/Desktop/DATASETS/OUTPUTS/cherry_fruitzipnerf/cherry_sift_ds2/fruit_zipnerf/run_100k/wandb/run-20260805_000754-h3jzhmmm/run-h3jzhmmm.wandb",
 "fruit_nerf / ALIKED":    "/workspace/Desktop/DATASETS/OUTPUTS/aliked_lightglue_fruitnerf/cherry_aliked_ds2/fruit_nerf/run_100k/wandb/run-20260802_222907-n8svo1yc/run-n8svo1yc.wandb",
 "fruit_zipnerf / ALIKED": "/workspace/Desktop/DATASETS/OUTPUTS/aliked_lightglue_fruitzipnerf/cherry_aliked_ds2/fruit_zipnerf/run_100k/wandb/run-20260805_052826-daydtfai/run-daydtfai.wandb",
}
WANT = ["Eval Metrics Dict/psnr", "Train Metrics Dict/psnr",
        "Eval Metrics Dict/semantic_f1", "Eval Metrics Dict/semantic_iou"]

out = {}
for name, path in RUNS.items():
    ds = DataStore(); ds.open_for_scan(path)
    series = {k: [] for k in WANT}
    summ = {}
    while True:
        d = ds.scan_data()
        if d is None: break
        r = pb.Record()
        try: r.ParseFromString(d)
        except Exception: continue
        w = r.WhichOneof("record_type")
        if w == "history":
            cur = {}
            for it in r.history.item:
                k = "/".join(it.nested_key) if it.nested_key else it.key
                try: cur[k] = json.loads(it.value_json)
                except Exception: pass
            step = cur.get("_step")
            for k in WANT:
                if k in cur and isinstance(cur[k], (int, float)):
                    series[k].append((step, cur[k]))
        elif w == "summary":
            for it in r.summary.update:
                k = "/".join(it.nested_key) if it.nested_key else it.key
                try: summ[k] = json.loads(it.value_json)
                except Exception: pass
    out[name] = {"series": series,
                 "summary": {k: v for k, v in summ.items() if isinstance(v, (int, float))}}
    print(name, {k: len(v) for k, v in series.items()}, file=sys.stderr)

json.dump(out, open("/tmp/psnr_data.json", "w"))
print("written")

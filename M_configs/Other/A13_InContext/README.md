# A13 In-Context (experimental)

This directory publishes an **experimental, non-paper** A13 branch that adds a
nearest-memory / hindsight box-prompt stage on top of OpenRSD. It is provided
for research inspection and follow-up experiments; it is not the final A12
model used for the paper tables.

## Published experiment

- Config: `A13_Hin_rtm_v2_NearestMem.py`
- Checkpoint: `results/MMR_AD_A13_Hin_rtm_v2_NearestMem/epoch_9.pth`
- Checkpoint size: `2,034,151,654` bytes
- SHA-256: `ba1db0ccd06c45a66a8ecb2cd19afe637ce85ae9544a8f0db43fe7abaa05e10b`
- Baidu Netdisk directory: `/OpenRSD/results/MMR_AD_A13_Hin_rtm_v2_NearestMem/`

Download the checkpoint from the OpenRSD Baidu share linked in the repository
README, preserve the relative path above, and prepare the regular OpenRSD data
tree under `./data`. A13 uses the same PCA metadata, negative supports, class
normalization file, training datasets and visual/text support PKLs as A12.

The historical run used four GPUs. Its logged DOTA2 validation AP50 values were
0.646 (epoch 2), 0.645 (epoch 4), 0.641 (epoch 6), and 0.643 (epoch 8). These are
development-run measurements, not a new paper claim. `epoch_9.pth` is the last
available checkpoint from that run.

## Evaluation

After preparing DOTA2 at `data/DOTA2_1024_500`:

```bash
python tools/test.py \
  M_configs/Other/A13_InContext/A13_Hin_rtm_v2_NearestMem.py \
  results/MMR_AD_A13_Hin_rtm_v2_NearestMem/epoch_9.pth \
  --work-dir work_dirs/a13_nearestmem_eval
```

For distributed evaluation, use `tools/dist_test.sh` with the same config and
checkpoint. The configuration contains the complete training data definitions,
but evaluation is the recommended entry point for this experimental release.

## Code relationship and limitations

The implementation is intentionally isolated here and on the experimental Git
branch. The config registers `OpenRTMDet`, the hindsight RTMDet head, the box
prompt head and its bbox head. The additional source files published with this
experiment close that import dependency chain.

`A13_Hin_rtm_v2_NearestMem_LARGE.py` is retained as a historical configuration,
but its referenced `NearestMem_LARGE/epoch_11.pth` was not part of the audited
public assets. Use the non-LARGE config and `epoch_9.pth` above for the released
experiment. A13 has not received the same end-to-end reproduction validation as
the final A12 release.

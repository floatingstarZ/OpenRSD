import torch

src_ckpt = '/workspace/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth'
dst_ckpt = '/workspace/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'

# 显式关闭 weights_only
ckpt = torch.load(src_ckpt, map_location='cpu', weights_only=False)

# 只保留模型权重
new_ckpt = {
    'state_dict': ckpt['state_dict']
}

# 如果你用到了 EMA（很多 mmengine 项目会用）
if 'ema_state_dict' in ckpt:
    new_ckpt['ema_state_dict'] = ckpt['ema_state_dict']

torch.save(new_ckpt, dst_ckpt)

print('Saved weights-only checkpoint to:', dst_ckpt)

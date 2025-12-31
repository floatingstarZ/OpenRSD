# os.system('CUDA_VISIBLE_DEVICES=2 python train_dota.py '
#           './DOTA_configs/DOTA_hbb/retinanet_r50_fpn_2x_dota.py '
#           '--gpus 1 '
#           '--no-validate '
#           '--work-dir ./results/retinanet_hbb_tv_test')
from EXP_CONFIG.CONFIGS.base import Project_root


def gen_dict(name, config,
             result_root=Project_root + '/results',
             epoch=12,
             result_name='results.pkl',
             gpu_num=1,
             note=''):
    name = 'MMR_AD_' + name
    work_dir = result_root + '/' + name

    return dict(
        name=name,
        config=config,
        work_dir=work_dir,
        cp_file=work_dir+'/epoch_%d.pth' % epoch,
        result=work_dir+'/' + result_name,
        submission_dir=work_dir + '/Task1_results',
        eval_results=work_dir + '/eval_results.json',

        bbox_type='HBB',
        data_type='COCO',
        gpu=gpu_num,
        note=note
    )

root = Project_root + '/M_configs/'
aligndet_cfgs = [
    gen_dict('A08_e_rtm_v2_base',
            root + 'Step1_A08_Large_Pretrain/' + 'A08_e_rtm_v2_base.py',
            epoch=36, gpu_num=4),
             epoch=24, gpu_num=4),
    gen_dict('A10_flex_rtm_v3_1_formal_with_hbb',
             root + 'Step2_A10_Large_Pretrain_Stage3/' + 'A10_flex_rtm_v3_1_formal_with_hbb.py',
             epoch=8, gpu_num=4),
    gen_dict('A10_flex_rtm_v3_1_formal',
             root + 'Step2_A10_Large_Pretrain_Stage3/' + 'A10_flex_rtm_v3_1_formal.py',
             epoch=24, gpu_num=4),
    gen_dict('A12_flex_rtm_v3_1_self_training_Labelver5',
            root + 'Step3_A12_SelfTrain/' + 'A12_flex_rtm_v3_1_self_training_Labelver5.py',
            epoch=24, gpu_num=4),
    gen_dict('A12_flex_rtm_v3_1_self_training_Labelver5_HBB',
            root + 'Step3_A12_SelfTrain/' + 'A12_flex_rtm_v3_1_self_training_Labelver5_HBB.py',
            epoch=24, gpu_num=4),

]



aligndet_cfgs = {cfg.pop('name'):cfg for cfg in aligndet_cfgs}

def show_dict(d, n):
    for k,v in d.items():
        print('    ' * n, end='')
        if isinstance(v, dict):
            print(k, ':')
            show_dict(v, n+1)
        else:
            print(k, ':', v)
if __name__ == '__main__':
    show_dict(aligndet_cfgs, 0)


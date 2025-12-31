ood_data_infos = dict(
    Data21_VEDAI=dict(
        data_root='./data/OOD_RSOD_Datasets/Data21_VEDAI',
        train_img_dir='images',
        train_ann_dir='ann_DOTAtype',
        val_img_dir='images',
        val_ann_dir='ann_DOTAtype',
        val_cross_ann_dir='F_DOTA1_Val_Anns',
        class_names=['boat', 'bus', 'camping_car', 'car',
                     'motorcycle', 'pickup', 'plane', 'tractor', 'truck', 'van'],
        img_scale=(1024, 1024),
        val_dataset_flag='Data21_VEDAI',
        support_data='./data/OOD_RSOD_Datasets/Data21_VEDAI/support.pkl',
        cross_name_mapping={
            'boat': 'ship',
            'bus': 'large-vehicle',
            'camping_car': 'large-vehicle',
            'car': 'small-vehicle',
            'motorcycle': None,    # None表示忽略该类别
            'pickup': 'small-vehicle',
            'plane': 'plane',
            'tractor': None,
            'truck': 'large-vehicle',
            'van': 'small-vehicle'
        }
    ),
    # Data22_NWPU_VHR_10=dict(
    #     data_root='./data/OOD_RSOD_Datasets/Data22_NWPU_VHR_10',
    #     train_img_dir='images',
    #     train_ann_dir='ann_DOTAtype',
    #     val_img_dir='images',
    #     val_ann_dir='ann_DOTAtype',
    #     class_names=['airplane', 'baseball_diamond',
    #                  'basketball_court', 'bridge', 'ground_track_field',
    #                  'harbor', 'ship', 'storage_tank', 'tennis_court', 'vehicle'],
    #     img_scale=(1024, 1024),
    #     val_dataset_flag='Data22_NWPU_VHR_10',
    #     support_data='./data/OOD_RSOD_Datasets/Data22_NWPU_VHR_10/support.pkl',
    #     cross_name_mapping={
    #         'airplane': 'plane',
    #         'baseball_diamond': 'baseball-diamond',
    #         'basketball_court': 'basketball-court',
    #         'bridge': 'small-vehicle',
    #         'ground_track_field': None,  # None表示忽略该类别
    #         'harbor': 'small-vehicle',
    #         'ship': 'plane',
    #         'storage_tank': None,
    #         'tennis_court': 'large-vehicle',
    #         'vehicle': 'small-vehicle'
    #     }
    # ),
    Data23_UCAS_AOD=dict(
        data_root='./data/OOD_RSOD_Datasets/Data23_UCAS_AOD',
        train_img_dir='images',
        train_ann_dir='ann_DOTAtype',
        val_img_dir='images',
        val_ann_dir='ann_DOTAtype',
        val_cross_ann_dir='F_DOTA1_Val_Anns',
        class_names=['car', 'plane'],
        img_scale=(1024, 1024),
        val_dataset_flag='Data23_UCAS_AOD',
        support_data='./data/OOD_RSOD_Datasets/Data23_UCAS_AOD/support.pkl',
        cross_name_mapping={
            'car': 'small-vehicle',
            'plane': 'plane',
        }
    ),
    Data24_CORS_ADD_OBB=dict(
        data_root='./data/OOD_RSOD_Datasets/Data24_CORS_ADD_OBB',
        train_img_dir='images/train2017_png',
        train_ann_dir='dota_labels/train2017',
        val_img_dir='images/val2017_png',
        val_ann_dir='dota_labels/val2017',
        val_cross_ann_dir='F_DOTA1_Val_Anns',
        class_names=['plane'],
        img_scale=(1024, 1024),
        val_dataset_flag='Data24_CORS_ADD_OBB',
        support_data='./data/OOD_RSOD_Datasets/Data24_CORS_ADD_OBB/support.pkl',
        cross_name_mapping={
            'plane': 'plane',
        }
    ),
    Data25_DOSR=dict(
        data_root='./data/OOD_RSOD_Datasets/Data25_DOSR',
        train_img_dir='Images',
        train_ann_dir='labelTxt_Ship',
        val_img_dir='Images',
        val_ann_dir='labelTxt_Ship',
        val_cross_ann_dir='F_DOTA1_Val_Anns',
        class_names=['ship'],
        # class_names=['auxiliary_ship', 'barge', 'bulk_cargo_vessel', 'cargo',
        #              'communication_ship', 'container', 'cruise', 'deckbarge',
        #              'deckship', 'fishing_boat', 'flat_traffic_ship', 'floating_crane',
        #              'military_ship', 'multihull', 'speedboat', 'submarine', 'tanker',
        #              'transpot', 'tug', 'yacht'],
        img_scale=(1024, 1024),
        val_dataset_flag='Data25_DOSR',
        support_data='./data/OOD_RSOD_Datasets/Data25_DOSR/support.pkl',
        cross_name_mapping={
            'ship': 'ship',
        }
    ),
    Data26_SODA_A_800_150=dict(
        data_root='./data/OOD_RSOD_Datasets/Data26_SODA_A_800_150',
        train_img_dir='train/images',
        train_ann_dir='train/labelTxt',
        val_img_dir='val/images',
        val_ann_dir='val/labelTxt',
        val_cross_ann_dir='F_DOTA1_Val_Anns',
        class_names=['airplane', 'container',
                     'helicopter', 'large-vehicle', 'ship', 'small-vehicle',
                     'storage-tank', 'swimming-pool', 'windmill'],
        img_scale=(1024, 1024),
        val_dataset_flag='Data26_SODA_A_800_150',
        support_data='./data/OOD_RSOD_Datasets/Data26_SODA_A_800_150/support.pkl',
        cross_name_mapping={
            'airplane': 'plane',
            'container': None,
            'helicopter': 'helicopter',
            'large-vehicle': 'large-vehicle',
            'ship': 'ship',
            'small-vehicle': 'small-vehicle',
            'storage-tank': 'storage-tank',
            'swimming-pool': 'swimming-pool',
            'windmill': None,
        }
    ),
    Data27_UBCv2_Finegrained_DOTA=dict(
        data_root='./data/OOD_RSOD_Datasets/Data27_UBCv2_Finegrained_DOTA',
        train_img_dir='train/images',
        train_ann_dir='train/labelTxt',
        val_img_dir='val/images',
        val_ann_dir='val/labelTxt',
        val_cross_ann_dir='F_DOTA1_Val_Anns',
        class_names=['building'],
        img_scale=(800, 800),
        val_dataset_flag='Data27_UBCv2_Finegrained_DOTA',
        support_data='./data/OOD_RSOD_Datasets/Data27_UBCv2_Finegrained_DOTA/support.pkl',
        cross_name_mapping={
            'building': 'building'
        }
    ),

)
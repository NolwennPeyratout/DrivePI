data_args = dict(
    data_path=[
       "/home/renault/repo/dataset/nuscenes/LiDAR-LLM-Nu-Caption/train_llava.json"
    ],
    bev_feature_folder="/home/renault/repo/dataset/nuscenes/unilion_bev_feats_train/",
    lazy_preprocess=True,
    is_multimodal=False,
    image_aspect_ratio='square',
    feature_hidden_size=384,  # BEV feature dimension - matches model config
)
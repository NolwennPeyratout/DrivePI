# ──────────────────────────────────────────────────────────────────────────────
# DrivePI — 0.5B model, 1 epoch, lr=1e-4
# Full fine-tuning with OCC prediction + path planning on NuScenes BEV features.
# ──────────────────────────────────────────────────────────────────────────────

# Base configs that are merged and then overridden by the dicts below.
#   - qwen2_5_empty_vision : base model architecture (Qwen2.5 + EmptyVisionTower)
#   - bev_feature_qa_* dataset : NuScenes BEV feature / QA / OCC / action data pipeline
#   - training/default.py  : shared optimizer, LoRA, quantization defaults
_base_ = [
    '../../_base_/models/qwen2_5_empty_vision.py',
    '../../_base_/datasets/bev_feature_qa_pretrain_0722_0815_occ_fullv3vol_det_qa_action.py',
    '../../_base_/training/default.py'
]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARGS
# Controls the architecture, pretrained weights, and which heads are active.
# ══════════════════════════════════════════════════════════════════════════════
model_args = dict(
    # Conversation template / tokenizer chat format to use.
    # "qwen2" selects the Qwen2/2.5 chat template.
    version="qwen2",

    # Path to a pretrained multimodal projector (mm_projector.bin) produced by
    # a previous pretraining stage. The projector maps BEV tokens into the LLM
    # embedding space. Set to None to train from scratch.
    pretrain_mm_mlp_adapter="/path/work_dirs/emova_llava-qwen2_5-0.5b-bev-featurev2-sms-pretrain0722_three/mm_projector.bin",

    # ── Language model ──────────────────────────────────────────────────────
    language_model=dict(
        # Custom Qwen2 class that adds OCC / action prediction hooks.
        type='EmovaQwen2ForCausalLM',
        # HuggingFace model ID (or local path) for the base LLM weights.
        pretrained_model_name_or_path='Qwen/Qwen2.5-0.5B-Instruct',
        # Use Flash-Attention 2 for faster, memory-efficient attention.
        attn_implementation="flash_attention_2",
        # Load weights from the pretrained checkpoint above (vs. random init).
        from_pretrained=True,
        # How to fuse multiple hidden states when multi-layer features are used.
        # 'none' = use the last hidden state only.
        # Other options: 'weighted_sum', 'gated', 'max_pooling', 'avg_pooling'
        hidden_fusion_method='none',
        # Whether the fusion weights themselves are learnable parameters.
        learn_fusion_weights=True,
        # Use the last-token hidden state as the text feature for downstream heads.
        use_text_last_fetas=True,
        # Scalar multiplier applied to the OCC segmentation loss inside the LLM.
        occ_loss_weight=1.0,
        # Scalar multiplier applied to the OCC flow regression loss inside the LLM.
        occ_flow_loss_weight=1.0,
        # Allow gradients to flow through the LLM (full fine-tune).
        trainable=True,
    ),

    # ── Vision tower ────────────────────────────────────────────────────────
    # EmptyVisionTower: a pass-through encoder for pre-computed BEV features.
    # Kept frozen — the BEV encoder is not retrained here.
    mm_vision_tower=dict(trainable=False),

    # ── Multimodal projector ─────────────────────────────────────────────────
    # Maps BEV tokens (180×180) into a smaller token sequence for the LLM.
    mm_projector=dict(
        # SelfMiningMMProjector: learns which spatial tokens are most informative.
        type='SelfMiningMMProjector',
        # Number of stacked MLP layers in the projector.
        mlp_depth=2,
        # Spatial downsampling ratio applied before the MLP (9× → 20×20 patches).
        downsample_rate=9,
        # Target spatial grid size after downsampling.
        downsample_size=(60, 60),
        # Total number of input tokens from the vision tower (180*180 = 32 400).
        num_input_token=180 * 180,
        # Train the projector weights.
        trainable=True,
    ),

    # ── BEV Occupancy estimation head ────────────────────────────────────────
    # Takes LLM features and predicts a 3-D occupancy voxel grid over the scene.
    bev_occ_estimator=dict(
        type='OCCHead',
        # Number of channels in the internal feature maps of the head.
        hidden_channel=256,
        # Number of vertical (Z-axis) voxel bins.
        Dz=16,
        # Also predict per-voxel flow vectors (velocity in BEV space).
        flow=True,
        # Defines the metric range and resolution of the BEV voxel grid.
        # Each sub-list is [min, max, step] for [X, Y] axes (metres).
        bev_occ_config=dict(
            input_scope=[[-51.2, 51.2, 0.4], [-51.2, 51.2, 0.4]],
            output_scope=[[-51.2, 51.2, 0.4], [-51.2, 51.2, 0.4]],
            # Multiplier applied to input features before the head (1.0 = no scaling).
            prescale_factor=1.0
        ),
        # Apply a validity mask so that unobserved voxels do not contribute to loss.
        use_mask=True,
        # Semantic categories predicted per voxel (16 objects + 'free').
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation', 'free'
        ],
        # Must match len(class_names).
        num_classes=17,
        # Spatial output resolution of the occupancy grid [W, H] in voxels.
        size=[200, 200],
        # Re-weight loss by inverse class frequency to handle class imbalance.
        class_balance=True,
        # Voxel-wise semantic classification loss.
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            # Scaled by occ_loss_weight at the model level.
            loss_weight=1.0
        ),
        # Per-voxel flow regression loss (L1 on predicted vs. GT velocity).
        loss_occ_flow=dict(
            type='L1Loss',
            # Scaled by occ_flow_loss_weight at the model level.
            loss_weight=1.0
        ),
        # Add a linear layer to project LLM hidden states → hidden_channel before
        # feeding them into the OCC head.
        feature_proj=True,
        trainable=True,
        # How to fuse LLM features with BEV features. 'none' = no BEV fusion,
        # use LLM features only.
        fusion_type='none'
    ),

    # ── Diffusion anchor planner head ────────────────────────────────────────
    # Predicts future ego-vehicle trajectories using a diffusion-based decoder
    # conditioned on anchor waypoints derived from k-means clustering.
    diff_anchor_planner_head=dict(
        type='DiffAnchorPlannerHead',
        # Pre-computed k-means cluster centres used as planning anchors.
        planning_anchor='/home/renault/repo/dataset/nuscenes/kmeans_planning_6.npy',
        # Total input feature dimension fed to the planner
        # (sum of three 128-channel feature maps).
        in_channels=sum([128, 128, 128]),
        # Hidden feature size inside the planner MLP / decoder.
        hidden_channel=128,
        # Number of transformer decoder layers for trajectory refinement.
        num_decoder_layers=1,
        # Voxel grid parameters used when encoding spatial context.
        planning_config=dict(
            # [W, H, Z] grid resolution for the planner's BEV encoding.
            grid_size=[360, 360, 32],
            # Stride factor: effective spatial resolution = voxel_size * out_size_factor.
            out_size_factor=2
        ),
        # Focal loss for anchor classification (which k-means cluster is closest
        # to the GT trajectory).
        loss_plan_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,   # Focusing parameter — down-weights easy negatives.
            alpha=0.25,  # Class balance factor.
            reduction='mean',
            loss_weight=5.0),
        # L1 loss for continuous waypoint regression relative to the chosen anchor.
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        # Add a linear projection from LLM hidden states → hidden_channel.
        feature_proj=True,
        trainable=True,
        # 'none' = use projected LLM features only, no additional BEV fusion.
        fusion_type='none'
    ),

    # ── Global loss weights ──────────────────────────────────────────────────
    # Top-level scalars that multiply the corresponding head losses before summing
    # into the total training loss.
    occ_loss_weight=1.0,       # Weight for occupancy classification loss.
    occ_flow_loss_weight=1.0,  # Weight for occupancy flow regression loss.
    planning_loss_weight=1.0,  # Weight for trajectory planning loss.

    # ── Task switches ────────────────────────────────────────────────────────
    enable_occ_prediction=True,  # Run OCC head and include its loss during training.
    enable_planning=True,        # Run planner head and include its loss during training.
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA ARGS
# Paths and options for the training data pipeline.
# ══════════════════════════════════════════════════════════════════════════════
data_args = dict(
    # Root directory containing pre-extracted BEV feature files (.npy / .pt)
    # produced by the upstream BEV encoder (e.g. UniLion).
    bev_feature_folder="/home/renault/repo/dataset/nuscenes/unilion_bev_feats_train/",

    # Defer tokenisation to the DataLoader workers instead of doing it all
    # upfront — reduces RAM usage for large datasets.
    lazy_preprocess=True,

    # Whether the dataset returns image tensors alongside text.
    # False here because we use pre-extracted BEV features, not raw images.
    is_multimodal=False,

    # Padding / resize strategy for raw images (unused when is_multimodal=False).
    image_aspect_ratio='square',

    # Channel dimension of each BEV feature vector.
    # Must match mm_vision_tower.hidden_size in the base model config (384).
    feature_hidden_size=384,

    # ── Occupancy grid data ──────────────────────────────────────────────────
    # Directory with ground-truth 3-D occupancy voxel grids (OpenOcc format).
    occ_grid_dir="/path/DrivePI_Data/saved_openocc_gt_occ_train",
    # Directory with ground-truth per-voxel flow vectors.
    occ_flow_dir="/path/DrivePI_Data/saved_openocc_gt_occ_flow_train",
    # Load and feed occupancy labels to the model during training.
    include_occ_grid=True,
    include_occ_flow=True,

    # ── Path planning data ───────────────────────────────────────────────────
    # Directory with ground-truth ego-vehicle future trajectories / actions.
    planning_dir="/path/DrivePI_Data/saved_action_train",
    # Load and feed planning labels to the model during training.
    include_planning=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ARGS
# HuggingFace Trainer / DeepSpeed settings.
# ══════════════════════════════════════════════════════════════════════════════
training_args = dict(
    # Where checkpoints and TensorBoard logs are written.
    output_dir="/path/work_dirs/drivepi_05b_1e_4/",

    # DeepSpeed ZeRO-3 config: shards optimizer states, gradients, and model
    # parameters across all GPUs to minimise per-GPU memory.
    deepspeed="./scripts/zero3.json",

    # ── Precision ────────────────────────────────────────────────────────────
    bf16=True,   # Use bfloat16 mixed precision (preferred on Ampere+ GPUs).
    fp16=False,  # Disable float16 (use bf16 instead).
    tf32=True,   # Enable TensorFloat-32 on Ampere+ for faster matmuls.

    # ── Schedule ─────────────────────────────────────────────────────────────
    num_train_epochs=1,  # Total number of passes over the training dataset.

    # ── Batch size & gradient accumulation ───────────────────────────────────
    per_device_train_batch_size=1,   # Samples per GPU per forward pass.
    per_device_eval_batch_size=4,    # Samples per GPU during evaluation.
    # Effective batch size = per_device * num_gpus * gradient_accumulation_steps.
    # e.g. 1 * 8 GPUs * 8 = 64 effective samples per optimizer step.
    gradient_accumulation_steps=8,

    # ── Evaluation & checkpointing ───────────────────────────────────────────
    evaluation_strategy="no",  # No periodic evaluation during training.
    save_strategy="steps",     # Save a checkpoint every `save_steps` steps.
    save_steps=500,            # Checkpoint frequency (in optimizer steps).
    save_total_limit=3,        # Keep only the 3 most recent checkpoints.
    # Each node writes its own checkpoint shard (required for multi-node ZeRO-3).
    save_on_each_node=True,

    # ── Optimiser ────────────────────────────────────────────────────────────
    learning_rate=1e-4,   # Peak learning rate for AdamW.
    weight_decay=0.01,    # L2 regularisation coefficient.
    # Fraction of total steps used for linear LR warm-up before cosine decay.
    warmup_ratio=0.4,
    lr_scheduler_type="cosine",  # Cosine annealing after the warm-up phase.

    # ── Logging ──────────────────────────────────────────────────────────────
    logging_steps=10,  # Log training metrics every N optimizer steps.

    # ── Sequence length ───────────────────────────────────────────────────────
    # Maximum token length for inputs + generation.
    # 4096 for the LLM context + 1024 extra budget for long BEV token sequences.
    model_max_length=4096+1024,

    # ── Memory optimisations ──────────────────────────────────────────────────
    # Recompute activations during the backward pass instead of storing them,
    # trading compute for GPU memory.
    gradient_checkpointing=True,
    # Number of CPU worker processes for the DataLoader.
    dataloader_num_workers=4,
    # Group samples of similar sequence length into the same batch to reduce
    # padding waste and speed up training.
    group_by_modality_length=True,

    # ── Experiment tracking ───────────────────────────────────────────────────
    report_to="tensorboard"  # Stream metrics to TensorBoard.
)
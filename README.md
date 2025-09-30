# DRAIL_tiamat

```
python tiamat_fsm_task_scripts/offline_perception_processor.py \
    -p "tiamat_fsm_task_scripts/obs_buffer.pkl" \
    -s "tiamat_fsm_task_scripts/models/sam_vit_l_0b3195.pth" \
    -v "HuggingFaceTB/SmolVLM-256M-Instruct" \
    -d "cuda" \
    -l 50 \
    --centroid_threshold 200 \
    --coverage_threshold 0.70 \
    --voxel_size 0.30 \
    --min_mask_area 100 \
    --vlm_padding 0.10
```

"-l" : #image processing limitation. To ignore it, just put 'false' or '0', then it iterates all images of PKL file

# HandFirst-SLG
HandFirst-SLG: Two-Stage Pose-Guided Hand-Optimized Sign Language Generation

## Environment
We Recommend a python version `=3.10` and CUDA `=11.8`
```shell
# Install with pip:
pip install -r requirements.txt
```

## Inference
```shell
# For Stage1: Hand Region Image Generation
python scripts/stage1.py
# Then align generated hand-region image&mask to corresponding place
pyrhon scripts/hand_align.py
# For Stage2: Body Outpainting & Video Generation
python scripts/stage2.py
```

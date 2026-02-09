# 1. In login node
conda create -n env python=3.1x -y
conda activate env
pip install torch torchvision torchaudio --index-url https:/download.pytorch.org/whl/cu12x

# 2. Switch to GPU node
srun -w ...
conda activate env
cd model_folder
pip install -e ".[notebooks]"

# 3. Modification in model_builder.py (line 560 & line 653)

## Line 560
``` python
def build_sam3_image_model( bpe_path=None, 
	device="cuda" if torch.cuda.is_available() else "cpu", 
	eval_mode=True,   
	checkpoint_path=None, 
	load_from_HF=True, 
	enable_segmentation=True, 
	enable_inst_interactivity=False, 
	compile=False, 
):
```

to load the ckpts correctly, set `checkpoint_path` manually and set `load_from_HF` to false

```python
def build_sam3_image_model( 
	bpe_path=None, 
	device="cuda" if torch.cuda.is_available() else "cpu", 
	eval_mode=True,
checkpoint_path="/bd_byt4090i0/users/omnimotion/dayu/sam3/checkpoints/sam3.pt", 
	load_from_HF=False, 
	enable_segmentation=True, 
	enable_inst_interactivity=False, 
	compile=False, 
):
```

## Line 653
```python
def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
```
to
```python
def build_sam3_video_model(
    checkpoint_path: Optional[str] = "/bd_byt4090i0/users/omnimotion/dayu/sam3/checkpoints/sam3.pt",
    load_from_HF=False,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
```

# 4. 无法解析导入"..."
尝试重新选择解释器或pip install


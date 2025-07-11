export FLUX_DEV="FLUX.1-dev-PATH/flux1-dev.safetensors"
export AE="FLUX.1-dev-PATH/ae.safetensors"
export T5="xflux_text_encoders-PATH"
export CLIP="clip-vit-large-patch14-PATH"
export LORA="./ckpts/dit_lora.safetensors"


# img-guided style transfer
CUDA_VISIBLE_DEVICES=0 python inference_img_guided.py

# instruction-guided style transfer
CUDA_VISIBLE_DEVICES=0 python inference_instruction_guided.py
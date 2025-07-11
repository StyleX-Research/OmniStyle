import os
import dataclasses
from typing import Literal
from accelerate import Accelerator
from transformers import HfArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image, ImageDraw, ImageFont


from omnistyle.flux.pipeline import DSTPipeline
from tqdm import tqdm


@dataclasses.dataclass
class InferenceArgs:
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 1024
    height: int = 1024
    ref_size: int = 1024
    num_steps: int = 25
    guidance: float = 4
    seed: int = 0
    only_lora: bool = True
    concat_refs: bool = True
    lora_rank: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'


def crop_if_not_square(img):
    w, h = img.size
    if w != h:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img = img.crop((left, top, right, bottom))
    return img


def main(args: InferenceArgs):
    
    accelerator = Accelerator()
    device = accelerator.device

    test_cnt_folder = "./test/content"
    test_sty_instruction_txt = "./test/instruction.txt"
    save_folder = "./output/instruction"
    os.makedirs(save_folder, exist_ok=True)
    
    
    from omnistyle.flux.pipeline import DSTPipeline
    pipeline = DSTPipeline(
        args.model_type,
        device,
        False,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank
    )

    
    for cnt_img in os.listdir(test_cnt_folder):
        for sty_instruction in open(test_sty_instruction_txt).readlines():
    
            cnt_path = os.path.join(test_cnt_folder, cnt_img)
            cnt_pil = Image.open(cnt_path).convert("RGB")
            cnt_crop = crop_if_not_square(cnt_pil).resize((args.width, args.height))
            
            image_gen = pipeline(
                prompt=sty_instruction,
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed,
                ref_imgs=[cnt_crop],
                pe=args.pe,
            )

            if args.concat_refs:
                save_name = os.path.join(save_folder, f"{os.path.splitext(cnt_img)[0]}_{sty_instruction}.jpg")
                canvas = Image.new("RGB", (args.width * 2, args.height), (255, 255, 255))  # 纯白背景
                canvas.paste(cnt_crop, (0, 0))
                canvas.paste(image_gen, (args.width, 0))
                # 在顶部写文字
                canvas.save(save_name)
                
            else:
                image_gen.save(save_name)



if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)

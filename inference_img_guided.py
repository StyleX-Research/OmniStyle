import os
import dataclasses
from typing import Literal
from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
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
    test_sty_folder = "./test/style"
    
    save_folder = "./output/img"
    os.makedirs(save_folder, exist_ok=True)
    
    pipeline = DSTPipeline(
        args.model_type,
        device,
        accelerator.state.deepspeed_plugin is not None,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank
    )

    for sty_img in os.listdir(test_sty_folder):
        for cnt_img in os.listdir(test_cnt_folder):
            
            save_name = os.path.join(save_folder, f"{os.path.splitext(cnt_img)[0]}@{os.path.splitext(sty_img)[0]}.jpg")
            if os.path.exists(save_name):
                continue
            
            cnt_path = os.path.join(test_cnt_folder, cnt_img)
            sty_path = os.path.join(test_sty_folder, sty_img)

            cnt_img_pil = Image.open(cnt_path).convert('RGB')
            sty_img_pil = Image.open(sty_path).convert('RGB')
            cnt_center_crop = crop_if_not_square(cnt_img_pil)
            sty_center_crop = crop_if_not_square(sty_img_pil)
            
            cnt_img_pil = cnt_center_crop.resize((args.width, args.height))
            sty_img_pil = sty_center_crop.resize((args.width, args.height))
            

            ref_imgs = [sty_img_pil, cnt_img_pil]

            image_gen = pipeline(
                prompt="",
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed,
                ref_imgs=ref_imgs,
                pe=args.pe,
            )

            if args.concat_refs:
                new_blank_img = Image.new('RGB', (args.width * 3, args.height))
                new_blank_img.paste(cnt_img_pil, (0, 0))
                new_blank_img.paste(sty_img_pil, (args.width, 0))
                new_blank_img.paste(image_gen, (args.width * 2, 0))

            new_blank_img.save(save_name)

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)

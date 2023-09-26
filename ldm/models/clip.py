import torch
from PIL import Image
import open_clip
from torch import nn, Tensor
import torchvision.transforms.v2 as T


class OpenClip(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        precision: str = "fp16",
        jit: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            jit=jit,
            device=device,
        )
        self.model = model
        resize = preprocess.transforms[0]
        resize.antialias = True
        crop = preprocess.transforms[1]
        normalize = preprocess.transforms[4]
        self.preprocess = T.Compose([resize, crop, normalize])
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_image(self, x: Tensor) -> Tensor:
        x = self.preprocess(x)
        x_emb = self.model.encode_image(x).unsqueeze(1)
        return x_emb

    @torch.no_grad()
    def encode_text(self, x: list[str]) -> Tensor:
        x = self.tokenizer(x)
        x = self.model.encode_text(x).unsqueeze(1)
        return x

import torch
from torchvision import transforms
from torch import nn
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image
from collections.abc import Sequence
from pathlib import Path

from attnreg.classes import IMAGENET2012_CLASSES as imgnet_dict
from attnreg.LitModel import LitModel


imgnet_dict_inv = {v: k for k, v in imgnet_dict.items()}
idx2class = {i: j for i, j in enumerate(imgnet_dict.values())}
class2idx = {i: j for j, i in idx2class.items()}

def print_top_classes(predictions, **kwargs):
    """Print Top-5 predictions"""
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(idx2class[cls_idx])
        if len(idx2class[cls_idx]) > max_str_len:
            max_str_len = len(idx2class[cls_idx])

    print("Top 5 classes:")
    for cls_idx in class_indices:
        output_string = "\t{} : {}".format(cls_idx, idx2class[cls_idx])
        output_string += " " * (max_str_len - len(idx2class[cls_idx])) + "\t\t"
        output_string += "value = {:.3f}\t prob = {:.1f}%".format(
            predictions[0, cls_idx], 100 * prob[0, cls_idx]
        )
        print(output_string)


class ViTWrapper:
    """
    High-level wrapper for vision transformers that allows to implement custom models, while preserving the workflow of image processing and regularization pipelines. 
    The wrapper standardizes image preprocessing (including resize and normalization) and attention map extraction -- any suitable ViT model should contain the 'get_last_selfattention' method. 
    The wrapper also contains the hooks for gradient-based methods (CDAM).
    """
    def __init__(
        self,
        model,
        img_size=(488,488),
        patch_size=8,
        device="cpu",
        preprocess=None,
        normalization: tuple[Sequence[float],Sequence[float]] | None = None,
        register_hooks=False,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (
            (img_size[0] // patch_size) * (img_size[1] // patch_size)
        )

        self.preprocess = preprocess
        self.normalization = normalization
        
        self.transform = transforms.Compose(
            [transforms.ToTensor()] + 
            (preprocess.transforms if preprocess else []) + 
            (
                [transforms.Normalize(
                    mean=self.normalization[0],
                    std=self.normalization[1]
                )] 
                if self.normalization else []
            )
        )

        # Ensure required method exists
        if not hasattr(model, "get_last_selfattention"):
            raise TypeError(
                "The provided model does not implement `get_last_selfattention(x)`"
            )

        if not callable(model.get_last_selfattention):
            raise TypeError(
                "`get_last_selfattention` exists but is not callable"
            )

        # Storage for hooks
        self.activation = {}
        self.grad = {}
        self._activation_hook = None
        self._grad_hook = None

        if register_hooks:
            self._register_last_block_hooks()

    @classmethod
    def default(cls):
        """
        Default instance of the wrapper with a LitModel and a loaded checkpoint.
        """

        device="cpu"

        ckpt_path = Path(__file__).parent / "best-checkpoint-full-imgnet-augment.ckpt"
        lit_model = LitModel.load_from_checkpoint(str(ckpt_path)).to(device).eval()
      
        model = lit_model.model
        model.head = lit_model.linear
        
        return cls(
            model=model,
            img_size=(488,488),
            patch_size=8,
            device=device,
            preprocess=None,
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            register_hooks=False,
        )
            

    def _register_last_block_hooks(self):
        """
        Register hooks on the final block normalization layer
        """
        final_block_norm1 = self.model.blocks[-1].norm1

        def forward_hook(module, input, output):
            # output is a Tensor
            self.activation["last_att_in"] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple
            self.grad["last_att_in"] = grad_output[0]

        self._activation_hook = final_block_norm1.register_forward_hook(
            forward_hook
        )

        self._grad_hook = final_block_norm1.register_full_backward_hook(
            backward_hook
        )

    def remove_hooks(self):
        """
        Cleanly remove hooks to avoid memory leaks
        """
        if self._activation_hook is not None:
            self._activation_hook.remove()
            self._activation_hook = None

        if self._grad_hook is not None:
            self._grad_hook.remove()
            self._grad_hook = None

    def load_img(self, image_path, show=False):
        """Returns the image as resized and normalized tensor and as original (only resized)"""
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize(self.img_size)
            original_img = copy.deepcopy(img)

            if show:
                plt.imshow(original_img)
                plt.axis("off")
                plt.show()
    
        img = self.transform(img).to(self.device)
    
        # make image divisible by patch size
        w, h = (
            img.shape[1] - img.shape[1] % self.patch_size,
            img.shape[2] - img.shape[2] % self.patch_size,
        )
        img = img[:, :w, :h].unsqueeze(0)
        img.requires_grad = True
        return img, original_img

    def load_PIL(self, image_path, verbose = True):
        """Returns the image as resized and normalized tensor and as original (only resized)"""
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize(self.img_size)
            
        # make image divisible by patch size
        w, h = (
            img.size[0] - img.size[0] % self.patch_size,
            img.size[1] - img.size[1] % self.patch_size,
        )
        
        img = img.crop((0, 0, w, h))
        if verbose:
            print(f"Size: {img.size}")
        return img

    def PIL_to_tensor(self, img):
        """Converts a PIL image to a tensor"""
        img = self.transform(img).to(self.device).unsqueeze(0)
        return img

    def denormalize(self, img: torch.tensor):
        """Applies an inverse normalization to a tensor image"""
        mean, std = self.normalization
        inv_normalize = transforms.Normalize(
            mean=[-m/s for m,s in zip(mean,std)],
            std=[1/s for s in std]
        )
        img = inv_normalize(img)
        return img
        
    def __getattr__(self, name):
        """
        Delegate all unknown attributes/methods to the wrapped model.
        """
        return getattr(self.model, name)



def get_attention_map(
    model: ViTWrapper, 
    sample_img: torch.tensor, 
    head=None, 
    return_raw=False, 
    return_mean=True, 
    interpolation_mode="nearest",
)->np.ndarray:
    
    """This returns the attentions when CLS token is used as query in the last attention layer, averaged over all attention heads"""

    if sample_img.ndim < 4:
        sample_img = sample_img.unsqueeze(0)

    attentions = model.get_last_selfattention(sample_img) # (batch, head, num_patches, num_patches)

    w_featmap = sample_img.shape[-2] // model.patch_size
    h_featmap = sample_img.shape[-1] // model.patch_size

    batch_size = attentions.shape[0]
    nh = attentions.shape[1]  # number of heads

    # this extracts the attention when cls is used as query
    attentions = attentions[:, :, 0, 1:].reshape(batch_size, nh, -1) # (batch, head, num_patches-1)
    
    if return_raw:
        return torch.mean(attentions, dim=1).detach().cpu().numpy() # (batch, num_patches-1)

    attentions = attentions.reshape(batch_size, nh, w_featmap, h_featmap) # (batch, head, w_featmap, h_featmap)
    # attentions = torch.nn.functional.interpolate(
    #     attentions.unsqueeze(0), scale_factor=model.patch_size, mode=interpolation_mode
    # )[0]
    attentions = torch.nn.functional.interpolate(
        attentions, scale_factor=model.patch_size, mode=interpolation_mode
    ) # (batch, head, img_w, img_h)
    
    if head == None:
        if not return_mean:
            return attentions.detach().cpu().numpy() # (batch, head, img_w, img_h)
        mean_attention = torch.mean(attentions, dim=1).detach().cpu().numpy() # (batch, img_w, img_h)
        return mean_attention
    else:
        return attentions[:,head].squeeze().detach().cpu().numpy() # (batch, img_w, img_h)


def get_CDAM(class_score, activation, grad, patch_size = 8, clip=False, return_raw=False):
    """The class_score can either be the activation of a neuron in the prediction vector or a similarity score between the latent representations of a concept and a sample"""
    class_score.backward()
    # Token 0 is CLS and others are 60x60 image patch tokens
    tokens = activation["last_att_in"][1:]
    grads = grad["last_att_in"][0][0, 1:]

    attention_scores = torch.tensor(
        [torch.dot(tokens[i], grads[i]) for i in range(len(tokens))]
    )

    if return_raw:
        return attention_scores
    else:
        # clip for higher contrast plots
        if clip:
            attention_scores = torch.clamp(
                attention_scores,
                min=torch.quantile(attention_scores, 0.001),
                max=torch.quantile(attention_scores, 0.999),
            )
        w = int(np.sqrt(attention_scores.squeeze().shape[0]))
        attention_scores = attention_scores.reshape(w, w)

        return torch.nn.functional.interpolate(
            attention_scores.unsqueeze(0).unsqueeze(0),
            scale_factor=patch_size,
            mode="nearest",
        ).squeeze()

def get_class_map(model: ViTWrapper, image: torch.tensor, target_class: str = None, return_raw=False, clip=False):
    """Wrapper function to get the attention map and the concept map for a given image and target class"""

    if target_class is None:
        target_class = "placeholder"
        
    pred = model(image)
    class_idx = class2idx[target_class]
    class_attention_map = get_CDAM(
        class_score=pred[0][class_idx],
        activation=model.activation,
        grad=model.grad,
        return_raw=return_raw,
        clip=clip
    )
    
    return class_attention_map



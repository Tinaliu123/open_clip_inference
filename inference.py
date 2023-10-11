# inference.py
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import open_clip
import os
from collections import OrderedDict
from typing import Optional, Union, Tuple,List
from open_clip import load_checkpoint,get_model_config,list_models
import logging
import json
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.model import CLIP, CustomTextCLIP, convert_weights_to_lp, get_cast_dtype
from open_clip.coca_model import CoCa
from open_clip.openai import load_openai_model
from open_clip.pretrained import list_pretrained_tags_by_model, download_pretrained_from_hf
from open_clip.transform import image_transform
HF_HUB_PREFIX = 'hf-hub:'

def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        pretrained_path: Optional[str] = None,  # 新增参数，用于指定权重文件路径
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cuda',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
)-> torch.nn.Module:
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config_path = download_pretrained_from_hf(model_id, filename='open_clip_config.json', cache_dir=cache_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        pretrained_cfg = config['preprocess_cfg']
        model_cfg = config['model_cfg']
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        pretrained_cfg = {}
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        if custom_text:
            if is_hf_model:
                model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf
            if "coca" in model_name:
                model = CoCa(**model_cfg, **model_kwargs, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, **model_kwargs, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, **model_kwargs, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from open_clip.transformer import LayerNormFp32
                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = pretrained_path
# 确保路径存在
            if os.path.exists(checkpoint_path):
                print(f'Checkpoint path: {checkpoint_path}')
            else:
                print('Checkpoint file not found.')
            #pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
            #     checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                pass
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif has_hf_hub_prefix:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model

    
    
def inference(args) -> Tuple[List[int], List[int], List[str]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(args.model_name, pretrained_path=args.model_path, pretrained='laion2b_s34b_b79k')
    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k',device=device)
    image_mean =getattr(model.visual, 'image_mean', None)
    image_std = getattr(model.visual, 'image_std', None)
    preprocess = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    dataset_path = args.dataset_path # 替换为数据集路径
    all_labels = []   #所有输入数据的标签
    unique_labels = [] #数据集中所有类别
#处理输入数据集
    class CustomDataset(Dataset):
      def __init__(self, dataset_path, preprocess_fn):
        self.processed_dataset = []
        self.label_idx_counter = 0

        for label in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label)
            if label == '.DS_Store':
                continue
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)

                    image = preprocess_fn(Image.open(image_path)).to(device)
                    all_labels.append(label)
                    processed_sample = {
                        "image": image,
                        "label": self.label_idx_counter
                    }

                    self.processed_dataset.append(processed_sample)
                self.label_idx_counter += 1
      def __len__(self):
        return len(self.processed_dataset)

      def __getitem__(self, idx):
        return {"image": self.processed_dataset[idx]["image"], "label": self.processed_dataset[idx]["label"]}

#将输入数据处理为Dataset类型 
    custom_dataset = CustomDataset(dataset_path, preprocess)
#将Dataset转为Dataloader并设置batch_size
    test_data = DataLoader(custom_dataset,batch_size=128,shuffle=False)
#将所有输入图片的label存放在all_labels再去重
    unique_labels =  list(OrderedDict.fromkeys(all_labels))
#去重后作为text_prompt
    text_prompt = tokenizer(unique_labels).to(device)
# 遍历batch进行预测
    true_labels = []
    predicted_labels = []
    for batch in test_data:
      batch_images = batch["image"].to(device)
      batch_labels = batch["label"].to(device)
      with torch.no_grad(), torch.cuda.amp.autocast():
        batch_image_features = model.encode_image(batch_images)
        batch_text_features = model.encode_text(text_prompt)  
        batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
        batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)

        batch_text_probs = (100.0 * batch_image_features @ batch_text_features.T).softmax(dim=-1)
        if batch_text_probs.size(1) > 0:
            batch_predicted_labels = batch_text_probs.argmax(dim=-1)
        else:
            # 处理空维度的情况
            batch_predicted_labels = torch.zeros(batch_text_probs.size(0), dtype=torch.long)

        true_labels.extend(batch_labels.cpu().numpy())
        predicted_labels.extend(batch_predicted_labels.cpu().numpy())

    return true_labels, predicted_labels, unique_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--true_labels_output_path", type=str, default=None, help="Path to the true_labels_output")
    parser.add_argument("--predicted_labels_output_path", type=str, default=None, help="Path to the predicted_labels_output")
    parser.add_argument("--unique_labels_output_path", type=str, default=None, help="Path to the unique_labels_output")
    args = parser.parse_args()
     # 调用 inference 函数获取结果
    true_labels, predicted_labels, unique_labels = inference(args)
    # 将结果保存为三个 CSV 文件
    true_labels_df = pd.DataFrame({"true_labels": true_labels})
    predicted_labels_df = pd.DataFrame({"predicted_labels": predicted_labels})
    unique_labels_df = pd.DataFrame({"unique_labels": unique_labels})

    true_labels_csv_path = args.true_labels_output_path  # 指定保存路径和文件名
    predicted_labels_csv_path = args.predicted_labels_output_path # 指定保存路径和文件名
    unique_labels_csv_path = args.unique_labels_output_path  # 指定保存路径和文件名

    true_labels_df.to_csv(true_labels_csv_path, index=False)
    predicted_labels_df.to_csv(predicted_labels_csv_path, index=False)
    unique_labels_df.to_csv(unique_labels_csv_path, index=False)

    print(f"True labels saved to {true_labels_csv_path}")
    print(f"Predicted labels saved to {predicted_labels_csv_path}")
    print(f"Unique labels saved to {unique_labels_csv_path}")
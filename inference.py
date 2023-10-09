import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip
from sklearn.metrics import accuracy_score,confusion_matrix
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Optional, Union, Tuple
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
):
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = ""#指定模型权重文件路径
model = create_model("ViT-B-32", pretrained_path=model_path)
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k',device=device)
image_mean =getattr(model.visual, 'image_mean', None)
image_std = getattr(model.visual, 'image_std', None)
preprocess = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )
tokenizer = open_clip.get_tokenizer('ViT-B-32')
dataset_path = ""  # 替换为数据集路径
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

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# # 构建DataFrame
# conf_matrix_df = pd.DataFrame(conf_matrix,index=unique_labels,columns=unique_labels)

# # 保存为CSV文件
# csv_filename = "confusion_matrix_RESISC45.csv"
# conf_matrix_df.to_csv(csv_filename)
# print(f"Confusion Matrix saved to {csv_filename}")

#计算recall,precision,F1
class_metrics = []
for idx, label in enumerate(unique_labels):
    true_positive = conf_matrix[idx, idx]
    false_positive = conf_matrix[:, idx].sum() - true_positive
    false_negative = conf_matrix[idx, :].sum() - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    class_metrics.append({
        "Label": label,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

# 计算所有类别的平均F1和top-1准确率
average_f1 = np.mean([metric["F1"] for metric in class_metrics])
top1_accuracy = accuracy_score(true_labels, predicted_labels)

# 构建DataFrame
metrics_df = pd.DataFrame(class_metrics)
overall_metrics_df = pd.DataFrame({
    "Average_F1": [average_f1],
    "Top1_Accuracy": [top1_accuracy]
})

# 保存为CSV文件
metrics_filename = "class_metrics_RESISC45_remote.csv"
overall_metrics_filename = "overall_metrics_RESISC45_remote.csv"
metrics_df.to_csv(metrics_filename, index=False)
overall_metrics_df.to_csv(overall_metrics_filename, index=False)
print(f"Class Metrics saved to {metrics_filename}")
print(f"Overall Metrics saved to {overall_metrics_filename}")
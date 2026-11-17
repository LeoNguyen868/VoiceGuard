"""wav2vec 2.0 XLS-R 300M frontend on Hugging Face transformers instead of fairseq.

The authors' ITW checkpoint stores the fine-tuned XLS-R weights in fairseq naming
(prefix ``ssl_model.model.``). ``map_fairseq_to_hf`` renames them to the
transformers Wav2Vec2Model naming so the same weights run without fairseq.
The model output used by Nes2Net-X is the last encoder layer after the final
layer norm, which is ``last_hidden_state`` in transformers and
``model(x, mask=False, features_only=True)['x']`` in fairseq.
"""
import re

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

# Architecture of facebook/wav2vec2-xls-r-300m (fairseq xlsr2_300m.pt).
# Dropout and masking are disabled: the frontend is only used for inference.
XLSR_300M = dict(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
    hidden_act="gelu", layer_norm_eps=1e-5,
    conv_dim=(512,) * 7, conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_bias=True, feat_extract_norm="layer", feat_extract_activation="gelu",
    do_stable_layer_norm=True, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
    hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
    final_dropout=0.0, layerdrop=0.0, mask_time_prob=0.0, mask_feature_prob=0.0,
    apply_spec_augment=False,
)

# Same structure with tiny sizes (about 2 M parameters): tests on weak machines only.
TINY = dict(XLSR_300M, hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=256, conv_dim=(64,) * 7)

FAIRSEQ_PREFIX = "ssl_model.model."
# fairseq keys that the features_only forward pass never uses.
FAIRSEQ_UNUSED = ("quantizer.", "project_q.", "final_proj.", "mask_emb", "label_embs_concat",
                  "target_glu.")

_LAYER_RULES = [
    (r"self_attn\.(k|v|q|out)_proj\.", r"attention.\1_proj."),
    (r"self_attn_layer_norm\.", "layer_norm."),
    (r"fc1\.", "feed_forward.intermediate_dense."),
    (r"fc2\.", "feed_forward.output_dense."),
    (r"final_layer_norm\.", "final_layer_norm."),
]


class XLSRFrontend(nn.Module):
    def __init__(self, tiny=False, num_layers=None):
        super().__init__()
        cfg = dict(TINY if tiny else XLSR_300M)
        if num_layers:
            # Encoder truncation: keep the CNN encoder and the first num_layers Transformer layers.
            if not 1 <= num_layers <= cfg["num_hidden_layers"]:
                raise ValueError(f"num_layers must be in 1..{cfg['num_hidden_layers']}")
            cfg["num_hidden_layers"] = num_layers
        self.out_dim = cfg["hidden_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.model = Wav2Vec2Model(Wav2Vec2Config(**cfg))

    def forward(self, wav):  # wav: B x L, raw 16 kHz waveform, not normalised
        return self.model(wav).last_hidden_state  # B x T x 1024


def _pos_conv_names(hf_keys):
    """transformers uses either old weight_norm (weight_g/weight_v) or parametrizations."""
    if "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in hf_keys:
        return ("encoder.pos_conv_embed.conv.parametrizations.weight.original0",
                "encoder.pos_conv_embed.conv.parametrizations.weight.original1")
    return "encoder.pos_conv_embed.conv.weight_g", "encoder.pos_conv_embed.conv.weight_v"


def map_fairseq_key(key, hf_keys):
    """Return the transformers name of one fairseq wav2vec2 key, or None if unused."""
    if key.startswith(FAIRSEQ_UNUSED):
        return None
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", key)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.conv.{m[2]}"
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", key)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.layer_norm.{m[2]}"
    if key.startswith("layer_norm."):
        return "feature_projection." + key
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.")
    g_name, v_name = _pos_conv_names(hf_keys)
    if key == "encoder.pos_conv.0.weight_g":
        return g_name
    if key == "encoder.pos_conv.0.weight_v":
        return v_name
    if key == "encoder.pos_conv.0.bias":
        return "encoder.pos_conv_embed.conv.bias"
    if key.startswith("encoder.layer_norm."):
        return key
    m = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", key)
    if m:
        rest = m[2]
        for pat, rep in _LAYER_RULES:
            if re.match(pat, rest):
                return f"encoder.layers.{m[1]}." + re.sub(pat, rep, rest, count=1)
    raise KeyError(f"no mapping for fairseq key: {key}")


def map_fairseq_to_hf(fairseq_sd, hf_keys):
    """Map a fairseq wav2vec2 state dict (without prefix) to transformers names.

    Returns (mapped_state_dict, skipped_keys). Raises KeyError on an unknown key,
    so that a renamed or missing tensor can never be dropped silently.
    """
    mapped, skipped = {}, []
    for k, v in fairseq_sd.items():
        new = map_fairseq_key(k, hf_keys)
        if new is None:
            skipped.append(k)
            continue
        if new in mapped:
            raise KeyError(f"duplicate target key {new}")
        mapped[new] = v
    return mapped, skipped


@torch.no_grad()
def build_random_frontend():
    """Frontend with random weights: valid for latency, parameter and MAC counts only."""
    return XLSRFrontend().eval()

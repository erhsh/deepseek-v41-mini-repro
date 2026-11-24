"""Run a single-rank V4.1 OfflineInference request with synthetic MX weights.

Requires the matching V4.1 operators and a sourced CANN environment (see
README.md). Use --dspark to run proposal and verification. The EP1 decode
path reuses the recipe's ordinary expert routing and shared FFN.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path
from unittest.mock import patch

RECIPES_DIR = Path(__file__).resolve().parent.parent / '3rdparty' / 'cann-recipes-infer'
if not (RECIPES_DIR / 'executor' / 'offline' / 'offline_inference.py').is_file():
    raise SystemExit(
        f'Missing cann-recipes-infer submodule at {RECIPES_DIR}. '
        'From the repository root, run: git submodule update --init --recursive'
    )
sys.path.insert(0, str(RECIPES_DIR))
# These upstream directories are namespace packages. Limit their search paths
# to this checkout even if the environment also provides a package named models.
for name in ('executor', 'models', 'module'):
    spec = PathFinder.find_spec(name, [str(RECIPES_DIR)])
    if spec is None:
        raise ImportError(f'Missing upstream package {name} in {RECIPES_DIR}')
    spec.submodule_search_locations = list(spec.submodule_search_locations)
    package = module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)

import torch
import torch.distributed as dist
import torch_npu
import yaml
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from executor.core.config import InferenceConfig
from executor.core.forward_data_info import SamplingParams
from executor.core.kv_cache import cache_utils
from executor.offline.offline_inference import OfflineInference
from models.deepseek_v4_1.models.modeling_deepseek import DeepseekV41ForCausalLM, DeepseekV41MoE
from models.deepseek_v4_1.models.modeling_deepseek_vision import DeepseekV41VisionModel
from models.deepseek_v4_1.utils.convert_config import main as convert_config
from models.deepseek_v4_1.utils.encoding_dsv4 import IMAGE_PLACEHOLDER


def make_fixture(path, source, recipe, dspark):
    raw = json.loads((source / 'config.json').read_text())
    raw['image_token_id'] = 1022
    raw['text_config'].update(
        vocab_size=1024, num_hidden_layers=6, n_routed_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=512, engram_layer_ids=[1], engram_vocab_size=257,
        engram_num_embeddings=[16384], engram_compressed_vocab_size=1024,
        kv_source_layer_ids=[1, 3], index_source_layer_ids=[1, 3, 4], candidate_source_layer_id=3,
        compress_ratios=[0, 2, 2, 1, 1, 1, 0, 0, 0], dspark_target_layer_ids=[4, 5],
        dspark_noise_token_id=1023, dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )
    raw['vision_config'].update(num_hidden_layers=1, min_pixels=42**2)
    (path / 'config.json').write_text(json.dumps(raw))
    convert_config(path, is_mx=True)
    vocab = {'<s>': 0, '</s>': 1, '<pad>': 2, '<unk>': 3}
    vocab.update({f'tok{i}': i for i in range(4, 1024) if i != 1022})
    vocab[IMAGE_PLACEHOLDER] = 1022
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token='<s>', eos_token='</s>', pad_token='<pad>',
        unk_token='<unk>', additional_special_tokens=[IMAGE_PLACEHOLDER],
        chat_template=(source / 'chat_template.jinja').read_text(),
    )
    assert hf.convert_tokens_to_ids(IMAGE_PLACEHOLDER) == 1022
    hf.save_pretrained(path)
    settings = yaml.safe_load(recipe.read_text())
    settings['model_config'].update(model_path=str(path), with_ckpt=False, force_eplb=True)
    params = settings['model_config']['custom_params']
    params.update(enable_mega_moe=False, enable_multi_streams=False, engram_tp_size=1,
                  enable_engram_offload=False, enable_ced_prefill_tail=True)
    params['kernel_config']['swiglu'] = 'custom'  # use the matching custom vendor ABI
    settings['parallel_config']['world_size'] = 1
    settings['scheduler_config'].update(batch_size=1, max_new_tokens=3, max_prefill_tokens=256)
    settings['data_config'].update(dataset='default_multimodal', input_truncated_len=256)
    if dspark:
        settings['speculative_config'] = dict(method='dspark', num_speculative_tokens=2,
                                              markov_lmhead_tp_size=1, confidence_threshold=0.0)
        settings['scheduler_config']['max_new_tokens'] = 5
    return InferenceConfig.from_dict(settings, global_rank=0, local_rank=0)


@torch.no_grad()
def initialize(model):
    for name, param in model.named_parameters():
        if param.dtype in (torch.int8, torch.uint8):
            if name.endswith('scale'):
                param.fill_(120)  # E8M0 code for 2**-7
            else:
                param.view(torch.uint8).random_(0, 256)
        elif param.dtype == torch.float8_e4m3fn:
            param.copy_(torch.randn(param.shape, device=param.device, dtype=torch.bfloat16).clamp(-4, 4))
        elif 'norm.weight' in name or name.endswith(('q_weight', 'k_weight')):
            param.fill_(1)
        elif param.is_floating_point():
            param.normal_(0, 0.01)


def single_rank_moe(self, x, topk_ids, topk_weight, shared_expert_stream, moe_events, engram_resource_events=None):
    assert self.moe_ep_size == 1 and not self.enable_mega_moe
    events = moe_events.get('shared_expert', [])
    shared = self.forward_shared_expert(x, shared_expert_stream, events) if self.n_shared_experts else None
    return self.moe_infer_double_routing(x, topk_ids, topk_weight, shared, events, 1)


def run(args, tmp):
    torch.manual_seed(41)
    torch.npu.set_device(0)
    cfg = make_fixture(tmp, args.source, args.recipe, args.dspark)
    dist.init_process_group('hccl', init_method=(tmp / 'rank0').as_uri(), rank=0, world_size=1)
    calls = Counter()
    head = DeepseekV41ForCausalLM.forward_lm_head
    vision = DeepseekV41VisionModel.forward

    allocate = cache_utils._alloc_aligned_hbm_tensor

    def allocate_cache(device, dtype, numel, shape):
        cache = allocate(device, dtype, numel, shape)
        # Warm-up uses page zero as a null page. Give its packed KV and scales
        # deterministic initial bytes, including before the first CED tail.
        cache[0].zero_()
        return cache

    def checked_logits(model, outputs, kv_len, is_prefill=True, attn_metadata=None):
        logits = head(model, outputs, kv_len, is_prefill, attn_metadata)
        assert logits.device.type == 'npu' and logits.shape[-1] == 1024
        assert logits.isfinite().all().item(), 'nonfinite logits'
        assert logits.abs().max().item() > 0, 'all-zero logits'
        calls['prefill' if is_prefill else 'decode'] += 1
        return logits

    def checked_vision(model, *args, **kwargs):
        result = vision(model, *args, **kwargs)
        if any(value is not None and value.numel() for value in result):
            calls['vision'] += 1
        return result

    try:
        with patch.object(cache_utils, '_alloc_aligned_hbm_tensor', allocate_cache), \
             patch('executor.model_loader.dummy_loader.initialize_dummy_weights', initialize), \
             patch.object(DeepseekV41MoE, 'moe_infer_dispatch_combine', single_rank_moe), \
             patch.object(DeepseekV41ForCausalLM, 'forward_lm_head', checked_logits), \
             patch.object(DeepseekV41VisionModel, 'forward', checked_vision):
            engine = OfflineInference(cfg)
            calls.clear()  # count only the submitted request
            torch.manual_seed(9)
            pixels = torch.randint(0, 256, (42, 42, 3), dtype=torch.uint8)
            sample = tmp / 'input.png'
            Image.fromarray(pixels.numpy()).save(sample)
            text = ' '.join(f'tok{n}' for n in range(4, 143))
            prompt = [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': str(sample)}},
                {'type': 'text', 'text': text},
            ]}]
            outputs, stats, _ = engine.generate([prompt], SamplingParams(temperature=0))
            torch.npu.synchronize()
            assert len(outputs) == 1 and calls['vision'] > 0 and calls['prefill'] == 1 and calls['decode'] >= 1
            spec_steps = stats['spec_num_forward_ct'] if args.dspark else []
            assert not args.dspark or sum(spec_steps) > 0
    finally:
        dist.destroy_process_group()
    print(json.dumps({'result': 'pass', 'calls': calls, 'finish': outputs[0].finish_reason,
                      'output': outputs[0].output_text, 'speculative_forwards': spec_steps}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).resolve().parent.parent / 'build' / 'model'
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('MODEL_SOURCE', default_source)),
                        help='Directory containing the official config.json and chat_template.jinja.')
    model_config_dir = RECIPES_DIR / 'models' / 'deepseek_v4_1' / 'config'
    parser.add_argument('--recipe', type=Path,
                        default=model_config_dir.joinpath('deepseek_v4_1_flash_rank_8_8ep_multimodal.yaml'))
    parser.add_argument('--dspark', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='dsv41_fixture_') as work:
        run(args, Path(work))

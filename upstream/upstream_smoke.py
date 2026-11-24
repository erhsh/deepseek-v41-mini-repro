"""Run DeepSeek's released inference/model.py using a tiny multimodal config.

The upstream CUDA TileLang kernels are evaluated as PyTorch expressions. The
checkpoint weights and the repository's optimized CANN executor are not used.
"""
import argparse
import functools
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


FP8 = torch.float8_e4m3fn
FP4 = torch.float4_e2m1fn_x2
E8 = torch.float8_e8m0fnu
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def act_quant(x, block_size=32, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    blocks = x.float().unflatten(-1, (-1, block_size))
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    if scale_fmt is not None:
        scale = scale.log2().ceil().exp2()
    encoded = (blocks / scale).clamp(-448.0, 448.0).to(FP8)
    if inplace:
        return x.copy_((encoded.float() * scale).reshape_as(x))
    # Internal interface uses the mathematical scale directly for the GEMMs.
    return encoded.reshape_as(x), scale.squeeze(-1)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=E8):
    blocks = x.float().unflatten(-1, (-1, block_size))
    scale = blocks.abs().amax(-1, keepdim=True) / 6.0
    if scale_dtype == FP8:
        scale = scale.clamp_min(2**-9).to(FP8).float()
    else:
        scale = scale.clamp_min(2**-126).log2().ceil().exp2()
    values = blocks / scale
    levels = values.new_tensor(FP4_VALUES)
    boundaries = (levels[1:] + levels[:-1]) * 0.5
    absolute = values.abs().unsqueeze(-1)
    # Exact midpoints round to the representable value with an even mantissa.
    odd = torch.arange(7, device=x.device) % 2 == 1
    codes = ((absolute > boundaries) | ((absolute == boundaries) & odd)).sum(-1)
    if inplace:
        return x.copy_((levels[codes] * values.sign() * scale).reshape_as(x))
    codes = (codes + 8 * (values < 0)).reshape_as(x).to(torch.uint8)
    return (codes[..., ::2] | (codes[..., 1::2] << 4)).view(FP4), scale.squeeze(-1)


def decode_scale(x):
    return (x.view(torch.uint8).float() - 127).exp2() if x.dtype == E8 else x.float()


def engram_embedding_npu(layer, indices):
    # NPU embedding accepts uint8. Gather each FP8 row and its scale as raw
    # bytes, then restore their original values and dequantize on the NPU.
    assert layer.vocab_start_idx == 0 and layer.part_num_embeddings == layer.num_embeddings
    rows = F.embedding(indices, layer.weight.view(torch.uint8)).view(FP8).float()
    scale = F.embedding(indices, layer.scale.view(torch.uint8)).view(E8)
    rows = rows.unflatten(-1, (-1, layer.block_size)) * decode_scale(scale).unsqueeze(-1)
    return rows.flatten(-2).bfloat16()


class NpuCopySign(TorchFunctionMode):
    """Express the float32 Engram signed square root without a CPU fallback."""
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.copysign and args[0].device.type == 'npu':
            magnitude, sign = args
            assert sign.dtype == torch.float32
            return torch.where(sign.view(torch.int32) < 0, -magnitude.abs(), magnitude.abs())
        return func(*args, **(kwargs or {}))


def fp8_gemm(a, a_s, b, b_s, scale_dtype=None, block_size=32):
    weights = decode_scale(b_s).repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    activations = a.float() * a_s.repeat_interleave(block_size, dim=-1)
    return F.linear(activations, b.float() * weights[:b.shape[0], :b.shape[1]]).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=None, act_block_size=32):
    packed = b.view(torch.uint8).int()
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    levels = a_s.new_tensor(FP4_VALUES)
    weight = levels[(codes & 7).long()] * (1 - 2 * (codes >> 3))
    weight *= decode_scale(b_s).repeat_interleave(32, dim=-1)
    return F.linear(a.float() * a_s.repeat_interleave(act_block_size, dim=-1), weight).to(torch.get_default_dtype())


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    sizes = (hc_mult, hc_mult, hc_mult**2)
    pre, post, comb = [v * scale + base for v, scale, base in
                       zip(mixes.split(sizes, -1), hc_scale.unbind(), hc_base.split(sizes))]
    comb = comb.unflatten(-1, (hc_mult, hc_mult)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.sigmoid() + eps, 2 * post.sigmoid(), comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    selected = kv[batch, topk_idxs.clamp_min(0).long()].float()
    scores = torch.einsum('bshd,bskd->bshk', q.float(), selected) * softmax_scale
    scores = scores.masked_fill(topk_idxs[:, :, None, :] < 0, -torch.inf)
    sink = attn_sink.reshape(1, 1, -1, 1).expand(*scores.shape[:-1], 1)
    weights = torch.cat((scores, sink), dim=-1).softmax(-1)[..., :-1]
    return torch.einsum('bshk,bskd->bshd', weights, selected).to(q.dtype)


def load_upstream(source, counts):
    if not (source / 'model.py').is_file():
        raise FileNotFoundError(f'{source}/model.py')
    sys.path.insert(0, str(source.resolve()))
    kernel = ModuleType('kernel')
    def counted(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            result = fn(*args, **kwargs)
            counts[fn.__name__] += 1
            return result
        return run
    for fn in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, hc_split_sinkhorn, sparse_attn):
        setattr(kernel, fn.__name__, counted(fn))
    sys.modules['kernel'] = kernel
    model = importlib.import_module('model')
    return model


def make_model(model):
    from tokenizers import Tokenizer, models

    vocab_size = 128
    vocabulary = {f'tok{i}': i for i in range(vocab_size)}
    class TestTokenizer:
        backend_tokenizer = Tokenizer(models.WordLevel(vocab=vocabulary))
        def __len__(self):
            return len(vocabulary)
    args = model.ModelArgs(
        max_batch_size=1, max_seq_len=32, temperature=0, vocab_size=vocab_size,
        dim=64, moe_inter_dim=64, n_layers=6, n_mtp_layers=3, n_heads=2,
        n_routed_experts=4, n_activated_experts=2, swiglu_limit=10, q_lora_rank=64,
        head_dim=64, rope_head_dim=16, o_groups=2, o_lora_rank=32, window_size=8,
        compress_ratios=(0, 2, 2, 1, 1, 1, 0, 0, 0), kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 4), index_n_heads=2, index_head_dim=32, index_topk=4,
        candidate_source_layer=3, candidate_topk_blocks=2, candidate_block_size=2,
        hc_sinkhorn_iters=3, original_seq_len=16, engram_layer_ids=(1,),
        engram_num_embeddings=(256,), engram_max_ngram_size=4, engram_vocab_size=17,
        engram_n_heads=2, engram_head_dim=32, engram_compressed_vocab_size=vocab_size,
        vision_n_layers=1, vision_dim=32, vision_n_heads=2, vision_inter_dim=64,
        vision_patch_size=2, vision_downsample_ratio=3, image_token_id=126,
        dspark_block_size=3, dspark_noise_token_id=127, dspark_target_layer_ids=(4, 5),
        dspark_markov_rank=32,
    )
    return model.Transformer(args, TestTokenizer()).eval(), args


def initialize(model):
    # Initialize on CPU because FP4/E8M0 are storage dtypes. Weight.scale is
    # a non-registered alias attached to each quantized Linear by upstream.
    for name, param in model.named_parameters():
        if param.dtype == E8:
            param.view(torch.uint8).fill_(121)  # 2^-6
        elif param.dtype == FP4:
            param.view(torch.uint8).random_(0, 256)
        elif param.dtype == FP8:
            param.copy_(torch.randn(param.shape, dtype=torch.float32).clamp(-4, 4).to(FP8))
        elif name.endswith('norm.weight') or name.endswith(('q_weight', 'k_weight')):
            param.fill_(1)
        else:
            param.normal_(0, 0.06)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).resolve().parent.parent / 'build' / 'inference'
    parser.add_argument('--source', type=Path, default=default_source,
                        help='Directory containing the four official inference source files.')
    parser.add_argument('--device', default='cpu', help='Run on CPU (default) or an Ascend device such as npu:0.')
    options = parser.parse_args()
    if options.device.startswith('npu'):
        import torch_npu  # registers the backend
        torch.npu.set_device(options.device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(41)
    counts = Counter()
    model = load_upstream(options.source, counts)
    if options.device.startswith('npu'):
        model.ParallelEngramEmbedding.forward = engram_embedding_npu
    with torch.inference_mode(), NpuCopySign():
        net, args = make_model(model)
        initialize(net)
        net.to(options.device)
        for layer in net.modules():
            if isinstance(layer, model.Linear) and layer.scale is not None:
                layer.weight.scale = layer.scale
        torch.set_default_device(options.device)
        def record(name):
            def hook(module, inputs, output):
                counts[name] += 1
            return hook
        for name, layer in net.named_modules():
            if isinstance(layer, (model.Block, model.Indexer, model.Compressor, model.Engram,
                                  model.ViT, model.Aligner, model.DSparkMarkovHead,
                                  model.DSparkConfidenceHead, model.Expert)):
                layer.register_forward_hook(record(name))
        from image_processor import IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END, TEXT, ImageInput
        types = torch.tensor([IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END])
        # Use identical synthetic pixels for CPU and NPU runs.
        patches = torch.randn(9, 3, args.vision_patch_size, args.vision_patch_size, device='cpu').to(options.device)
        images = [[ImageInput(1, patches, 3, 3, types)]]
        length = 13  # crosses the ring boundary and leaves one unpaired C2A token
        tokens = torch.arange(3, 3 + length).reshape(1, -1)
        tokens[:, 1:5] = args.image_token_id
        token_types = torch.full_like(tokens, TEXT)
        token_types[:, 1:5] = types
        summaries = []
        for pos in (0, length, length + 1):
            if pos == 0:
                token, logits, hidden = net(tokens, images=images, token_types=token_types)
            else:
                token, logits, hidden = net(token[:, None], pos)
            assert logits.shape == (1, args.vocab_size) and logits.device == tokens.device
            assert torch.isfinite(logits).all().item() and logits.abs().sum().item() > 0
            draft = net.forward_spec(token, hidden, pos)
            item = {'position': pos, 'logits_shape': list(logits.shape), 'next_token': token.item()}
            if pos:
                ids, draft_logits, confidence = draft
                assert ids.shape == (1, args.dspark_block_size + 1)
                assert draft_logits.shape == (1, args.dspark_block_size, args.vocab_size)
                assert confidence.shape == (1, args.dspark_block_size)
                assert torch.isfinite(draft_logits).all().item() and torch.isfinite(confidence).all().item()
                item['draft_ids'] = ids.tolist()
            else:
                assert draft is None
            summaries.append(item)
            print(json.dumps(item), flush=True)
        assert counts['vision'] == counts['aligner'] == 1
        assert all(counts[f'layers.{i}'] == 3 for i in range(args.n_layers))
        assert all(counts[f'layers.{i}.attn.indexer'] == 3 for i in args.index_source_layers)
        assert all(counts[f'layers.{i}.attn.compressor'] == 3 for i in args.kv_source_layers)
        assert counts['layers.1.engram'] == 3 and counts['mtp.2.confidence_head'] == 2
        assert all(counts[f'mtp.{i}'] == 3 for i in range(args.n_mtp_layers))
        assert all(counts[name] > 0 for name in ('act_quant', 'fp4_act_quant', 'fp8_gemm',
                                                  'fp4_gemm', 'sparse_attn', 'hc_split_sinkhorn'))
        if options.device.startswith('npu'):
            torch.npu.synchronize()
        print(json.dumps({'device': str(tokens.device), 'source': str(options.source), 'stages': summaries,
                          'calls': counts, 'result': 'pass'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

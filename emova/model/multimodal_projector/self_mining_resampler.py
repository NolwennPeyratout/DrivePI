import torch
from torch import nn
from torch import einsum

# Register the projector and inherit the shared downsampling implementation.
from emova.model.multimodal_projector.builder import MM_PROJECTOR, BaseDownsamplerMMProjector

from einops import rearrange, repeat
from einops_exts import rearrange_many


# Apply multi-head cross-attention to a query and its local feature context.
class CrossAttentionDropOutProj(nn.Module):
    # dim is the BEV feature width; each attention head has dim_head channels.
    def __init__(self, *, dim, dim_head=64, heads=8, kv_concat_x=False):
        super().__init__()
        # Scale query vectors to keep attention logits numerically well behaved.
        self.scale = dim_head ** -0.5
        # Store the head count for the projections and tensor reshapes.
        self.heads = heads
        # Total width of the concatenated attention heads.
        inner_dim = dim_head * heads

        # Normalize the query/media features and the context features separately.
        self.norm_media = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        # If enabled, append the query features to the context used for keys/values.
        self.kv_concat_x = kv_concat_x

        # Project query features into the concatenated multi-head representation.
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        # Produce key and value projections together; the result is split in forward().
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

    # x is the query sequence; context supplies the features attended to by each query.
    def forward(self, x, context):
        # Normalize both input sequences before projecting them.
        x = self.norm_media(x)
        context = self.norm_context(context)

        # Here h is the attention-head count; it is unrelated to the spatial h below.
        h = self.heads

        # Optionally include the query sequence alongside the context as key/value input.
        if self.kv_concat_x:
            kv_input = torch.cat([context, x], dim=1)
        else:
            kv_input = context

        # Project the queries and split the joint key/value projection into two tensors.
        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # Change [batch, sequence, heads * head_dim] into [batch, heads, sequence, head_dim].
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        # Apply the standard inverse-square-root head-dimension scaling to queries.
        q = q * self.scale

        # Compute scaled dot-product attention logits for every query/key pair.
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # Subtract the row maximum for numerical stability without changing softmax.
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        # Normalize over keys so each query's attention weights sum to one.
        attn = sim.softmax(dim=-1)
        # Combine values using the attention weights.
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        # Merge the head and head-dimension axes back into the feature width.
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        # Return one output feature per query position.
        return out


# Make this implementation available through the project's MM_PROJECTOR registry.
@MM_PROJECTOR.register_module()
# Reduce spatial BEV tokens with the shared downsampling-projector interface.
class SelfMiningMMProjector(BaseDownsamplerMMProjector):
    # Build cross-attention with one 64-channel head for each 64 input channels.
    # kv_concat_x=True also appends the pooled query to the K/V context.
    def build_downsampler(self, mm_hidden_size, kv_concat_x=True, **kwargs):
        return CrossAttentionDropOutProj(dim=mm_hidden_size,
                                         heads=mm_hidden_size // 64,
                                         kv_concat_x=kv_concat_x)

    # x starts as flattened BEV features: [B, H*W, C].
    def forward_downsampler(self, x):
        # K*K is the downsampling rate. For a square H-by-W grid, h=H/K and w=W/K.
        # In this implementation the grid is square, so the same h is used for both axes.
        # IMPORTANT: this spatial h is not the patch size K or the attention-head count.
        h = int((x.size(1) // self.downsample_rate) ** 0.5)
        # p and k are the patch height and width; each equals K=sqrt(downsample_rate).
        # Split F_bev into local patches: [B, H*W, C] -> [B*h*w, K*K, C] = [B*N, K², C] = F_patch
        x = rearrange(x, 'b (h p w k) c -> (b h w) (p k) c',
                      h=h,
                      p=self.downsample_rate_per_side,
                      k=self.downsample_rate_per_side)

        # x is now F_patch. Its mean over K*K features is F_pool, the single query.
        # Attend that query to the patch features; the result has one vector per patch.
        x = self.downsampler(x.mean(1, keepdims=True), x) #downsampler of F_patch using the pooled query F_pool

        # Reassemble patch outputs into F_v's spatial sequence: [B, h*w, C].
        x = rearrange(x, '(b h w) 1 c -> b (h w) c', h=h, w=h)
        # Return the downsampled BEV tokens; the parent then maps C to the MLLM width.
        return x

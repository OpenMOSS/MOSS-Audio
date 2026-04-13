from typing import Optional, List, Union, Tuple, Any
import math
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils.auto_docstring import auto_docstring
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from src.configuration_moss_audio import MossAudioEncoderConfig, MossAudioConfig


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        max_timescale = 10000.0
        log_timescale_increment = math.log(max_timescale) / (embedding_dim // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(embedding_dim // 2).float()
        )
        self.register_buffer("inv_timescales", inv_timescales, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        scaled_time = torch.arange(
            seq_len, device=device, dtype=self.inv_timescales.dtype
        ).unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        sin_emb = torch.sin(scaled_time)
        cos_emb = torch.cos(scaled_time)
        pos_emb = torch.cat([sin_emb, cos_emb], dim=1)
        return pos_emb.unsqueeze(0)


class MossAudioEncoder(nn.Module):
    """Audio encoder with conv-stem downsampling and Whisper transformer layers."""

    def __init__(self, config: MossAudioEncoderConfig):
        super().__init__()
        self.config = config
        self.gelu = nn.GELU()

        self.conv1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        # 128 mel bins / 8 = 16 after 3 convs with stride=2
        self.stem_proj = nn.Linear(config.downsample_hidden_size * 16, config.d_model)
        self.embed_positions = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_proj = (
            nn.Linear(config.d_model, config.output_dim, bias=False)
            if config.output_dim != config.d_model
            else nn.Identity()
        )

        self._deepstack_indexes_set = set(config.deepstack_encoder_layer_indexes or [])

    def _compute_downsampled_length(self, lengths: torch.Tensor) -> torch.Tensor:
        def conv_out_len(L):
            return (L - 1) // 2 + 1

        l1 = conv_out_len(lengths)
        l2 = conv_out_len(l1)
        l3 = conv_out_len(l2)
        return l3

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        output_deepstack_hidden_states: bool = True,
    ):
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        if feature_lens is None:
            feature_lens = torch.full(
                (input_features.size(0),),
                input_features.size(-1),
                device=input_features.device,
                dtype=torch.long,
            )

        downsampled_lengths = self._compute_downsampled_length(feature_lens)

        # [B, n_mels, T] -> [B, 1, n_mels, T]
        x = input_features.unsqueeze(1)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))

        # [B, C, F, T] -> [B, T, C*F]
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.stem_proj(x)

        max_len = int(downsampled_lengths.max().item())
        if x.size(1) > max_len:
            x = x[:, :max_len, :]

        positions = self.embed_positions(x.shape[1], x.device)
        x = x + positions.to(x.dtype)

        padding_mask = (
            torch.arange(x.size(1), device=x.device)[None, :] >= downsampled_lengths[:, None]
        )
        attention_mask = (1.0 - (~padding_mask).to(dtype=x.dtype)) * torch.finfo(x.dtype).min
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        deepstack_states: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                x,
                attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )
            x = layer_outputs[0]
            if output_deepstack_hidden_states and layer_idx in self._deepstack_indexes_set:
                deepstack_states.append(x)

        x = self.layer_norm(x)
        x = self.out_proj(x)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            hidden_states=tuple(deepstack_states) if output_deepstack_hidden_states else None,
        )


class GatedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@auto_docstring
class MossAudioPreTrainedModel(PreTrainedModel):
    config_class = MossAudioConfig
    config: MossAudioConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Qwen3DecoderLayer}


class MossAudioModel(MossAudioPreTrainedModel, GenerationMixin):
    config_class = MossAudioConfig
    _tied_weights_keys: List[str] = []

    def __init__(self, config: MossAudioConfig):
        super().__init__(config)

        self.audio_encoder = MossAudioEncoder(config.audio_config)
        self.language_model = Qwen3Model(config.language_config)

        self.audio_adapter = GatedMLP(
            input_size=config.audio_config.output_dim,
            hidden_size=config.adapter_hidden_size,
            output_size=config.language_config.hidden_size,
        )

        deepstack_k = len(getattr(config.audio_config, "deepstack_encoder_layer_indexes", []) or [])
        if config.deepstack_num_inject_layers is not None:
            deepstack_k = min(deepstack_k, int(config.deepstack_num_inject_layers))
        self.deepstack_audio_merger_list = nn.ModuleList(
            [
                GatedMLP(
                    input_size=config.audio_config.output_dim,
                    hidden_size=config.adapter_hidden_size,
                    output_size=config.language_config.hidden_size,
                )
                for _ in range(deepstack_k)
            ]
        )

        self.vocab_size = config.language_config.vocab_size
        self.lm_head = nn.Linear(config.language_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_audio_features(self, input_features, feature_lens):
        audio_outputs = self.audio_encoder(
            input_features=input_features,
            feature_lens=feature_lens,
            output_deepstack_hidden_states=True,
        )
        deepstack = list(audio_outputs.hidden_states) if audio_outputs.hidden_states is not None else None
        return audio_outputs.last_hidden_state, deepstack

    def _apply_deepstack_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        audio_input_mask: torch.Tensor,
        deepstack_embeds: torch.Tensor,
    ) -> torch.Tensor:
        audio_input_mask = audio_input_mask.to(hidden_states.device)
        deepstack_embeds = deepstack_embeds.to(hidden_states.device, hidden_states.dtype)
        flat = deepstack_embeds.reshape(-1, deepstack_embeds.shape[-1])
        hs = hidden_states.clone()
        hs[audio_input_mask] = hs[audio_input_mask] + flat
        return hs

    def _register_llm_deepstack_hooks(
        self,
        audio_input_mask: torch.Tensor,
        deepstack_audio_embeds: List[torch.Tensor],
    ):
        if deepstack_audio_embeds is None or len(deepstack_audio_embeds) == 0:
            return []

        layers = getattr(self.language_model, "layers", None)
        if layers is None:
            raise RuntimeError("Qwen3Model does not expose `.layers`; cannot register DeepStack hooks.")

        num_inject = len(deepstack_audio_embeds)
        handles = []

        for layer_idx, layer in enumerate(layers):
            if layer_idx >= num_inject:
                break

            def _make_llm_hook(k: int):
                def _hook(_module, _inputs, _output):
                    if isinstance(_output, (tuple, list)):
                        hs = _output[0]
                        new_hs = self._apply_deepstack_to_hidden_states(
                            hs, audio_input_mask, deepstack_audio_embeds[k]
                        )
                        return (new_hs,) + tuple(_output[1:])
                    else:
                        return self._apply_deepstack_to_hidden_states(
                            _output, audio_input_mask, deepstack_audio_embeds[k]
                        )

                return _hook

            handles.append(layer.register_forward_hook(_make_llm_hook(layer_idx)))

        return handles

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        hook_handles = []
        if audio_data is not None:
            if audio_input_mask is None:
                raise ValueError("audio_input_mask is required when audio_data is provided.")

            audio_embeds, deepstack = self.get_audio_features(audio_data, audio_data_seqlens)
            audio_embeds = self.audio_adapter(audio_embeds)

            audio_token_count = int(audio_input_mask.to(torch.int32).sum().item())
            if audio_token_count != int(audio_embeds.shape[1]):
                raise ValueError(
                    f"Audio token count mismatch: audio_input_mask has {audio_token_count} audio tokens, "
                    f"but audio_embeds has length {int(audio_embeds.shape[1])}."
                )

            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds)

            if deepstack is not None and len(self.deepstack_audio_merger_list) > 0:
                deepstack_audio_embeds = []
                for i, x in enumerate(deepstack[: len(self.deepstack_audio_merger_list)]):
                    ds = self.deepstack_audio_merger_list[i](x)
                    if int(ds.shape[1]) != audio_token_count:
                        raise ValueError(
                            f"DeepStack audio seq_len mismatch at index {i}: "
                            f"expected {audio_token_count}, got {int(ds.shape[1])}."
                        )
                    deepstack_audio_embeds.append(ds)

                try:
                    hook_handles = self._register_llm_deepstack_hooks(audio_input_mask, deepstack_audio_embeds)
                except Exception:
                    for h in hook_handles:
                        h.remove()
                    raise

        try:
            outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            for h in hook_handles:
                h.remove()

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
            shift_logits = shift_logits.view(-1, self.config.language_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        if cache_position is not None and cache_position[0] > 0:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            audio_data = None
            audio_input_mask = None
            audio_data_seqlens = None
        else:
            audio_data = kwargs.get("audio_data", None)
            audio_input_mask = kwargs.get("audio_input_mask", None)
            audio_data_seqlens = kwargs.get("audio_data_seqlens", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "audio_data": audio_data,
                "audio_input_mask": audio_input_mask,
                "audio_data_seqlens": audio_data_seqlens,
            }
        )

        return model_inputs


__all__ = [
    "MossAudioEncoderConfig",
    "MossAudioConfig",
    "MossAudioModel",
]

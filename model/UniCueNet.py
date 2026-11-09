import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # uncomment if you need a HF mirror
import numpy as np
import math
import laion_clap


from utils import PositionalEncoding, PositionalEncodingPermute2D
from speechbrain.inference.speaker import SpeakerRecognition

EPS = 1e-8

# LAION-CLAP audio-text checkpoint. Downloaded automatically on first use from
# HuggingFace (`lukewys/laion_clap`). Set HF_ENDPOINT if you need a mirror.
CLAP_CKPT = os.path.join("pretrained_models", "630k-audioset-fusion-best.pt")


def _get_clap_ckpt():
    """Return the CLAP fusion checkpoint path, downloading it on first use."""
    if os.path.exists(CLAP_CKPT):
        return CLAP_CKPT
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(CLAP_CKPT) or ".", exist_ok=True)
    return hf_hub_download(
        repo_id="lukewys/laion_clap",
        filename="630k-audioset-fusion-best.pt",
        local_dir=os.path.dirname(CLAP_CKPT),
    )


class FiLM(nn.Module):
    def __init__(self, dim_in=512, hidden_dim=256):
        super(FiLM, self).__init__()
        
        # 1. 移除 ReLU 和 Dropout，只保留 Linear
        # 这样 gamma 可以是负数（反相），也可以是正数
        self.gamma = nn.Linear(dim_in, hidden_dim)
        self.beta = nn.Linear(dim_in, hidden_dim)

        # 2. 强制初始化 (Identity Initialization)
        self._init_weights()

    def _init_weights(self):
        # 让 gamma 初始值为 1，这样初始状态下：y = 1 * x + 0 = x (无损通过)
        nn.init.constant_(self.gamma.weight, 0) # 权重设为0
        nn.init.constant_(self.gamma.bias, 1)   # 偏置设为1 -> 输出恒为1
        
        # 让 beta 初始值为 0
        nn.init.constant_(self.beta.weight, 0)
        nn.init.constant_(self.beta.bias, 0)

    def forward(self, hidden_state, embed):
        # embed: [B, D]
        # hidden_state: [B, C, T]
        
        gamma = self.gamma(embed).unsqueeze(-1) # [B, C, 1]
        beta = self.beta(embed).unsqueeze(-1)   # [B, C, 1]
        
        return gamma * hidden_state + beta

sector_mapping = {
    0: (-22.5, 22.5, "Front"),
    1: (-67.5, -22.5, "Right Front"),
    2: (-112.5, -67.5, "Right"),
    3: (-157.5, -112.5, "Right Rear"),
    4: (157.5, 202.5, "Rear"),
    5: (112.5, 157.5, "Left Rear"),
    6: (67.5, 112.5, "Left"),
    7: (22.5, 67.5, "Left Front"),
    8: (0.5, 2, "near"),
    9: (2, 3.5, "middle"),
    10: (3.5, 4.2, "far"),
}

spk_mapping = {
    1: "female",
    2: "male",
    3: "low",
    4: "loud",
}

mic_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

class TimeDomainIPD(nn.Module):
    def __init__(self, kernel_size, num_filters):
        super(TimeDomainIPD, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        # 可学习窗函数（初始为 hamming 窗）
        window = torch.hamming_window(kernel_size)
        # self.window = nn.Parameter(window, requires_grad=True)
        self.register_buffer("window", window)
        # 构造余弦和正弦表
        n = torch.arange(kernel_size).unsqueeze(1)
        k = torch.arange(num_filters).unsqueeze(0)
        w_cos = torch.cos(2 * math.pi * n * k / kernel_size)  # [T, K]
        w_sin = torch.sin(2 * math.pi * n * k / kernel_size)  # [T, K]

        self.register_buffer("cos_table", w_cos.T)  # [K, T]
        self.register_buffer("sin_table", w_sin.T)  # [K, T]

    def forward(self, mixture):
        # print(mixture.shape)
        B, T, C = mixture.shape
        mixture = mixture.permute(0, 2, 1)  # => [B, C, T]

        kre = (self.window[None, :] * self.cos_table).unsqueeze(1)  # [K, 1, T]
        kim = (self.window[None, :] * self.sin_table).unsqueeze(1)

        ipd_features = []

        for (p1, p2) in mic_pairs:
            y1 = mixture[:, p1, :].unsqueeze(1)  # [B, 1, T]
            y2 = mixture[:, p2, :].unsqueeze(1)

            real_y1 = F.conv1d(y1, kre, stride=self.kernel_size // 2)
            imag_y1 = F.conv1d(y1, kim, stride=self.kernel_size // 2)
            real_y2 = F.conv1d(y2, kre, stride=self.kernel_size // 2)
            imag_y2 = F.conv1d(y2, kim, stride=self.kernel_size // 2)

            ipd1 = torch.atan2(imag_y1, real_y1)  # [B, K, T']
            ipd2 = torch.atan2(imag_y2, real_y2)
            ipd = ipd1 - ipd2
            ipd_features.append(ipd)

        ipd_features = torch.stack(ipd_features, dim=1)  # [B, num_pairs, K, T']
        return ipd_features



class TPDLayer(nn.Module):
    def __init__(self,
                 mic_pairs=[(0, 1), (0, 2), (0,3),(1,2),(1,3),(2, 3)]  ,
                 radius=0.05,
                 fs=16000,
                 v=343,
                 n_mics=4):
        super().__init__()
        self.fs = fs
        self.freqs = torch.linspace(0, fs // 2, 256)
        self.v = v
        self.mic_pairs = mic_pairs
        self.radius = radius
        self.n_mics = n_mics
        self.room_sz = [6, 6, 3]
        self.circle_radius=0.05
        # 预计算并缓存麦克风位置
        self.register_buffer("mic_positions", self._get_mic_positions())  # [4, 3]
        # 预计算并缓存每对麦克风间的相对向量 Δ(p)
        self.register_buffer("delta_vecs", self._get_delta_vectors())  # [P, 3]

    def _get_mic_positions(self):
        mic_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        mic_positions = np.array([
            [self.room_sz[0] / 2 + self.circle_radius * np.cos(a),
             self.room_sz[1] / 2 + self.circle_radius * np.sin(a),
             1.7] for a in mic_angles
        ])
        return torch.tensor(mic_positions, dtype=torch.float32)

    def _get_delta_vectors(self):
        """计算所有 mic 对之间的 Δ(p) 向量"""
        delta_list = []
        for (p1, p2) in self.mic_pairs:
            delta = self.mic_positions[p1] - self.mic_positions[p2]  # [3]
            delta_list.append(delta)
        return torch.stack(delta_list, dim=0)  # [P, 3]
   
    def forward(self, text):
        device = self.delta_vecs.device
        freqs = self.freqs.to(device)
        z1 = torch.tensor([
            float(sector_mapping[text[i].item()][0])
            for i in range(text.size(0))
        ], dtype=torch.float32).unsqueeze(1)  # [B, 1]

        z2 = torch.tensor([
            float(sector_mapping[text[i].item()][1])
            for i in range(text.size(0))
        ], dtype=torch.float32).unsqueeze(1)  # [B, 1]
        t1 = torch.cat([z1, z2], dim=1).to(device)

        phi = torch.zeros((t1.size(0), 2))
        phi[:, 0] = -90
        phi[:, 1] = 90
        t2=phi.to(device)
        N=8
        theta_samples = torch.linspace(0, 1, N, device=device).unsqueeze(0) * (t1[:, 1] - t1[:, 0]).unsqueeze(1) + t1[:, 0].unsqueeze(1)
        phi_samples   = torch.linspace(0, 1, N, device=device).unsqueeze(0) * (t2[:, 1] - t2[:, 0]).unsqueeze(1) + t2[:, 0].unsqueeze(1)

        theta_rad = torch.deg2rad(theta_samples)  # [B, N]
        phi_rad   = torch.deg2rad(phi_samples)    # [B, N]

        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_phi   = torch.cos(phi_rad)
        sin_phi   = torch.sin(phi_rad)
        u = torch.stack([
            cos_theta * cos_phi,  # x
            sin_theta * cos_phi,  # y
            sin_phi               # z
        ], dim=-1)  # [B, N, 3]
        d_p = torch.einsum('pc,bnc->bnp', self.delta_vecs, u)  # [B, N, P]
        tau = d_p * self.fs / self.v  # [B, N, P]
        tpd = 2 * torch.pi * tau.unsqueeze(-1) * freqs.view(1, 1, 1, -1)  # [B, N, P, F]
        tpd =torch.mean(tpd,dim=1)
        return tpd


class SpeakerEmbeddingExtractor(nn.Module):
    """使用ECAPA-TDNN提取说话人嵌入（适配 SpeechBrain ≥1.0.0）"""
    def __init__(self, output_dim=192, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.output_dim = output_dim
        self.device = device

        try:
            self.ecapa = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                # source="/path/to/your/av_sep/pretrained_models/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device}   # 直接支持 cuda
            )
            self.embed_dim = 192
            self.use_projection = False
            print("--------------ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) 加载成功！-------------")
        except Exception as e:
            print(f"-------------Warning: 无法加载官方ECAPA-TDNN ({e})，使用简化替代模型--------------")
            raise e

    def forward(self, enrollment_audio):
        """
        Args:
            enrollment_audio: [B, T] 或 [B, 1, T] 的波形
        Returns:
            embedding: [B, output_dim]
        """
        # 确保在正确设备上
        enrollment_audio = enrollment_audio.to(self.device)

        if hasattr(self.ecapa, 'encode_batch'):
            # print("==yes==" * 10)
            # 官方模型：直接返回 [B, 1, 192] 或 [B, 192]
            with torch.no_grad():
                embeddings = self.ecapa.encode_batch(enrollment_audio)
                embeddings = embeddings.squeeze(1)  # [B, 192]
        else:
            # 简化模型
            if enrollment_audio.dim() == 2:
                enrollment_audio = enrollment_audio.unsqueeze(1)  # [B, 1, T]
            embeddings = self.ecapa(enrollment_audio).squeeze(-1)  # [B, 512]
            if self.use_projection:
                embeddings = self.projection(embeddings)

        if self.use_projection and embeddings.shape[-1] != self.output_dim:
            embeddings = F.adaptive_avg_pool1d(embeddings.unsqueeze(1), self.output_dim).squeeze(1)

        return embeddings  # [B, 192]

class AudioPromptProcessor(nn.Module):
    """
    不再强行扩展时间维度，只做特征映射。
    将 ECAPA 的 192 维映射到 SepFormer 的 256 维。
    """
    def __init__(self, embed_dim=192, out_channels=256):
        super(AudioPromptProcessor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid() # Sigmoid 用于 FiLM 门控机制非常好
        )

    def forward(self, speaker_embedding):
        """
        Args:
            speaker_embedding: [B, embed_dim]
        Returns:
            prompt_vec: [B, out_channels] (作为全局声纹特征)
        """
        return self.fc(speaker_embedding)


class PromptAdapter(nn.Module):
    def __init__(self, in_channels, prompt_dim, bottleneck=64):
        super().__init__()
        self.h_proj = nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False)
        self.p_proj = nn.Linear(prompt_dim, bottleneck, bias=False)
        self.p_proj_time = nn.Conv1d(prompt_dim, bottleneck, kernel_size=1, bias=False)
        self.act = nn.PReLU()
        self.out_proj = nn.Conv2d(bottleneck, in_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def _prepare_prompt(self, p, target_s):
        if p.dim() == 2:
            p = self.p_proj(p).unsqueeze(-1).unsqueeze(-1)
            return p.expand(-1, -1, 1, target_s)
        if p.dim() == 3:
            if p.shape[1] != self.p_proj_time.in_channels and p.shape[2] == self.p_proj_time.in_channels:
                p = p.transpose(1, 2)
            if p.shape[-1] != target_s:
                p = F.interpolate(p, size=target_s, mode="linear", align_corners=False)
            p = self.p_proj_time(p).unsqueeze(-2)
            return p
        raise ValueError("Unsupported prompt shape")

    def forward(self, h, p):
        b, _, k, s = h.shape
        h_proj = self.h_proj(h)
        p_proj = self._prepare_prompt(p, s)
        z = self.act(h_proj + p_proj)
        return self.out_proj(z)


class PromptGate(nn.Module):
    def __init__(self, prompt_dim, K=160, hidden_dim=32):
        super().__init__()
        self.K = K
        self.p_proj = nn.Conv1d(prompt_dim, hidden_dim, kernel_size=1)
        self.h_proj = nn.Conv1d(K, hidden_dim, kernel_size=1)       # K -> hidden
        self.out = nn.Conv1d(hidden_dim, K, kernel_size=1)          # hidden -> K

    def _prepare_prompt(self, p, target_s):
        if p.dim() == 2:
            return p.unsqueeze(-1).expand(-1, -1, target_s)         # [B,D,S]
        if p.dim() == 3:
            if p.shape[1] != self.p_proj.in_channels and p.shape[2] == self.p_proj.in_channels:
                p = p.transpose(1, 2)
            if p.shape[-1] != target_s:
                p = F.interpolate(p, size=target_s, mode="linear", align_corners=False)
            return p
        raise ValueError("Unsupported prompt shape")

    def _prepare_reliability(self, r, target_s, device):
        if r is None:
            return None
        if not torch.is_tensor(r):
            r = torch.tensor(r, dtype=torch.float32, device=device)
        if r.dim() == 0:
            r = r.view(1, 1, 1)                       # [1,1,1]
        elif r.dim() == 1:
            r = r.unsqueeze(1).unsqueeze(-1)          # [B,1,1]
        elif r.dim() == 2:
            # 常见两种：[B,S] 或 [B,1]
            if r.shape[1] == target_s:
                r = r.unsqueeze(1)                     # [B,1,S]
            else: 
                r = r.unsqueeze(-1)                    # [B,S,1]
        elif r.dim() == 3 and r.shape[-1] != target_s:
            r = F.interpolate(r, size=target_s, mode="linear", align_corners=False)
        return r.clamp(0.0, 1.0)

    def forward(self, h, p, reliability=None):
        # h: [B,N,K,S]
        B, N, K, S = h.shape
        assert K == self.K, f"K mismatch: got {K}, expect {self.K}"

        # h_stat: [B,K,S]
        h_stat = torch.mean(h, dim=1)                               # [B,K,S]

        p_feat = self._prepare_prompt(p, S)                         # [B,D,S]
        z = self.p_proj(p_feat) + self.h_proj(h_stat)               # [B,hidden,S]
        g = torch.sigmoid(self.out(torch.tanh(z)))                  # [B,K,S]

        r = self._prepare_reliability(reliability, S, h.device)
        if r is not None:
            # r :[B,1,S] 或 [B,1,1]
            g = g * r

        return g.unsqueeze(1)                                       # [B,1,K,S]


class ModalityTokenFusion(nn.Module):
    def __init__(
        self,
        token_dim=128,
        pre_dim=192,
        text_dim=512,
        spa_dim=24,
        video_dim=256,
        video_dim_in_1024=1024,
        video_dim_in_512=512,
        nhead=4,
        ff_mult=4,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.pre_proj = nn.Linear(pre_dim, token_dim)
        self.text_proj = nn.Linear(text_dim, token_dim)
        self.spa_proj = nn.Linear(spa_dim, token_dim)
        self.pre_ln = nn.LayerNorm(token_dim)
        self.text_ln = nn.LayerNorm(token_dim)
        self.spa_ln = nn.LayerNorm(token_dim)

        self.video_proj_1024 = nn.Linear(video_dim_in_1024, video_dim)
        self.video_proj_512 = nn.Linear(video_dim_in_512, video_dim)
        self.video_to_token = nn.Linear(video_dim, token_dim)
        self.video_ln = nn.LayerNorm(token_dim)

        self.mod_embed = nn.Embedding(4, token_dim)

        self.video_self_attn = nn.MultiheadAttention(token_dim, nhead, dropout=0.0, batch_first=True)
        self.video_self_ln = nn.LayerNorm(token_dim)
        self.other_self_attn = nn.MultiheadAttention(token_dim, nhead, dropout=0.0, batch_first=True)
        self.other_self_ln = nn.LayerNorm(token_dim)
        self.o2v_attn = nn.MultiheadAttention(token_dim, nhead, dropout=0.0, batch_first=True)
        self.o2v_ln = nn.LayerNorm(token_dim)
        self.v2o_attn = nn.MultiheadAttention(token_dim, nhead, dropout=0.0, batch_first=True)
        self.v2o_ln = nn.LayerNorm(token_dim)

        self.pre_out = nn.Linear(token_dim, pre_dim)
        self.text_out = nn.Linear(token_dim, text_dim)
        self.spa_out = nn.Linear(token_dim, spa_dim)
        self.video_out = nn.Linear(token_dim, video_dim)

    def _make_dummy_token(self, batch_size, length, device, dtype):
        return torch.zeros(batch_size, length, self.token_dim, device=device, dtype=dtype)

    def _get_batch_size(self, *tensors):
        for t in tensors:
            if t is not None:
                return t.shape[0]
        return None

    def _pool_spatial(self, spa):
        if spa.dim() == 3:
            return spa.mean(dim=-1)
        if spa.dim() == 2:
            return spa
        raise ValueError("Unsupported spatial prompt shape")

    def _sinusoidal_pos_enc(self, length, device, dtype):
        if length <= 0:
            return None
        half_dim = self.token_dim // 2
        div_term = torch.exp(
            torch.arange(half_dim, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half_dim))
        )
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        pe = torch.zeros(length, self.token_dim, device=device, dtype=dtype)
        pe[:, 0:half_dim] = torch.sin(positions * div_term)
        pe[:, half_dim:2 * half_dim] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def _tokenize_video(self, video):
        if video.dim() != 3:
            raise ValueError("Unsupported video shape for fusion")
        # Expect [B, T, C] or [B, C, T]
        if video.shape[-1] in (256, 512, 1024):
            v = video
            is_btc = True
        elif video.shape[1] in (256, 512, 1024):
            v = video.transpose(1, 2)
            is_btc = False
        else:
            raise ValueError("Unexpected video channel size for fusion")
        if v.shape[-1] == 1024:
            v = self.video_proj_1024(v)
        elif v.shape[-1] == 512:
            v = self.video_proj_512(v)
        # Now v is [B, T, 256]
        tokens = self.video_ln(self.video_to_token(v))  # [B, T, D]
        video_valid = (v.abs().sum(dim=-1) > 0)
        return tokens, is_btc, video_valid

    def forward(self, pre=None, text=None, spa=None, video=None):
        batch_size = self._get_batch_size(pre, text, spa, video)
        if batch_size is None:
            return None

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        present = {
            "pre": pre is not None,
            "text": text is not None,
            "spa": spa is not None,
            "video": video is not None,
        }

        u_pre = self.pre_ln(self.pre_proj(pre)) if pre is not None else None
        u_txt = self.text_ln(self.text_proj(text)) if text is not None else None
        if spa is not None:
            spa_pooled = self._pool_spatial(spa)
            u_spa = self.spa_ln(self.spa_proj(spa_pooled))
        else:
            u_spa = None

        if video is not None:
            v_tokens, video_is_btc, video_valid = self._tokenize_video(video)
            video_len = v_tokens.size(1)
            video_pe = self._sinusoidal_pos_enc(video_len, v_tokens.device, v_tokens.dtype)
            if video_pe is not None:
                v_tokens = v_tokens + video_pe
            video_embed = self.mod_embed(torch.tensor(0, device=v_tokens.device))
            v_tokens = v_tokens + video_embed.view(1, 1, -1).expand(batch_size, video_len, -1)
            video_pad_mask = ~video_valid
            v_sa, _ = self.video_self_attn(v_tokens, v_tokens, v_tokens, key_padding_mask=video_pad_mask)
            v_tokens = self.video_self_ln(v_tokens + v_sa)
        else:
            v_tokens = None
            video_is_btc = True
            video_pad_mask = None

        other_tokens = []
        other_ids = []
        if u_spa is not None:
            spa_tok = u_spa.unsqueeze(1)
            spa_embed = self.mod_embed(torch.tensor(1, device=device))
            other_tokens.append(spa_tok + spa_embed.view(1, 1, -1))
            other_ids.append("spa")
        if u_txt is not None:
            txt_tok = u_txt.unsqueeze(1)
            txt_embed = self.mod_embed(torch.tensor(2, device=device))
            other_tokens.append(txt_tok + txt_embed.view(1, 1, -1))
            other_ids.append("text")
        if u_pre is not None:
            pre_tok = u_pre.unsqueeze(1)
            pre_embed = self.mod_embed(torch.tensor(3, device=device))
            other_tokens.append(pre_tok + pre_embed.view(1, 1, -1))
            other_ids.append("pre")

        if other_tokens:
            others = torch.cat(other_tokens, dim=1)
            # print("others.shape1 : ", others.shape)
            o_sa, _ = self.other_self_attn(others, others, others, key_padding_mask=None)
            others = self.other_self_ln(others + o_sa)
            # print("others.shape2 : ", others.shape)
        else:
            others = None

        if others is not None and v_tokens is not None:
            # print("video_pad_mask : ", video_pad_mask)
            o2v, _ = self.o2v_attn(others, v_tokens, v_tokens, key_padding_mask=video_pad_mask)
            others = self.o2v_ln(others + o2v)

            v2o, _ = self.v2o_attn(v_tokens, others, others, key_padding_mask=None)
            v_tokens = self.v2o_ln(v_tokens + v2o)

        u_spa = None
        u_txt = None
        u_pre = None
        if others is not None:
            for idx, name in enumerate(other_ids):
                if name == "spa":
                    u_spa = others[:, idx]
                elif name == "text":
                    u_txt = others[:, idx]
                elif name == "pre":
                    u_pre = others[:, idx]

        u_vid = v_tokens
        outputs = {
            "pre": self.pre_out(u_pre) if present["pre"] else None,
            "text": self.text_out(u_txt) if present["text"] else None,
            "spa": self.spa_out(u_spa) if present["spa"] else None,
            "video": None,
        }
        if present["video"]:
            v_out = self.video_out(u_vid)  # [B, T, 256]
            if not video_is_btc:
                v_out = v_out.transpose(1, 2)  # [B, 256, T]
            outputs["video"] = v_out

        # Build memory tokens M = [V'; O'] with type embeddings and mask for missing modalities.
        mem_tokens = []
        mem_masks = []

        if v_tokens is not None:
            mem_tokens.append(v_tokens)
            mem_masks.append(video_pad_mask if video_pad_mask is not None else torch.zeros(batch_size, v_tokens.size(1), dtype=torch.bool, device=device))
        else:
            dummy_v = self._make_dummy_token(batch_size, 1, device, dtype)
            mem_tokens.append(dummy_v)
            mem_masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        # Fixed order: spa, text, pre. One token each, masked if missing.
        spa_tok = u_spa.unsqueeze(1) if u_spa is not None else self._make_dummy_token(batch_size, 1, device, dtype)
        text_tok = u_txt.unsqueeze(1) if u_txt is not None else self._make_dummy_token(batch_size, 1, device, dtype)
        pre_tok = u_pre.unsqueeze(1) if u_pre is not None else self._make_dummy_token(batch_size, 1, device, dtype)

        mem_tokens.extend([spa_tok, text_tok, pre_tok])
        mem_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device) if u_spa is not None else torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        mem_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device) if u_txt is not None else torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        mem_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device) if u_pre is not None else torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        memory = torch.cat(mem_tokens, dim=1)
        memory_mask = torch.cat(mem_masks, dim=1)
        outputs["memory_tokens"] = memory
        outputs["memory_mask"] = memory_mask
        return outputs


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm,
              self).__init__(dim,
                             elementwise_affine=elementwise_affine,
                             eps=1e-8)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """
    def __init__(self, kernel_size=16, out_channels=256, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d_4 = nn.Conv1d(
            in_channels=4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.conv1d_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        if x.dim() == 3:
            if x.shape[2] == 4:
                x = x.transpose(1, 2)  # B x L x 4 -> B x 4 x L
                x = self.conv1d_4(x)
                x = F.relu(x)
                return x
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d_1(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class VisualConv1D(nn.Module):
    def __init__(self,input_dim=512):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        # norm_1 = nn.BatchNorm1d(input_dim)
        norm_1 = select_norm("ln", input_dim, 1)
        dsconv = nn.Conv1d(input_dim,
                           input_dim,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=input_dim,
                           bias=False)
        prelu = nn.PReLU()
        # norm_2 = nn.BatchNorm1d(input_dim)
        norm_2 = select_norm("ln", input_dim, 1)
        pw_conv = nn.Conv1d(input_dim, input_dim, 1, bias=False)
        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class cross_attention_layer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, norm_first=True, *args, **kwargs):
        super().__init__(d_model,
                         nhead,
                         norm_first=norm_first,
                         *args,
                         **kwargs)

    def forward(self, x, v):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        if self.norm_first:
            x = x + v + self._ca_block(self.norm1(x), self.norm1(v))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._ca_block(x, v))
            x = self.norm2(x + self._ff_block(v))

        return x

    def _ca_block(self, x, v):
        x = self.self_attn(v,
                           x,
                           x,
                           attn_mask=None,
                           key_padding_mask=None,
                           need_weights=False)[0]
        return self.dropout1(x)


class CrossTransformer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 depth=4,
                 dropout=0.1,
                 dim_feedforward=2048,
                 activation=F.relu):
        super(CrossTransformer, self).__init__()
        self.cross_attention_layer = cross_attention_layer(
            d_model,
            nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm_first=True)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model,
                                           nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout,
                                           activation=activation,
                                           norm_first=True)
                for _ in range(depth - 1)
            ])

    def forward(self, video, audio, cue=None, plugin_fn=None, plugin_ctx=None):
        x = self.cross_attention_layer(audio, video)
        if plugin_fn is not None:
            x = plugin_fn(x, cue, plugin_ctx)
        for layer in self.transformer_layers:
            x = layer(x)
            if plugin_fn is not None:
                x = plugin_fn(x, cue, plugin_ctx)
        return x


class CrossTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=True,
        norm_before=True,
    ):
        super(CrossTransformerBlock, self).__init__()

        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")
        self.mdl = CrossTransformer(d_model,
                                    nhead,
                                    dim_feedforward=d_ffn,
                                    depth=num_layers,
                                    dropout=dropout,
                                    activation=activation)
        self.self_mdl = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_before
                )
                for _ in range(num_layers)
            ])

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model, dropout=0.0)

    def forward(self, x, video=None, cue=None, plugin_fn=None, plugin_ctx=None):
        x = x.permute(1, 0, 2)
        if video is not None:
            video = video.permute(1, 0, 2)
            if self.use_positional_encoding:
                x = self.pos_enc(x)
                video = self.pos_enc(video)
                x = self.mdl(video, x, cue=cue, plugin_fn=plugin_fn, plugin_ctx=plugin_ctx)
            else:
                x = self.mdl(video, x, cue=cue, plugin_fn=plugin_fn, plugin_ctx=plugin_ctx)
            return x.permute(1, 0, 2)
        else:
            if self.use_positional_encoding:
                x = self.pos_enc(x)
            else:
                x = x
            for layer in self.self_mdl:
                x = layer(x)
                if plugin_fn is not None:
                    x = plugin_fn(x, cue, plugin_ctx)
            return x.permute(1, 0, 2)


class SBTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        dropout=0.0,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
    ):
        super(SBTransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")
        self.mdl = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=nhead,
                                           dim_feedforward=d_ffn,
                                           dropout=dropout,
                                           activation=activation,
                                           norm_first=norm_before)
                for _ in range(num_layers)
            ])

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x, cue=None, plugin_fn=None, plugin_ctx=None):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        x = x.permute(1, 0, 2)
        if self.use_positional_encoding:
            x = self.pos_enc(x)
        else:
            x = x
        for layer in self.mdl:
            x = layer(x)
            if plugin_fn is not None:
                x = plugin_fn(x, cue, plugin_ctx)
        return x.permute(1, 0, 2)


class Cross_Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        prompt_specs=None,
    ):
        super(Cross_Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.pos2d = PositionalEncodingPermute2D(256)
        self.prompt_adapters = nn.ModuleDict()
        self.prompt_gates = nn.ModuleDict()
        if prompt_specs:
            hidden_dim = max(8, out_channels // 8)
            for name, prompt_dim in prompt_specs.items():
                self.prompt_adapters[name] = PromptAdapter(out_channels, prompt_dim)
                self.prompt_gates[name] = PromptGate(prompt_dim, hidden_dim=hidden_dim, K=160)

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

    def forward(self, x, cue):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        pe = self.pos2d(x)
        x = x + pe
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        intra = self.intra_mdl(
            intra,
            cue=cue,
            plugin_fn=self._apply_prompt_plugins_intra,
            plugin_ctx={"B": B, "S": S},
        )
        intra = intra.view(B, S, K, N).permute(0, 3, 2, 1).contiguous()
        if self.norm is not None: 
            intra = self.intra_norm(intra)
        if self.skip_around_intra: 
            intra = intra + x
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        inter = self.inter_mdl(
            inter,
            cue=cue,
            plugin_fn=self._apply_prompt_plugins_inter,
            plugin_ctx={"B": B, "K": K},
        )
        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out

    def _apply_prompt_plugins(self, h, cue):
        if not self.prompt_adapters or cue is None:
            return h
        reliability = cue.get("reliability", {})
        for name, adapter in self.prompt_adapters.items():
            p = cue.get(name)
            if p is None:
                continue
            r = reliability.get(name) if isinstance(reliability, dict) else None
            g = self.prompt_gates[name](h, p , reliability=r)
            # g = g.unsqueeze(2)
            h = h + g * adapter(h, p)
        return h

    def _apply_prompt_plugins_intra(self, x, cue, plugin_ctx):
        if not self.prompt_adapters or cue is None:
            return x
        b = plugin_ctx["B"]
        s = plugin_ctx["S"]
        x = x.permute(1, 0, 2).contiguous()
        bs = x.size(0)
        k = x.size(1)
        n = x.size(2)
        if bs != b * s:
            raise ValueError(f"Intra plugin shape mismatch: got {bs}, expected {b * s}")
        h = x.view(b, s, k, n).permute(0, 3, 2, 1).contiguous()
        h = self._apply_prompt_plugins(h, cue)
        x = h.permute(0, 3, 2, 1).contiguous().view(bs, k, n)
        return x.permute(1, 0, 2).contiguous()

    def _apply_prompt_plugins_inter(self, x, cue, plugin_ctx):
        if not self.prompt_adapters or cue is None:
            return x
        b = plugin_ctx["B"]
        k = plugin_ctx["K"]
        x = x.permute(1, 0, 2).contiguous()
        bk = x.size(0)
        s = x.size(1)
        n = x.size(2)
        if bk != b * k:
            raise ValueError(f"Inter plugin shape mismatch: got {bk}, expected {b * k}")
        h = x.view(b, k, s, n).permute(0, 3, 1, 2).contiguous()
        h = self._apply_prompt_plugins(h, cue)
        x = h.permute(0, 2, 3, 1).contiguous().view(bk, s, n)
        return x.permute(1, 0, 2).contiguous()


class Cross_Dual_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=160,
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
        prompt_specs=None,
        memory_dim=128,
        memory_nhead=4,
    ):
        super().__init__()
        super(Cross_Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)
        vedio_dim_in = 1024
        vedio_dim_out = 256
        self.visual_proj_1024 = nn.Conv1d(vedio_dim_in, vedio_dim_out, kernel_size=1, bias=False)
        self.visual_proj_512 = nn.Conv1d(512, vedio_dim_out, kernel_size=1, bias=False)
        ve_blocks = []
        for _ in range(5):
            ve_blocks += [VisualConv1D(input_dim=vedio_dim_out)]
        ve_blocks += [nn.Conv1d(vedio_dim_out, 256, 1)]
        self.visual_conv = nn.Sequential(*ve_blocks)

        # Audio (query) to multimodal memory cross-attention
        self.memory_dim = memory_dim
        self.audio_q_proj = nn.Linear(out_channels, memory_dim, bias=False)
        self.mem_kv_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.audio_attn = nn.MultiheadAttention(memory_dim, memory_nhead, dropout=0.0, batch_first=True)
        self.audio_ln1 = nn.LayerNorm(memory_dim)
        self.audio_ln2 = nn.LayerNorm(memory_dim)
        self.audio_ffn = nn.Sequential(
            nn.Linear(memory_dim, memory_dim * 4),
            nn.ReLU(),
            nn.Linear(memory_dim * 4, memory_dim),
        )
        self.audio_out_proj = nn.Linear(memory_dim, out_channels, bias=False)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Cross_Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        prompt_specs=prompt_specs,
                    )))

        self.conv2d = nn.Conv2d(out_channels,
                                out_channels * num_spks,
                                kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x, cue):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """
        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (x.size(1)**0.5)

        # Audio query cross-attention over multimodal memory tokens
        memory = cue.get("memory_tokens") if cue is not None else None
        memory_mask = cue.get("memory_mask") if cue is not None else None
        if memory is not None:
            h = x.transpose(1, 2)  # [B, T, C]
            q = self.audio_q_proj(h)  # [B, T, d]
            m = self.mem_kv_proj(memory)  # [B, Lm, d]
            delta, _ = self.audio_attn(q, m, m, key_padding_mask=memory_mask)
            h_star = self.audio_ln1(q + delta)
            h_star = self.audio_ln2(h_star + self.audio_ffn(h_star))
            x = self.audio_out_proj(h_star).transpose(1, 2)  # [B, C, T]

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        if cue['video'] is not None and not cue.get("video_is_prompt", False):
            # print("video.shape", cue['video'].shape)
            video_input = cue['video'].transpose(1, 2)
            # print('video shape2', video_input.shape)
            if video_input.shape[1] == self.visual_proj_1024.in_channels:
                cue['video'] = self.visual_proj_1024(video_input)
            elif video_input.shape[1] == self.visual_proj_512.in_channels:
                cue['video'] = self.visual_proj_512(video_input)
            else:
                raise ValueError(
                    f"Unexpected video channel size {video_input.shape[1]} (expected 512 or 1024)"
                )
            # print("after proj video.shape", cue['video'].shape)

            cue['video'] = self.visual_conv(cue['video'])
            # print('video shape3', cue['video'].shape)
            # print('x shape1', x.shape)
            # cue['video'] = F.pad(cue['video'], (0, x.shape[-1] - cue['video'].shape[-1]), mode='replicate')
            # === 无论视频比音频长还是短，一律强制缩放对齐 ===
            if cue['video'].shape[-1] != x.shape[-1]:
                cue['video'] = F.interpolate(cue['video'], size=x.shape[-1], mode='linear', align_corners=False)
                # print('video shape4', cue['video'].shape)
                # print('x shape2', x.shape)
        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, cue)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (torch.cat([input1, input2], dim=3).view(B, N, -1,
                                                         K).transpose(2, 3))

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Cross_Sepformer(nn.Module):
    def __init__(self,
                 IntraSeparator,
                 InterSeparator,
                 kernel_size=16,
                 N_encoder_out=256,
                 num_spks=2):
        super(Cross_Sepformer, self).__init__()
        self.speaker_extractor = SpeakerEmbeddingExtractor()
        self.ipd = TimeDomainIPD(kernel_size=kernel_size, num_filters=N_encoder_out)
        self.tpd =TPDLayer()

        self.textmodel = laion_clap.CLAP_Module(enable_fusion=True) # 文本编码器（CLAP），把语义查询映射到语义向量
        self.textmodel.load_ckpt(ckpt=_get_clap_ckpt())
        for param in self.textmodel.parameters():
            param.requires_grad = False
        self.AudioEncoder = Encoder(kernel_size=kernel_size,
                                    out_channels=N_encoder_out)

        self.AudioDecoder = Decoder(in_channels=N_encoder_out,
                                    out_channels=1,
                                    kernel_size=kernel_size,
                                    stride=kernel_size // 2,
                                    bias=False)
        self.Separator = Cross_Dual_Path_Model(num_spks=num_spks,
                                               in_channels=N_encoder_out,
                                               out_channels=N_encoder_out,
                                               num_layers=2,
                                               K=160,
                                               intra_model=IntraSeparator,
                                               inter_model=InterSeparator,
                                               norm='ln',
                                               skip_around_intra=True,
                                               memory_dim=128,
                                               memory_nhead=4,
                                               prompt_specs={
                                                   "pre": 192,
                                                   "text": 512,
                                                   "spa": len(mic_pairs) * 4,
                                                   "video": N_encoder_out,
                                               })
        self.fusion = ModalityTokenFusion(
            token_dim=128,
            pre_dim=192,
            text_dim=512,
            spa_dim=len(mic_pairs) * 4,
            video_dim=N_encoder_out,
            nhead=4,
            ff_mult=4,
        )
        self.num_spks = num_spks

    def compute_V_time_domain(self, ipd, tpd):
        tpd =tpd.unsqueeze(-1).expand(-1, -1, -1, ipd.size(-1))
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)
        cos_tpd = torch.cos(tpd)  
        sin_tpd = torch.sin(tpd)
        e_ipd = torch.stack([cos_ipd, sin_ipd], dim=-1)  # IPD 的方向向量
        e_tpd = torch.stack([cos_tpd, sin_tpd], dim=-1)  # TPD 的方向向量
        # 计算每对麦克风的内积
        inner_product = torch.sum(e_ipd * e_tpd, dim=-1)  # 内积：<e(TPD), e(IPD)>
        return inner_product

    def forward(
        self,
        mix,
        video=None,
        pre=None,
        text=None,
        main=None,
        pre_rel=None,
        text_rel=None,
        video_rel=None,
        spa_rel=None,
    ):
        device = next(self.parameters()).device
        mix = mix.to(device)
        if video is not None:
            video = video.to(device)
        if pre is not None:
            pre = pre.to(device)
        if main is not None:
            main = main.to(device)
        # print("+++mix.shape:+++", mix.shape)
        mix_w = self.AudioEncoder(mix)
        mix_ = mix_w
        # print("+++mix_w.shape:+++", mix_w.shape)
        cue = {
            "pre": None,
            "spa": None,
            "text": None,
            "video": None,
            "reliability": {},
        }
        pre_embed = None
        text_embed = None
        spa_embed = None

        if pre is not None:
            # print("pre.shape++++++++", pre.shape)
            with torch.no_grad():
                # pre = pre[:, :, 0]
                pre_embed = self.speaker_extractor(pre) 
            # print("pre_embed.shape : ", pre_embed.shape)
            if pre_rel is not None:
                cue["reliability"]["pre"] = pre_rel
    
        if text is not None:
            text_embed = self.textmodel.get_text_embedding(text, use_tensor=True)  # [B, D]
            text_embed = text_embed.to(mix_w.device)
            # print("text_embed.shape : ", text_embed.shape)
            if text_rel is not None:
                cue["reliability"]["text"] = text_rel

        if main is not None:
            ipd = self.ipd(mix) 
            tpd = self.tpd(main)
            # print("ipd : ", ipd.shape)
            # print("tpd : ", tpd.shape)
            V = self.compute_V_time_domain(ipd, tpd)
            V= V.real.float()
            V = torch.mean(V,dim=2).repeat(1, 2, 1)         # [B, 2P, T]
            ipd = torch.mean(ipd,dim=2).repeat(1, 2, 1)     # [B, 2P, T]
            C_spa =torch.cat((ipd, V), dim=1) 
            spa_embed = C_spa
            # print("C_spa.shape : ", C_spa.shape)
            if spa_rel is not None:
                cue["reliability"]["spa"] = spa_rel

        if video is not None and video_rel is not None:
            cue["reliability"]["video"] = video_rel

        fused = self.fusion(pre=pre_embed, text=text_embed, spa=spa_embed, video=video)
        if fused is not None:
            cue["pre"] = fused.get("pre")
            cue["text"] = fused.get("text")
            cue["spa"] = fused.get("spa")
            cue["video"] = fused.get("video")
            cue["memory_tokens"] = fused.get("memory_tokens")
            cue["memory_mask"] = fused.get("memory_mask")
            cue["video_is_prompt"] = True
            # print("True fused")

        # print("mix_w.shape:", mix_w.shape)

        est_mask = self.Separator(mix_w, cue)
        mix_w = torch.stack([mix_] * self.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.AudioDecoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        # print(est_source.shape)
        est_source = est_source.permute(0, 2, 1).squeeze(1)  # [spks, B, L]
        # print("est_source.shape:", est_source.shape)
        return est_source


def Cross_Sepformer_warpper(kernel_size=16, N_encoder_out=256, num_spks=1):

    InterSeparator = CrossTransformerBlock(num_layers=8,
                                           d_model=N_encoder_out,
                                           nhead=8,
                                           d_ffn=1024,
                                           dropout=0,
                                           use_positional_encoding=False,
                                           norm_before=True)
    IntraSeparator = SBTransformerBlock(num_layers=8,
                                        d_model=N_encoder_out,
                                        nhead=8,
                                        d_ffn=1024,
                                        dropout=0,
                                        use_positional_encoding=True,
                                        norm_before=True)
    return Cross_Sepformer(IntraSeparator,
                           InterSeparator,
                           kernel_size=kernel_size,
                           N_encoder_out=N_encoder_out,
                           num_spks=num_spks)


# Alias used by `train.py` via `getattr(models, config["model"])()`.
uniCueNet = Cross_Sepformer_warpper


if __name__ == '__main__':
    model = Cross_Sepformer_warpper().cuda()
    # print(model(torch.randn(2, 16000), torch.randn(2, 300, 512)).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = torch.randn(2, 16000).cuda(), text = ["hello world", "hi there"], main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = None, text = None, main = None).shape)    
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = None, text = None, main = None).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = torch.randn(2, 16000).cuda(), text = None, main = None).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = None, text = ["hello world", "hi there"], main = None).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = None, text = None, main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = torch.randn(2, 16000).cuda(), text = ["hello world", "hi there"], main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = None, text = ["hello world", "hi there"], main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = torch.randn(2, 16000).cuda(), text = None, main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = torch.randn(2, 16000).cuda(), text = ["hello world", "hi there"], main = None).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = None, text = ["hello world", "hi there"], main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = torch.randn(2, 16000).cuda(), text = None, main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = torch.randn(2, 16000).cuda(), text = ["hello world", "hi there"], main = None).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = None, text = None, main = torch.randint(0, 10, (1, 2)).squeeze(0).cuda()).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = None, text = ["hello world", "hi there"], main = None).shape)
    print(model(torch.randn(2, 16000, 4).cuda(), video = torch.randn(2, 300, 1024).cuda(), pre = torch.randn(2, 16000).cuda(), text = None, main = None).shape)

    # print(model(torch.randn(2, 16000, 4).cuda(), video = None, pre = None, text = ["hello world", "hi there"], main = None).shape)

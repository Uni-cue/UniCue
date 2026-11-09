import torch
import torch.nn as nn

# ===== add for evaluation metrics (returns dB, higher is better) =====
# 注意：这些函数用于“评估指标”，返回的是每条样本的 dB 值（越大越好）
# 训练时仍然用下面的 SI_SNR 类（返回负的 mean 作为 loss）

def _trim_to_min_len(a: torch.Tensor, b: torch.Tensor):
    # a, b: [B, L] (or [B, 1, L])
    if a.dim() == 3:
        a = a[:, :, 0]
    if b.dim() == 3:
        b = b[:, :, 0]
    L = min(a.shape[-1], b.shape[-1])
    # 【修复】切片必须用 ... 保留 batch 维
    return a[..., :L], b[..., :L]

@torch.no_grad()
def si_snr_db(source: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns: [B] SI-SNR in dB (higher is better)
    """
    s, x = _trim_to_min_len(source, estimate)          # [B, L]
    s = s - s.mean(dim=-1, keepdim=True)
    x = x - x.mean(dim=-1, keepdim=True)

    ref_energy = torch.sum(s ** 2, dim=-1, keepdim=True) + eps
    proj = torch.sum(s * x, dim=-1, keepdim=True) * s / ref_energy
    noise = x - proj

    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)             # [B]

@torch.no_grad()
def si_sdr_db(source: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    A commonly-used SI-SDR implementation (projection-based), returns [B] in dB.
    In practice this matches many separation baselines' SI-SDR form.
    """
    # 对于大多数投影式实现，SI-SDR 与 SI-SNR 的计算形式一致（差别更多在命名/解释）
    return si_snr_db(source, estimate, eps=eps)


class SI_SNR(nn.Module):
    def __init__(self):
        super(SI_SNR, self).__init__()
        self.EPS = 1e-5

    def forward(self, source, estimate_source):
        if source.dim() == 3:
            source = source[:, :, 0]
        # print("source1 :", source.shape)
        # print("estimate_source1 :", estimate_source.shape)
        if source.shape[-1] > estimate_source.shape[-1]:
            source = source[..., :estimate_source.shape[-1]]
        if source.shape[-1] < estimate_source.shape[-1]:
            estimate_source = estimate_source[..., :source.shape[-1]]
        
        # print("source2 :", source.shape)
        # print("estimate_source2 :", estimate_source.shape)

        # step 1: Zero-mean norm
        source = source - torch.mean(source, dim=-1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

        # step 2: Cal si_snr
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, dim = -1, keepdim=True) + self.EPS
        proj = torch.sum(source * estimate_source, dim = -1, keepdim=True) * source / ref_energy
        # e_noise = s' - s_target
        noise = estimate_source - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, dim = -1) / (torch.sum(noise ** 2, dim = -1) + self.EPS)
        sisnr = 10 * torch.log10(ratio + self.EPS)

        return 0 - torch.mean(sisnr)

class PIT_SI_SNR(nn.Module):
    """2-speaker PIT SI-SNR
       source, estimate: [B, 2, L]
    """
    def __init__(self):
        super().__init__()
        self.base_si_snr = SI_SNR()

    def forward(self, source, estimate):
        assert source.dim() == 3 and estimate.dim() == 3
        assert source.shape[1] == 2 and estimate.shape[1] == 2

        s1, s2 = source[:, 0, :], source[:, 1, :]
        e1, e2 = estimate[:, 0, :], estimate[:, 1, :]

        # 排列 1: e1->s1, e2->s2
        loss_00 = self.base_si_snr(s1, e1)
        loss_11 = self.base_si_snr(s2, e2)
        loss_perm1 = 0.5 * (loss_00 + loss_11)

        # 排列 2: e1->s2, e2->s1
        loss_01 = self.base_si_snr(s1, e2)
        loss_10 = self.base_si_snr(s2, e1)
        loss_perm2 = 0.5 * (loss_01 + loss_10)

        # 取更小的那个（更大的 SI-SNR）
        return torch.minimum(loss_perm1, loss_perm2)


class MuSE_loss(nn.Module):
    def __init__(self):
        super(MuSE_loss, self).__init__()
        self.si_snr_loss = SI_SNR()
        self.speaker_loss = nn.CrossEntropyLoss()

    def forward(self, tgt_wav, pred_wav, tgt_spk, pred_spk):
        si_snr = self.si_snr_loss(tgt_wav, pred_wav)
        ce = self.speaker_loss(pred_spk[0], tgt_spk) + self.speaker_loss(pred_spk[1], tgt_spk) + self.speaker_loss(pred_spk[2], tgt_spk) + self.speaker_loss(pred_spk[3], tgt_spk)
        return {'si_snr': si_snr, 'ce': ce} 
    
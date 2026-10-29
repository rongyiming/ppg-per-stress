"""
Finetune model: backbone + mixture-of-experts + router + classifier.

    x (B, L)
      -> InceptionTime (or ResNet-1D) backbone, 4 feature levels
      -> residual domain adapter + MixStyle on the deepest map
      -> K experts: Linear(C->256) -> ReLU -> BN -> Dropout -> LSTM(256->64)
      -> routing weight
             w = (1 - alpha) * softmax(MLP(individual))
               + alpha       * softmax(Linear(expert 0 hidden state))
      -> x_fused = sum_k w_k h_k , plus a LayerNorm'd residual backbone skip
      -> BN -> FC(64->128) -> ReLU -> Dropout(0.5) -> BN -> FC(128->2)

``forward`` returns ``(logits, (diversity_loss, routing_loss))``.
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.resnet1d import resnet18_1d, resnet34_1d, resnet50_1d
from models.backbones.inception_time import InceptionTime


class TransformerExpert(nn.Module):
    """Single-layer Transformer encoder expert, output matching LSTM hidden (64-d).

    Replaces LSTM(256→64) with a comparable-parameter single-layer Transformer:
    input (B, T, 256) → 1 encoder layer (d_model=256, 8 heads, ff=512)
    → last-timestep output → Linear(256→64). Returns (B, 64) as the expert
    representation, plus (B, 64) "hidden" for the feature gate.
    """

    def __init__(self, input_size=256, hidden_size=64, num_layers=1,
                 num_heads=8, dim_ff=512, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x: (B, T, 256)
        out = self.encoder(x)              # (B, T, 256)
        last = out[:, -1, :]               # (B, 256)
        rep = self.proj(last)              # (B, 64)
        return rep, rep.unsqueeze(0)       # (B,64), (1,B,64) to match LSTM tuple


class FinetuneModel(nn.Module):
    def __init__(self, input_channels, resnet_depth, expert_num,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers,
                 output_size, alpha=0.7, use_input_norm=False,
                 top_k=None,
                 backbone='resnet18', single_level=False,
                 feature_gate='expert0', expert_type='lstm',
                 use_domain_adapter=True, use_head_skip=True,
                 use_gate_norm=True, use_mixstyle=True, use_experts=True):
        super(FinetuneModel, self).__init__()
        self.feature_gate_type = feature_gate
        self.expert_type = expert_type
        # Architecture switches; all True is the configuration used in the paper.
        self.use_domain_adapter = use_domain_adapter
        self.use_head_skip = use_head_skip
        self.use_gate_norm = use_gate_norm
        self.use_mixstyle = use_mixstyle
        self.use_experts = use_experts

        # 1. Backbone
        if backbone == 'inception_time':
            self.backbone = InceptionTime(input_channels=input_channels)
            self._backbone_resnet = False
        elif backbone == 'resnet18':
            self.backbone = resnet18_1d(input_channels=input_channels)
            self._backbone_resnet = True
        elif backbone == 'resnet34':
            self.backbone = resnet34_1d(input_channels=input_channels)
            self._backbone_resnet = True
        elif backbone == 'resnet50':
            self.backbone = resnet50_1d(input_channels=input_channels)
            self._backbone_resnet = True
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                             f"Choose from resnet18, resnet34, resnet50, inception_time.")
        # Alias kept for callers that refer to the ResNet by name.
        if self._backbone_resnet:
            self.resnet = self.backbone
        self.alpha = alpha
        self.use_input_norm = use_input_norm
        self.expert_num = expert_num
        self.top_k = top_k  # None=softmax, int=top-K sparse routing

        if use_input_norm:
            self.input_norm = nn.InstanceNorm1d(input_channels, affine=True)

        # ── Residual domain adapter: a zero-initialised 1x1 convolution over
        # the deepest backbone feature map.  It is the identity at step 0 and
        # gives the finetuning stage a full-learning-rate channel for correcting
        # the pretrained features, whose own parameters only move at the much
        # smaller backbone learning rate.  A pointwise channel re-mixing leaves
        # the temporal structure read by the LSTM experts untouched, which a
        # temporal (multi-tap) variant did not.
        if use_domain_adapter:
            self.domain_adapter = nn.Conv1d(512, 512, kernel_size=1, bias=False)
            nn.init.zeros_(self.domain_adapter.weight)
        else:
            self.domain_adapter = None

        # ── Residual backbone skip to the decision head (zero-init,
        # LayerNorm'd): a routing-independent shortcut from the domain-adapted
        # deepest-map GAP straight into the fused expert representation.
        # Without the LayerNorm the raw 512-d GAP carries per-channel scales
        # that the zero-initialised linear over-fits to the source domain;
        # normalising makes the shortcut per-sample scale-invariant and keeps
        # it safe on the cross-domain dataset. Identity at step 0.
        if use_head_skip:
            self.head_gap = nn.AdaptiveAvgPool1d(1)
            self.head_norm = nn.LayerNorm(512)
            self.head_skip = nn.Linear(512, lstm_hidden_size)
            nn.init.zeros_(self.head_skip.weight)
            nn.init.zeros_(self.head_skip.bias)
        else:
            self.head_gap = None
            self.head_norm = None
            self.head_skip = None

        # ── Single-expert mode (expert_num=1): no gating, no aux losses ──
        if expert_num == 1:
            self.is_single_expert = True
            # All feature levels → layer4 → projection → single expert
            self.single_projection = nn.Linear(512, lstm_input_size)
            self.single_bn = nn.BatchNorm1d(lstm_input_size)
            if expert_type == 'transformer':
                self.lstm_experts = nn.ModuleList([
                    TransformerExpert(input_size=lstm_input_size,
                                      hidden_size=lstm_hidden_size,
                                      num_layers=lstm_num_layers)
                ])
            elif expert_type == 'gru':
                # GRU experts: two gates instead of three, so fewer parameters
                # and less over-fitting on the small cross-domain finetuning
                # sets.  Bidirectional with a halved hidden size (32 per
                # direction, concatenated back to 64) so the readout sees both
                # past and future context at a parameter count below a
                # unidirectional GRU(64).
                self.lstm_experts = nn.ModuleList([
                    nn.GRU(input_size=lstm_input_size,
                           hidden_size=lstm_hidden_size // 2,
                           num_layers=lstm_num_layers,
                           batch_first=True, bidirectional=True)
                ])
                self._init_rnn_experts_weights()
            elif expert_type == 'lstm_bidir':
                # Bidirectional LSTM at halved hidden size; the readout is
                # [forward_last ; backward_first] rather than out[:, -1].
                self.lstm_experts = nn.ModuleList([
                    nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size // 2,
                            num_layers=lstm_num_layers,
                            batch_first=True, bidirectional=True)
                ])
                self._init_rnn_experts_weights()
            else:
                # Unidirectional LSTM (the configuration reported in the paper).
                self.lstm_experts = nn.ModuleList([
                    nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True, bidirectional=False)
                ])
                self._init_rnn_experts_weights()
            # No gating networks needed
            self.fusion_weights_individual = None
            self.fusion_weights_feature = None
        else:
            self.is_single_expert = False
            level_out_dims = [64, 128, 256, 512]
            if single_level:
                # All experts read the deepest level.
                self.expert_levels = [3] * expert_num
            else:
                # Multi-level: distribute experts across ResNet levels
                self.expert_levels = [e % 4 for e in range(expert_num)]
            self.expert_projections = nn.ModuleList([
                nn.Linear(level_out_dims[self.expert_levels[e]], lstm_input_size)
                for e in range(expert_num)
            ])
            self.bn0_list = nn.ModuleList([
                nn.BatchNorm1d(lstm_input_size) for _ in range(expert_num)
            ])

            # MoE experts — LSTM or Transformer
            if expert_type == 'transformer':
                self.lstm_experts = nn.ModuleList([
                    TransformerExpert(input_size=lstm_input_size,
                                      hidden_size=lstm_hidden_size,
                                      num_layers=lstm_num_layers)
                    for _ in range(expert_num)
                ])
            elif expert_type == 'gru':
                # GRU experts, bidirectional at halved hidden size (see the
                # single-expert note above).
                self.lstm_experts = nn.ModuleList([
                    nn.GRU(
                        input_size=lstm_input_size,
                        hidden_size=lstm_hidden_size // 2,
                        num_layers=lstm_num_layers,
                        batch_first=True, bidirectional=True
                    ) for _ in range(expert_num)
                ])
                self._init_rnn_experts_weights()
            elif expert_type == 'lstm_bidir':
                # Bidirectional LSTM at halved hidden size; the readout is
                # [forward_last ; backward_first] rather than out[:, -1].
                self.lstm_experts = nn.ModuleList([
                    nn.LSTM(
                        input_size=lstm_input_size,
                        hidden_size=lstm_hidden_size // 2,
                        num_layers=lstm_num_layers,
                        batch_first=True, bidirectional=True
                    ) for _ in range(expert_num)
                ])
                self._init_rnn_experts_weights()
            else:
                # Unidirectional LSTM (the configuration reported in the paper).
                self.lstm_experts = nn.ModuleList([
                    nn.LSTM(
                        input_size=lstm_input_size,
                        hidden_size=lstm_hidden_size,
                        num_layers=lstm_num_layers,
                        batch_first=True, bidirectional=False
                    ) for _ in range(expert_num)
                ])
                self._init_rnn_experts_weights()

            # Gating network; the input is the 10 handcrafted descriptors.
            gate_input_dim = 10
            # Per-sample, scale-invariant normalisation of the handcrafted
            # descriptors.  They carry very different units (ms, bpm, %) and
            # shift in absolute scale across devices, so the LayerNorm lets the
            # gate key on the relative physiological profile -- the part that
            # transfers across datasets -- instead of raw magnitude.
            if use_gate_norm:
                self.individual_norm = nn.LayerNorm(gate_input_dim,
                                                    elementwise_affine=False)
            else:
                self.individual_norm = None
            # Individual gate: deeper MLP for more routing capacity
            self.fusion_weights_individual = nn.Sequential(
                nn.Linear(gate_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, expert_num),
            )
            # Feature gate: which representation produces the routing logits.
            if feature_gate == 'per_expert':
                # Each expert scores itself; the scores are concatenated.
                self.fusion_weights_feature = nn.ModuleList([
                    nn.Linear(lstm_hidden_size, 1) for _ in range(expert_num)
                ])
            elif feature_gate == 'backbone_gap':
                # Pooled backbone features, skipping the expert bottleneck.
                self._feature_gate_gap = nn.AdaptiveAvgPool1d(1)
                self.fusion_weights_feature = nn.Linear(512, expert_num)
            elif feature_gate == 'expert_mean':
                # Unweighted mean of the K expert outputs.
                self.fusion_weights_feature = nn.Linear(lstm_hidden_size, expert_num)
            elif feature_gate == 'expert_concat':
                # Concatenation of the K expert outputs.
                self.fusion_weights_feature = nn.Linear(
                    lstm_hidden_size * expert_num, expert_num)
            else:
                # Default: the hidden state of expert 0.
                self.fusion_weights_feature = nn.Linear(lstm_hidden_size, expert_num)

        # Classifier: a plain MLP head.  Spectral normalisation was removed
        # after it showed no consistent benefit once the residual skip and the
        # gate-input normalisation were in place.
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size)
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.5)

    def _init_rnn_experts_weights(self):
        for rnn in self.lstm_experts:
            is_gru = isinstance(rnn, nn.GRU)
            for name, param in rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
                    if not is_gru:
                        if getattr(rnn, 'bidirectional', False):
                            # bidirectional LSTM: set both forget gates = 1
                            h = param.size(0) // 8
                            param.data[h:2 * h].fill_(1.0)     # forward forget
                            param.data[5 * h:6 * h].fill_(1.0)  # backward forget
                        else:
                            n = param.size(0)
                            param.data[n // 4: n // 2].fill_(1.0)  # LSTM forget gate = 1

    def _rnn_readout(self, out, hidden, B):
        """Read the expert summary from an RNN, per expert_type.

        - gru / lstm_bidir (bidirectional): [forward_last ; backward_first]
          from the top hidden layer — the full-sequence summary.
        - lstm (unidirectional): last output timestep (full causal context).
        """
        if self.expert_type in ('gru', 'lstm_bidir'):
            return hidden[-2:].transpose(0, 1).reshape(B, -1)
        return out[:, -1, :]

    def _head_skip(self, deepest_map, x):
        """Residual backbone-GAP shortcut into the decision-head feature.

        deepest_map: (B, 512, T) domain-adapted deepest feature map.
        x:           (B, H) fused expert representation (H = lstm_hidden_size).
        GAP → LayerNorm → zero-init Linear, then added to x (identity at step 0).
        LayerNorm removes the per-channel scale that would otherwise let the
        shortcut over-fit the source domain, making it a cross-domain-safe
        content anchor.
        """
        if not self.use_head_skip:
            return x
        pooled = self.head_gap(deepest_map).squeeze(-1)   # (B, 512)
        pooled = self.head_norm(pooled)                   # (B, 512) scale-invariant
        if x is None:
            return self.head_skip(pooled)                 # head-skip only (no experts)
        return x + self.head_skip(pooled)

    def _mixstyle(self, x, p=0.5):
        """MixStyle: shuffle per-channel statistics across the batch (train only).

        x: (B, C, T) feature map. Channel mean/std over the time axis capture
        the "style" (subject/domain-specific scale + offset) that drives the
        VitalDB→CLAS/CATSA negative transfer; mixing them with a random batch
        permutation decorrelates content from style and regularizes against
        domain shift. No trainable params, identity at eval (self.training
        gate) — WESAD is untouched at inference, only its training is
        regularized.
        """
        if (not self.use_mixstyle or not self.training
                or x.size(0) < 2 or torch.rand(1).item() >= p):
            return x
        mu = x.mean(dim=2, keepdim=True)                 # (B, C, 1)
        std = x.std(dim=2, keepdim=True) + 1e-6          # (B, C, 1)
        perm = torch.randperm(x.size(0), device=x.device)
        lam = torch.rand(x.size(0), 1, 1, device=x.device)  # per-sample mix ratio
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        std_mix = lam * std + (1 - lam) * std[perm]
        return (x - mu) / std * std_mix + mu_mix

    def finetune(self, partial_unfreeze: bool = False, unfreeze_all: bool = False):
        """Set which backbone parameters are trainable.

        Args:
            partial_unfreeze: train only the deeper stages (layer3/layer4 for
                the ResNet backbones, level2/level3 for InceptionTime).
            unfreeze_all: train the whole backbone.
            (neither): freeze the backbone entirely.

        When partial_unfreeze or unfreeze_all is set, the backbone's batch-norm
        layers are put in evaluation mode here.  The training loop calls
        ``model.train()`` at the start of every epoch, which returns them to
        training mode; that is the setting the reported results were produced
        with.
        """
        bk = self.backbone
        if unfreeze_all:
            for param in bk.parameters():
                param.requires_grad = True
            for m in bk.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        elif partial_unfreeze:
            if self._backbone_resnet:
                for name, param in bk.named_parameters():
                    if 'layer3' in name or 'layer4' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                for m in bk.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.train()
            else:
                # InceptionTime: partial_unfreeze = unfreeze level2+level3 blocks
                for name, param in bk.named_parameters():
                    if 'level2' in name or 'level3' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                for m in bk.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.train()
        else:
            for param in bk.parameters():
                param.requires_grad = False

    def expert_diversity_loss(self, expert_outputs):
        """Reward experts producing meaningfully different outputs.

        Each expert outputs (B, H). If experts collapse to the same output
        the std across experts is small → this loss penalises collapse.
        """
        stacked = torch.stack(expert_outputs, dim=1)   # (B, N, H)
        var_per_sample = torch.var(stacked, dim=1).mean()  # mean var over B, H
        return torch.exp(-var_per_sample)                    # 0→1,  ∞→0

    def feature_expert_consist_loss(self, individual, weights,
                                    sample_pairs_num=None):
        """Routing consistency: similar subjects should be routed similarly.

        Samples random pairs and regresses the cosine similarity of their
        individual descriptors onto the cosine similarity of their routing
        weights.
        """
        B, _ = individual.shape
        sample_pairs_num = sample_pairs_num or min(2 * B, 1024)

        b1 = torch.randint(0, B, (sample_pairs_num,), device=individual.device)
        b2 = torch.randint(0, B, (sample_pairs_num,), device=individual.device)

        f1, f2 = individual[b1], individual[b2]
        p1, p2 = weights[b1], weights[b2]

        # Individual features are [0,1]-normalized and routing weights
        # live on the simplex, so both are non-negative and their cosine
        # similarities collapse into [0,1] (weights often 0.8-1.0), making the
        # direct MSE ill-posed. Center both before the cosine so the two
        # similarity distributions span [-1,1] and are actually comparable.
        f1 = f1 - f1.mean(dim=0, keepdim=True)
        f2 = f2 - f2.mean(dim=0, keepdim=True)
        p1 = p1 - 1.0 / p1.size(1)
        p2 = p2 - 1.0 / p2.size(1)

        sim_f = F.cosine_similarity(f1, f2, dim=1)
        sim_p = F.cosine_similarity(p1, p2, dim=1)

        return F.mse_loss(sim_f, sim_p)

    def forward(self, x, individual=None):
        x = x.float()
        individual = individual.float() if individual is not None else None

        x = x.view(x.size(0), x.size(1), -1)   # (B, L) → (B, 1, L)
        x = x.permute(0, 2, 1)                  # (B, C=1, L)

        if self.use_input_norm:
            x = self.input_norm(x)

        # Backbone forward — capture per-layer features
        if self._backbone_resnet:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            f_levels = []
            for layer in [self.backbone.layer1, self.backbone.layer2,
                           self.backbone.layer3, self.backbone.layer4]:
                x = layer(x)
                f_levels.append(x)
        else:
            # InceptionTime returns f_levels directly
            f_levels = self.backbone(x)

        # Residual domain adapter on the deepest feature map (identity at
        # init); re-aligns the pretrained features to the target domain.
        if self.use_domain_adapter:
            f_levels[-1] = f_levels[-1] + self.domain_adapter(f_levels[-1])
        # MixStyle on the deepest map: decorrelate channel style from content
        # so the downstream experts and head do not over-fit the pretraining
        # corpus's style (train only; identity at eval).
        f_levels[-1] = self._mixstyle(f_levels[-1])

        # ── Single-expert path (no MoE, no gating, no aux losses) ──
        if self.is_single_expert:
            fe = f_levels[-1].permute(0, 2, 1)        # (B, T4, 512) — use layer4 only
            fe = self.single_projection(fe)            # (B, T4, 256)
            fe = self.relu(fe)
            B, Te, Fd = fe.size()
            fe = self.single_bn(fe.contiguous().view(-1, Fd)).view(B, Te, Fd)
            fe = self.dropout2(fe)
            if self.expert_type == 'transformer':
                x, _ = self.lstm_experts[0](fe)        # (B, 64)
            elif self.expert_type == 'gru':
                gru_out, gru_hidden = self.lstm_experts[0](fe)
                # Bidirectional GRU: [forward_last ; backward_first] from the
                # top hidden layer.
                x = self._rnn_readout(gru_out, gru_hidden, gru_out.size(0))
            elif self.expert_type == 'lstm_bidir':
                lstm_out, (lstm_hidden, _) = self.lstm_experts[0](fe)
                x = self._rnn_readout(lstm_out, lstm_hidden, lstm_out.size(0))
            else:
                lstm_out, _ = self.lstm_experts[0](fe)
                x = lstm_out[:, -1, :]                 # (B, 64) causal last step
            x = self._head_skip(f_levels[-1], x)       # residual backbone skip

            # Classifier
            x = self.dropout2(x)
            x = self.bn1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout5(x)
            x = self.bn2(x)
            x = self.fc2(x)

            # No experts -> no auxiliary losses.
            zero = torch.tensor(0.0, device=x.device)
            return x, (zero, zero)

        # ── MoE path (expert_num >= 2) ──
        deepest = f_levels[-1]           # deepest feature map (B, 512, T4)

        # head-skip only (no experts, no routing): ablation without experts.
        if not self.use_experts:
            x = self._head_skip(deepest, None)
            x = self.dropout2(x)
            x = self.bn1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout5(x)
            x = self.bn2(x)
            x = self.fc2(x)
            zero_loss = torch.tensor(0.0, device=x.device)
            return x, (zero_loss, zero_loss)

        # Per-expert projection → BN → ReLU → Dropout → LSTM
        expert_outputs = []
        expert_hiddens = []
        for e, lstm in enumerate(self.lstm_experts):
            level_idx = self.expert_levels[e]
            fe = f_levels[level_idx]                      # (B, C_e, T_e)
            fe = fe.permute(0, 2, 1)                      # (B, T_e, C_e)
            fe = self.expert_projections[e](fe)           # (B, T_e, 256)
            fe = self.relu(fe)
            B, Te, Fd = fe.size()
            fe = self.bn0_list[e](fe.contiguous().view(-1, Fd)).view(B, Te, Fd)
            fe = self.dropout2(fe)

            if self.expert_type == 'transformer':
                expert_out_last, expert_hidden = lstm(fe)
                expert_outputs.append(expert_out_last)
                expert_hiddens.append(expert_hidden)
            elif self.expert_type == 'gru':
                gru_out, gru_hidden = lstm(fe)
                # Bidirectional GRU: [forward_last ; backward_first] from the
                # top hidden layer.
                expert_out_last = self._rnn_readout(gru_out, gru_hidden, B)
                expert_outputs.append(expert_out_last)
                # (B, H) -> (1, B, H) so that .squeeze(0) yields (B, H),
                # matching the LSTM hidden-state layout the gate expects.
                expert_hiddens.append(expert_out_last.unsqueeze(0))
            elif self.expert_type == 'lstm_bidir':
                lstm_out, (lstm_hidden, _) = lstm(fe)
                expert_out_last = self._rnn_readout(lstm_out, lstm_hidden, B)
                expert_outputs.append(expert_out_last)
                expert_hiddens.append(expert_out_last.unsqueeze(0))
            else:
                lstm_out, (lstm_hidden, _) = lstm(fe)
                expert_out_last = lstm_out[:, -1, :]   # (B, 64) causal last step
                expert_outputs.append(expert_out_last)
                expert_hiddens.append(expert_out_last.unsqueeze(0))

        # Gated fusion — top-K sparse routing or softmax blend
        if self.use_gate_norm:
            individual_gate = self.fusion_weights_individual(
                self.individual_norm(individual))                            # (B,N)
        else:
            individual_gate = self.fusion_weights_individual(individual)

        if self.feature_gate_type == 'per_expert':
            # Each expert scores itself; concatenate -> (B, N)
            per_expert_scores = []
            for e, gate_linear in enumerate(self.fusion_weights_feature):
                score = gate_linear(expert_hiddens[e].squeeze(0))  # (B, 1)
                per_expert_scores.append(score)
            feature_gate = torch.cat(per_expert_scores, dim=1)    # (B, N)
        elif self.feature_gate_type == 'backbone_gap':
            # Pooled backbone features -> Linear
            pooled = self._feature_gate_gap(deepest).squeeze(-1)   # (B, 512)
            feature_gate = self.fusion_weights_feature(pooled)     # (B, N)
        elif self.feature_gate_type == 'expert_mean':
            # Unweighted mean over the K expert outputs
            gate_in = torch.stack(expert_outputs, dim=1).mean(1)   # (B, H)
            feature_gate = self.fusion_weights_feature(gate_in)    # (B, N)
        elif self.feature_gate_type == 'expert_concat':
            # Concatenation of the K expert outputs
            gate_in = torch.cat(expert_outputs, dim=1)             # (B, K*H)
            feature_gate = self.fusion_weights_feature(gate_in)    # (B, N)
        else:
            # Default: the hidden state of expert 0.
            feature_gate_input = expert_hiddens[0].squeeze(0)      # (B, 64)
            feature_gate = self.fusion_weights_feature(feature_gate_input)  # (B, N)

        fusion = ((1 - self.alpha) * individual_gate +
                   self.alpha * feature_gate)

        if self.top_k is not None and self.top_k < self.expert_num:
            # Top-K sparse routing: only k experts active per sample
            topk_vals, topk_idx = torch.topk(fusion, k=self.top_k, dim=1)
            topk_w = torch.softmax(topk_vals, dim=1)
            weights = torch.zeros_like(fusion).scatter(1, topk_idx, topk_w)
            # Load balance (Switch Transformer): prevent dead experts
            gating_probs = torch.softmax(fusion, dim=1)
            f = (weights > 0).float().mean(dim=0)   # fraction routed
            P = gating_probs.mean(dim=0)             # mean gating prob
            routing_loss = self.expert_num * (f * P).sum()
        else:
            # Dense softmax routing.
            individual_weights = torch.softmax(individual_gate, dim=1)
            feature_weights = torch.softmax(feature_gate, dim=1)
            weights = ((1 - self.alpha) * individual_weights +
                       self.alpha * feature_weights)
            routing_loss = self.feature_expert_consist_loss(individual, weights)
        x = torch.bmm(weights.unsqueeze(1),
                      torch.stack(expert_outputs, dim=1)).squeeze(1)
        x = self._head_skip(deepest, x)                # residual backbone skip

        diversity_loss = self.expert_diversity_loss(expert_outputs)

        # Classifier
        x = self.dropout2(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x, (diversity_loss, routing_loss)

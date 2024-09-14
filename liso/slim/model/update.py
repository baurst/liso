import torch
import torch.nn.functional as F
from torch import nn


class FlowOrClassificationHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, out_dims=2, **kwargs):
        super().__init__(**kwargs)
        assert out_dims in [
            2,
            3,
            4,
        ], "choose out_dims=2 for flow or out_dims=4 for classification or 3 if the paper DL is dangerously close"

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dims, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.conv2(self.relu(self.conv1(inputs)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=304):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        corr_planes = (
            self.cfg.model.corr_cfg.num_levels
            * (2 * self.cfg.model.corr_cfg.search_radius + 1) ** 2
        )
        self.conv_stat_corr1 = nn.Conv2d(corr_planes, 96, 1, padding=0)

        num_stat_flow_head_channels = (
            3
            if self.cfg.model.predict_weight_for_static_aggregation is not False
            else 2
        )

        self.conv_flow1 = nn.Conv2d(num_stat_flow_head_channels, 64, 7, padding=3)
        self.conv_flow2 = nn.Conv2d(
            64,
            32,
            3,
            padding=1,
        )
        self.predict_logits = self.cfg.model.flow_maps_archi != "vanilla"
        if self.predict_logits:
            self.conv_class1 = nn.Conv2d(4, 64, 7, padding=3)
            self.conv_class2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128 + int(self.predict_logits) * 32, 80, 3, padding=1)

    def forward(self, flow, corr, logits):
        corr = F.relu(self.conv_stat_corr1(corr))

        flow = F.relu(self.conv_flow1(flow))
        flow = F.relu(self.conv_flow2(flow))

        concat_vals = [corr, flow]
        if self.predict_logits:
            # without clone: "RuntimeError:
            # one of the variables needed for gradient computation
            # has been modified by an inplace operation"
            logits = F.relu(self.conv_class1(logits))
            logits = F.relu(self.conv_class2(logits))
            concat_vals.append(logits)
        else:
            assert logits is None

        cor_flo_logits = torch.cat(concat_vals, dim=1)
        out = F.relu(self.conv(cor_flo_logits))

        if self.predict_logits:
            return torch.cat([out, logits, flow], dim=1)
        else:
            return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(
        self,
        cfg,
        filters=96,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.filters = filters
        self.predict_logits = self.cfg.model.flow_maps_archi != "vanilla"
        self.motion_encoder = SmallMotionEncoder(
            cfg=cfg,
        )
        self.gru = ConvGRU(
            hidden_dim=filters, input_dim=272 + int(self.predict_logits) * 32
        )

        num_stat_flow_head_channels = (
            3
            if self.cfg.model.predict_weight_for_static_aggregation is not False
            else 2
        )
        self.static_flow_head = FlowOrClassificationHead(
            input_dim=filters, hidden_dim=128, out_dims=num_stat_flow_head_channels
        )
        if self.predict_logits:
            self.classification_head = FlowOrClassificationHead(
                input_dim=filters, hidden_dim=128, out_dims=4
            )
        else:
            self.classification_head = None

    def forward(
        self, net, inp, corr, flow, logits, weight_logits_for_static_aggregation
    ):
        if self.cfg.model.predict_weight_for_static_aggregation:
            motion_features = self.motion_encoder(
                torch.cat([flow, weight_logits_for_static_aggregation], dim=1),
                corr,
                logits,
            )
        else:
            assert weight_logits_for_static_aggregation is None
            motion_features = self.motion_encoder(flow, corr, logits)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)

        if self.cfg.model.predict_weight_for_static_aggregation:
            delta = self.static_flow_head(net)
            delta_static_flow = delta[:, 0:2, ...]
            delta_weights = delta[:, -1:, ...]
        else:
            delta_static_flow = self.static_flow_head(net)
            delta_weights = None

        if self.predict_logits:
            delta_logits = self.classification_head(net)
        else:
            delta_logits = None

        return (
            net,
            delta_static_flow,
            delta_logits,
            delta_weights,
        )

"""Read a TNM1 file written by export_nemo.py back into a NemoSmall model (weights dequantized: int8 * per-row scale).

Used to verify the exported file against the training checkpoint and as the weight source of parity_v2.py --tnm.
Reads the int8 and int4 (blocked16) CTC layouts (transducer flag 0).

python tnm_read.py model.tnm [ckpt.pt]     # prints the header; with a checkpoint also checks the weights match
"""
import struct, sys
import numpy as np, torch
from nemo_small import NemoSmall


class Reader:
    def __init__(self, buf, pos=0):
        self.buf, self.pos = buf, pos

    def align(self, a=16):
        self.pos = (self.pos + a - 1) // a * a

    def f32(self, n):
        self.align()
        x = np.frombuffer(self.buf, dtype="<f4", count=n, offset=self.pos)
        self.pos += 4 * n
        return torch.from_numpy(x.astype(np.float32).copy())

    def qlin(self, n, k, bits=8):
        """-> (dequantized weight (n, k), bias (n), int8 q, scale); rows are padded to a multiple of 16 values.
        int4 (n % 16 == 0, 'blocked16'): byte [n//16][k//2][j] = (q[16b+j][2i] + 8) | (q[16b+j][2i+1] + 8) << 4."""
        self.align()
        kp = (k + 15) // 16 * 16
        if bits == 8:
            q = np.frombuffer(self.buf, dtype=np.int8, count=n * kp, offset=self.pos).reshape(n, kp)[:, :k]
            self.pos += n * kp
        else:
            assert bits == 4 and n % 16 == 0, (bits, n)
            raw = np.frombuffer(self.buf, dtype=np.uint8, count=n * kp // 2, offset=self.pos).reshape(n // 16, kp // 2, 16)
            self.pos += n * kp // 2
            q = np.stack([(raw & 15).astype(np.int8) - 8, (raw >> 4).astype(np.int8) - 8], -1)   # (B, kp/2, 16, 2)
            q = q.transpose(0, 2, 1, 3).reshape(n, kp)[:, :k]
        s = self.f32(n)
        b = self.f32(n)
        return torch.from_numpy(q.astype(np.float32).copy()) * s[:, None], b, torch.from_numpy(q.astype(np.int8).copy()), s


HDR = ["version", "d", "heads", "ff", "k", "layers", "conv_ch", "vocab", "bits", "rnnt", "flags", "chunk", "left",
       "stack", "reserved"]


def read_tnm(path):
    """-> (model, header dict, norm_mean, norm_std, vocab pieces)"""
    buf = open(path, "rb").read()
    assert buf[:4] == b"TNM1", buf[:4]
    h = dict(zip(HDR, struct.unpack("<15I", buf[4:64])))
    assert h["bits"] in (4, 8) and not h["rnnt"], h
    qb = h["bits"]
    d, C, V, nl = h["d"], h["conv_ch"], h["vocab"], h["layers"]
    stack = 2 if h["stack"] == 2 else 1
    cfg = dict(d=d, h=h["heads"], ff=h["ff"], k=h["k"], vocab=V, sub_ch=C, stack=stack, n_layers=nl)
    # the CTC head has V + 1 outputs (blank = V): NemoSmall's vocab counts the real pieces
    m = NemoSmall(cfg=cfg)
    r = Reader(buf, 64)
    with torch.no_grad():
        m.pre.window.copy_(r.f32(400))
        m.pre.fb.copy_(r.f32(257 * 80).view(257, 80).t())
        m.conv0.weight.copy_(r.f32(C * 9).view(C, 1, 3, 3)); m.conv0.bias.copy_(r.f32(C))
        w, b, _, _ = r.qlin(C, C * 9); m.conv2.weight.copy_(w.view(C, C, 3, 3)); m.conv2.bias.copy_(b)
        w, b, _, _ = r.qlin(d, C * 20); m.sub_out.weight.copy_(w); m.sub_out.bias.copy_(b)
        if stack == 2:
            w, b, _, _ = r.qlin(d, 2 * d); m.stack_proj.weight.copy_(w); m.stack_proj.bias.copy_(b)
        for L in m.layers:
            def ln(n):
                n.weight.copy_(r.f32(d)); n.bias.copy_(r.f32(d))
            def lin(l, n, k):
                w, b, _, _ = r.qlin(n, k, qb); l.weight.copy_(w); l.bias.copy_(b)
            ln(L.norm_feed_forward1)
            lin(L.ff1_l1, h["ff"], d); lin(L.ff1_l2, d, h["ff"])
            ln(L.norm_self_att)
            w, b, _, _ = r.qlin(3 * d, d, qb)
            for i, l in enumerate([L.linear_q, L.linear_k, L.linear_v]):
                l.weight.copy_(w[i * d:(i + 1) * d]); l.bias.copy_(b[i * d:(i + 1) * d])
            lin(L.linear_out, d, d)
            w, _, _, _ = r.qlin(d, d); L.linear_pos.weight.copy_(w)
            L.pos_bias_u.copy_(r.f32(d).view(h["heads"], -1)); L.pos_bias_v.copy_(r.f32(d).view(h["heads"], -1))
            ln(L.norm_conv)
            lin(L.pw1, 2 * d, d)
            L.dw.weight.copy_(r.f32(d * h["k"]).view(d, 1, h["k"])); L.dw.bias.copy_(r.f32(d))
            lin(L.pw2, d, d)
            ln(L.norm_feed_forward2)
            lin(L.ff2_l1, h["ff"], d); lin(L.ff2_l2, d, h["ff"])
            ln(L.norm_out)
        w, b, _, _ = r.qlin(V + 1, d); m.head.weight.copy_(w); m.head.bias.copy_(b)
    r.align()
    pieces = []
    for _ in range(V):
        n = buf[r.pos]
        pieces.append(buf[r.pos + 1:r.pos + 1 + n].decode())
        r.pos += 1 + n
    mean = std = None
    if h["flags"] & 1:
        mean, std = r.f32(80), r.f32(80)
    assert r.pos == len(buf), (r.pos, len(buf))
    return m.eval(), h, mean, std, pieces


QUANT = ("conv2", "sub_out", "stack_proj", "head", "ff1_l1", "ff1_l2", "ff2_l1", "ff2_l2", "linear_q", "linear_k",
         "linear_v", "linear_out", "linear_pos", "pw1", "pw2")


if __name__ == "__main__":
    m, h, mean, std, pieces = read_tnm(sys.argv[1])
    print(h, "pieces", len(pieces), "fixed norm", mean is not None)
    if len(sys.argv) > 2:
        from quant_ops import fq_weight
        st = torch.load(sys.argv[2], map_location="cpu")
        LAYER_Q = ("ff1_l1", "ff1_l2", "ff2_l1", "ff2_l2", "linear_q", "linear_k", "linear_v", "linear_out", "pw1", "pw2")
        ref = NemoSmall.from_ckpt(st).eval()
        worst = 0.0
        for (n1, p1), (n2, p2) in zip(m.state_dict().items(), ref.state_dict().items()):
            assert n1 == n2 and p1.shape == p2.shape, (n1, n2)
            parts = n1.split(".")
            if parts[-1] == "weight" and (parts[0] in QUANT or (parts[0] == "layers" and parts[2] in QUANT)):
                bits = h["bits"] if (parts[0] == "layers" and parts[2] in LAYER_Q) else 8
                p2 = fq_weight(p2.flatten(1), bits).view_as(p2)    # the exporter's per-output-row grid
            e = (p1 - p2).abs().max().item()
            worst = max(worst, e)
            if e > 1e-5:
                print("  differs", n1, e)
        if mean is not None:
            worst = max(worst, (mean - st["norm_mean"]).abs().max().item(), (std - st["norm_std"]).abs().max().item())
        print("max |tnm - checkpoint (on its quantization grid)| over all tensors:", worst)

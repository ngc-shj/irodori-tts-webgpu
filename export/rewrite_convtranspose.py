#!/usr/bin/env python3
"""Rewrite the decoder's ConvTranspose layers as Conv + pixel-shuffle.

onnxruntime-web's WebGPU fp16 ConvTranspose kernel is broken (produces noise);
its fp16 Conv kernel is fine. ConvTranspose dominates decode cost, so dodging it
with fp32 gives no speedup. Here we replace each ConvTranspose with a
mathematically identical sub-pixel (polyphase) form built from Conv only, so the
whole decoder can run fp16 and actually go fast.

All four decoder ConvTransposes share the regular shape kernel==2*stride,
pad==stride/2, output_padding==0. For that case a stride-s upsample equals a
kernel-3 Conv producing Cout*s channels (the s polyphase filters, third tap zero),
followed by pixel-shuffle (reshape/transpose/reshape) to length Lin*s. Verified
exact vs the original ConvTranspose (corr 1.0).

The (weight-normalised) ConvTranspose weights are read from the running fp32 graph
and baked as constants, so the per-layer weight-norm subgraph drops out too.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def probe_weights(model_path: str, ct_nodes) -> dict[str, np.ndarray]:
    """Run the fp32 graph once to capture each ConvTranspose's runtime weight+bias."""
    m = onnx.load(model_path)
    g = m.graph
    existing = {o.name for o in g.output}
    wanted = set()
    for ct in ct_nodes:
        wanted.add(ct.input[1])  # normalised weight (Cin, Cout, k)
        wanted.add(ct.input[2])  # bias
    for nm in wanted:
        if nm not in existing:
            g.output.append(helper.ValueInfoProto(name=nm))
    onnx.save(m, "/tmp/_ct_probe.onnx", save_as_external_data=True,
              all_tensors_to_one_file=True, location="_ct_probe.onnx.data")
    sess = ort.InferenceSession("/tmp/_ct_probe.onnx", providers=["CPUExecutionProvider"])
    z = np.random.randn(1, 32, 64).astype(np.float32)  # any frames; weights are input-independent
    outs = sess.run(list(wanted), {"z": z})
    return dict(zip(wanted, outs))


def subpixel_weight(W: np.ndarray, s: int, p: int) -> np.ndarray:
    """W (Cin,Cout,k=2s), p=s/2 -> Wsp (Cout*s, Cin, 3) kernel-3 Conv weight.

    out[q*s+r] = center(x[q]) + left(x[q-1], r<s/2) or right(x[q+1], r>=s/2).
    """
    Cin, Cout, k = W.shape
    h = s // 2
    Wsp = np.zeros((Cout * s, Cin, 3), np.float32)
    for co in range(Cout):
        for r in range(s):
            Wsp[co * s + r, :, 1] = W[:, co, r + h]          # center: x[q]
            if r < h:
                Wsp[co * s + r, :, 0] = W[:, co, s + r + h]   # left: x[q-1]
            else:
                Wsp[co * s + r, :, 2] = W[:, co, r - h]       # right: x[q+1]
    return Wsp


def dce(graph: onnx.GraphProto) -> None:
    """Drop nodes/initializers not feeding any graph output."""
    producer = {o: n for n in graph.node for o in n.output if o}
    needed, stack = set(), [o.name for o in graph.output]
    while stack:
        t = stack.pop()
        if t in needed:
            continue
        needed.add(t)
        n = producer.get(t)
        if n:
            stack.extend(i for i in n.input if i)
    keep = [n for n in graph.node if any(o in needed for o in n.output)]
    del graph.node[:]
    graph.node.extend(keep)
    inits = [i for i in graph.initializer if i.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(inits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/onnx/dacvae_decoder.onnx")
    ap.add_argument("--out", default="artifacts/onnx/dacvae_decoder_subpix.onnx")
    args = ap.parse_args()

    model = onnx.load(args.inp)
    g = model.graph
    ct_nodes = [n for n in g.node if n.op_type == "ConvTranspose"]
    print(f"[rewrite] {len(ct_nodes)} ConvTranspose -> Conv+pixel-shuffle")
    weights = probe_weights(args.inp, ct_nodes)

    new_nodes, new_inits = [], []
    drop = set()
    for i, ct in enumerate(ct_nodes):
        a = {at.name: (list(at.ints) if at.ints else at.i) for at in ct.attribute}
        s = a["strides"][0]; p = a["pads"][0]
        op = a.get("output_padding", [0])[0]
        data_in, w_in, b_in = ct.input
        W = weights[w_in]; b = weights[b_in]
        Cin, Cout, k = W.shape
        if not (k == 2 * s and p == s // 2 and op == 0):
            raise ValueError(f"{ct.name}: unsupported shape k={k} s={s} p={p} op={op}")

        Wsp = subpixel_weight(W, s, p)
        wname = f"subpix_w_{i}"; bname = f"subpix_b_{i}"
        s1name = f"subpix_s1_{i}"; s2name = f"subpix_s2_{i}"
        new_inits += [
            numpy_helper.from_array(Wsp, wname),
            numpy_helper.from_array(b.reshape(1, Cout, 1).astype(np.float32), bname),
            numpy_helper.from_array(np.array([1, Cout, s, -1], np.int64), s1name),
            numpy_helper.from_array(np.array([1, Cout, -1], np.int64), s2name),
        ]
        conv_o = f"subpix_conv_{i}"; r1 = f"subpix_r1_{i}"; t1 = f"subpix_t1_{i}"; r2 = f"subpix_r2_{i}"
        new_nodes += [
            helper.make_node("Conv", [data_in, wname], [conv_o], name=f"subpix_conv_n_{i}",
                             strides=[1], pads=[1, 1], dilations=[1], group=1),
            helper.make_node("Reshape", [conv_o, s1name], [r1], name=f"subpix_re1_{i}"),
            helper.make_node("Transpose", [r1], [t1], perm=[0, 1, 3, 2], name=f"subpix_tr_{i}"),
            helper.make_node("Reshape", [t1, s2name], [r2], name=f"subpix_re2_{i}"),
            helper.make_node("Add", [r2, bname], [ct.output[0]], name=f"subpix_add_{i}"),
        ]
        drop.add(ct.name)

    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    keep = [n for n in g.node if n.name not in drop]
    del g.node[:]; g.node.extend(keep)
    dce(g)

    # topological sort (new nodes were appended; ORT needs def-before-use)
    avail = {i.name for i in g.initializer} | {i.name for i in g.input}
    ordered, pending = [], list(g.node)
    while pending:
        progress = False
        rest = []
        for n in pending:
            if all((i in avail or i == "") for i in n.input):
                ordered.append(n); avail.update(n.output); progress = True
            else:
                rest.append(n)
        pending = rest
        if not progress:
            raise RuntimeError(f"cycle / missing inputs for {[n.name for n in pending][:5]}")
    del g.node[:]; g.node.extend(ordered)

    out_path = Path(args.out)
    data_path = out_path.parent / f"{out_path.stem}.onnx.data"
    if data_path.exists():
        data_path.unlink()
    onnx.save(model, str(out_path), save_as_external_data=True,
              all_tensors_to_one_file=True, location=f"{out_path.stem}.onnx.data")
    print(f"[rewrite] -> {out_path}")

    # verify exact vs original
    z = np.fromfile("artifacts/ref/final_latent.bin", dtype=np.float32).reshape(1, 139, 32)
    z = np.ascontiguousarray(z.transpose(0, 2, 1))
    a = ort.InferenceSession(args.inp, providers=["CPUExecutionProvider"]).run(["audio"], {"z": z})[0].flatten()
    bb = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"]).run(["audio"], {"z": z})[0].flatten()
    n = min(len(a), len(bb))
    corr = float(np.dot(a[:n], bb[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(bb[:n]) + 1e-12))
    print(f"[verify] corr(orig, subpix)={corr:.6f}  max|Δ|={np.abs(a[:n]-bb[:n]).max():.3e}  len {len(a)} vs {len(bb)}")


if __name__ == "__main__":
    main()

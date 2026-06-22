from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class UltralyticsReIDWrapper(torch.nn.Module):
    """Adapt FastReID's 0-255 input expectation to Ultralytics' 0-1 ReID crop tensors."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images * 255.0)
        return F.normalize(features, p=2, dim=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a FastReID checkpoint as an Ultralytics-compatible ONNX encoder")
    parser.add_argument("--fastreid-root", type=str, default="external/fastreid", help="Path to FastReID checkout")
    parser.add_argument("--config-file", type=str, default="configs/fastreid/vehicle_bootstrap_bagtricks.yml")
    parser.add_argument("--weights", type=str, default="weights/fastreid_bootstrap_smoke/model_final.pth")
    parser.add_argument("--output", type=str, default="weights/fastreid_bootstrap_smoke/fastreid_vehicle_reid.onnx")
    parser.add_argument(
        "--torchscript-output",
        type=str,
        default=None,
        help="Optional .torchscript output path for Ultralytics' PyTorch backend",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Square crop size expected by Ultralytics ReID loader")
    parser.add_argument("--batch-size", type=int, default=1, help="Static export batch size")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fastreid_root = Path(args.fastreid_root).resolve()
    sys.path.insert(0, str(fastreid_root.parent))

    from fast_reid.fastreid.config import get_cfg
    from fast_reid.fastreid.modeling.meta_arch import build_model
    from fast_reid.fastreid.utils.checkpoint import Checkpointer

    config_file = Path(args.config_file).resolve()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not weights.exists():
        raise FileNotFoundError(f"FastReID weights not found: {weights}")

    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = "cpu"
    cfg.INPUT.SIZE_TEST = [args.imgsz, args.imgsz]
    cfg.MODEL.HEADS.NUM_CLASSES = _checkpoint_num_classes(weights) or cfg.MODEL.HEADS.NUM_CLASSES
    if cfg.MODEL.HEADS.POOL_LAYER == "FastGlobalAvgPool":
        cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
    cfg.freeze()

    model = build_model(cfg)
    Checkpointer(model).load(str(weights))
    model.eval()

    wrapper = UltralyticsReIDWrapper(model).eval()
    dummy = torch.randn(args.batch_size, 3, args.imgsz, args.imgsz, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            input_names=["images"],
            output_names=["embeddings"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"[INFO] ONNX ReID encoder exported: {output}")
    print(f"[INFO] Input shape : {tuple(dummy.shape)}")
    print("[INFO] Output      : L2-normalized embedding tensor")
    if args.torchscript_output:
        torchscript_output = Path(args.torchscript_output).resolve()
        torchscript_output.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(wrapper, dummy)
        traced.save(str(torchscript_output))
        print(f"[INFO] TorchScript ReID encoder exported: {torchscript_output}")
    return 0


def _checkpoint_num_classes(weights: Path) -> int | None:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    head_weight = state_dict.get("heads.weight")
    if head_weight is None:
        return None
    return int(head_weight.shape[0])


if __name__ == "__main__":
    raise SystemExit(main())

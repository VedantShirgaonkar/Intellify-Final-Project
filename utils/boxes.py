import torch

def rescale_bboxes(bboxes, size_hw):
    # Expect incoming boxes as normalized cxcywh (0..1), convert to xyxy in pixels of provided size (H,W)
    H, W = size_hw
    if hasattr(bboxes, 'detach'):
        b = bboxes.detach().cpu()
    else:
        b = torch.tensor(bboxes)
    cxcywh = b
    cx = cxcywh[..., 0] * W
    cy = cxcywh[..., 1] * H
    bw = cxcywh[..., 2] * W
    bh = cxcywh[..., 3] * H
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

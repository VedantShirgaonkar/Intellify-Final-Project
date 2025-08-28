import torch 

def box_cxcywh_to_xyxy(x):
    """Converts box coordinates from center x, center y, width height to top left, bottom right xy xy."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """Convert bbox from (x1, y1, x2, y2) to (center_x, center_y, width, height)"""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    """Scales boxes to output size"""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_area(boxes):
    """Compute area of boxes in xyxy format"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    eps = 1e-7
    iou = inter / (union + eps)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized IoU between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    """
    # Compute regular IoU
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    c_area = wh[:, :, 0] * wh[:, :, 1]
    eps = 1e-7
    return iou - (c_area - union) / (c_area + eps)


def stacker(batch):
    """
    Custom collate function for DETR.
    
    Handles batches with varying numbers of objects per image.
    The images get stacked, but targets remain as a list since
    each image can have a different number of objects.
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack all images into a single tensor
    # This works because all images have the same size after transforms
    images = torch.stack(images, dim=0)
    
    # Keep targets as a list - each element corresponds to one image
    # This allows each image to have a different number of objects
    return images, targets
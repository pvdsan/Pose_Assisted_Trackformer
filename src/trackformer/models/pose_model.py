import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class PoseEmbeddingModule(nn.Module):
    """
    Pose Embedding Module using a pretrained Keypoint R-CNN model.

    This module:
    - Takes a batch of images and corresponding bounding boxes.
    - Uses a pretrained Keypoint R-CNN model to detect persons and keypoints in the entire image.
    - Maps predicted boxes to the given target boxes.
    - Converts the resulting keypoints into fixed-size embeddings using a linear layer.
    """

    def __init__(self, pose_embedding_dim: int = 256,
                 bbox_format: str = 'xyxy', pose_threshold: float = 0.3):
        super(PoseEmbeddingModule, self).__init__()
        self.pose_embedding_dim = pose_embedding_dim
        self.bbox_format = bbox_format
        self.pose_threshold = pose_threshold

        # Number of keypoints predicted by Keypoint R-CNN on COCO dataset (17 keypoints)
        self.num_joints = 17

        # Load the pretrained Keypoint R-CNN model
        self.model = keypointrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # set to eval mode

        # Linear layer to convert keypoints to embeddings
        self.keypoint_linear = nn.Linear(self.num_joints * 3, pose_embedding_dim)
        nn.init.xavier_uniform_(self.keypoint_linear.weight)
        nn.init.constant_(self.keypoint_linear.bias, 0)

    def forward(self, images, targets):
        """
        Forward pass to extract pose embeddings.

        Args:
            images (NestedTensor): images.tensors is a [B, 3, H, W] Tensor of uint8 in [0, 255].
            targets (List[Dict]): Each element contains 'boxes': Tensor [N,4] in cxcywh format.

        Returns:
            pose_embeddings (List[torch.Tensor]): List of [N, pose_embedding_dim] per image.
            keypoints_out (List[np.ndarray]): Corresponding keypoints for inspection.
        """
        images_tensor = images.tensors
        device = self.keypoint_linear.weight.device
        batch_size = images_tensor.size(0)
        pose_embeddings = []
        keypoints_out = []


        # Run Keypoint R-CNN on the entire batch
        # The model expects a list of images in [C,H,W], each normalized, CPU or GPU.
        image_list = [img.to(device) for img in images_tensors]
        with torch.no_grad():
            predictions = self.model(image_list)  # List of dicts

        bbox_list = [t['boxes'] for t in targets]

        for idx in range(batch_size):
            # Get image dimensions
            _, img_h, img_w = images_tensor[idx].shape
            boxes_cxcywh = bbox_list[idx].to(device)
            boxes_abs = self.cxcywh_to_xyxy(boxes_cxcywh, img_w, img_h)

            pred_boxes = predictions[idx]['boxes']
            pred_keypoints = predictions[idx]['keypoints']  # [N, 17, 3]
            pred_scores = predictions[idx]['scores']

            # Filter out low-confidence predictions
            keep = pred_scores > self.pose_threshold
            pred_boxes = pred_boxes[keep]
            pred_keypoints = pred_keypoints[keep]

            # For each target box, find the best matching predicted box by IoU
            keypoints_list = []
            for gt_box in boxes_abs:
                matched_kp = self.match_and_extract_keypoints(gt_box, pred_boxes, pred_keypoints)
                keypoints_list.append(matched_kp)

            if len(keypoints_list) == 0:
                # If no boxes are present, append an empty tensor
                pose_embed = torch.empty(0, self.pose_embedding_dim, device=device)
            else:
                # Convert keypoints to torch tensor
                # keypoints_list is [N, num_joints, 3]
                # Flatten to [N, num_joints * 3]
                keypoints_np = np.array(keypoints_list, dtype=np.float32).reshape(len(keypoints_list), -1)
                keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32, device=device)

                # Generate pose embeddings
                pose_embed = self.keypoint_linear(keypoints_tensor)  # [N, pose_embedding_dim]

            pose_embeddings.append(pose_embed)
            keypoints_out.append(keypoints_list)

        return pose_embeddings, keypoints_out

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        """
        Converts bounding boxes from [cx, cy, w, h] to [x_min, y_min, x_max, y_max].
        """
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_min = (cx - 0.5 * w) * img_w
        y_min = (cy - 0.5 * h) * img_h
        x_max = (cx + 0.5 * w) * img_w
        y_max = (cy + 0.5 * h) * img_h
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    @staticmethod
    def box_iou(box1, box2):
        """
        Compute IoU between box1 and box2.
        box1, box2: [4] tensors [x_min, y_min, x_max, y_max]
        """
        # Intersection
        inter_xmin = torch.max(box1[0], box2[:, 0])
        inter_ymin = torch.max(box1[1], box2[:, 1])
        inter_xmax = torch.min(box1[2], box2[:, 2])
        inter_ymax = torch.min(box1[3], box2[:, 3])

        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        iou = inter_area / (area1 + area2 - inter_area)
        return iou

    def match_and_extract_keypoints(self, gt_box, pred_boxes, pred_keypoints):
        """
        For a given ground-truth box, find the predicted box with the highest IoU and extract its keypoints.
        If no match is found (IoU < threshold), return zeros.
        """
        if len(pred_boxes) == 0:
            # No predictions available
            return np.zeros((self.num_joints, 3), dtype=np.float32)

        ious = self.box_iou(gt_box, pred_boxes)
        best_idx = torch.argmax(ious)
        best_iou = ious[best_idx]

        # You can set a minimum IoU threshold if desired
        if best_iou < 0.1:
            # No good match found
            return np.zeros((self.num_joints, 3), dtype=np.float32)

        # Extract keypoints from the best matched prediction
        kp = pred_keypoints[best_idx].cpu().numpy()  # shape: [17, 3] (x, y, visibility)
        # Keypoint R-CNN keypoints are absolute coordinates in the image.
        # If you need them normalized, you can divide by image width/height.
        return kp


def build_pose_model(args) -> PoseEmbeddingModule:
    """
    Builds and returns the PoseEmbeddingModule using Keypoint R-CNN.
    """
    return PoseEmbeddingModule(
        pose_embedding_dim=args.pose_embedding_dim,
        bbox_format=args.bbox_format,
        pose_threshold=args.pose_threshold
    )

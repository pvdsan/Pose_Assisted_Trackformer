import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be unnormalized.
        Returns:
            Tensor: Unnormalized image.
        """
        output_tensor = []
        for t, m, s in zip(tensor, self.mean, self.std):
            output_tensor.append(t.mul(s).add(m))
        return torch.stack(output_tensor, dim=0)

class PoseEmbeddingModule(nn.Module):
    """
    Pose Embedding Module

    This module extracts pose embeddings from a batch of images and their corresponding bounding boxes
    using MediaPipe's Pose solution. It converts the extracted keypoints into fixed-size
    pose embeddings using a linear layer.

    Args:
        pose_embedding_dim (int): Dimension of the output pose embeddings.
        bbox_format (str): Format of bounding boxes ('xyxy' or 'cxcywh').
        pose_threshold (float): Confidence threshold for pose keypoints.
    """

    def __init__(self, pose_embedding_dim: int = 256,
                 bbox_format: str = 'xyxy', pose_threshold: float = 0.3):
        super(PoseEmbeddingModule, self).__init__()
        self.pose_embedding_dim = pose_embedding_dim
        self.bbox_format = bbox_format
        self.pose_threshold = pose_threshold

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True,
                                      model_complexity=1,
                                      enable_segmentation=False,
                                      min_detection_confidence=pose_threshold)

        # Define the number of keypoints based on MediaPipe's Pose
        self.num_joints = 33  # MediaPipe Pose provides 33 keypoints

        # Linear layer to convert keypoints to embeddings
        self.keypoint_linear = nn.Linear(self.num_joints * 3, pose_embedding_dim)
        nn.init.xavier_uniform_(self.keypoint_linear.weight)
        nn.init.constant_(self.keypoint_linear.bias, 0)
        
        # UnNormalize instance
        self.unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, images, targets):
        """
        Forward pass to extract pose embeddings.

        Args:
            images (NestedTensor): images.tensors is a [B, 3, H, W] Tensor of uint8 in [0, 255].
            targets (List[Dict]): Each element contains 'boxes': Tensor [N,4] in specified format.

        Returns:
            List[torch.Tensor]: List of pose embeddings for each image in the batch.
                                Each tensor has shape [N, pose_embedding_dim].
            List[np.ndarray]: Corresponding keypoints for inspection.
        """
        images_tensor = images.tensors
        device = self.keypoint_linear.weight.device
        batch_size = images_tensor.size(0)
        pose_embeddings = []
        keypoints_out = []

        # Unnormalize and scale images
        for idx in range(batch_size):
            image = images_tensor[idx]  # Shape: [3, H, W]
            image = self.unnorm(image)
            image = (image * 255).clamp(0, 255).to(torch.uint8)
            images_tensor[idx] = image
            
        # Convert images to CPU for MediaPipe processing
        image_list_cpu = [img.cpu() for img in images_tensor]

        bbox_list = [t['boxes'] for t in targets]

        for idx in range(batch_size):
            # Extract the image and corresponding boxes
            image_tensor = image_list_cpu[idx]  # Shape: [3, H, W]

            # Extract box
            box = bbox_list[idx]

            # Convert image tensor to NumPy array in [H, W, 3] format
            image_np = image_tensor.permute(1, 2, 0).numpy()  # [H, W, 3]

            # Get image dimensions
            img_h, img_w, _ = image_np.shape

            # Handle different bbox formats
            if self.bbox_format == 'cxcywh':
                boxes_abs = self.cxcywh_to_xyxy(box, img_w, img_h)  # [N, 4]
                boxes_abs = boxes_abs.cpu().numpy().tolist()  # List of [x_min, y_min, x_max, y_max]
            elif self.bbox_format == 'xyxy':
                boxes_abs = box.cpu().numpy().tolist()  # Assuming box is already [x_min, y_min, x_max, y_max]
            else:
                raise ValueError(f"Unsupported bbox_format: {self.bbox_format}")

            # Initialize list to hold keypoints for each box
            keypoints = []

            # Iterate over each bounding box
            for box_abs in boxes_abs:
                x_min, y_min, x_max, y_max = box_abs

                # Ensure coordinates are within image bounds
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(img_w, int(x_max))
                y_max = min(img_h, int(y_max))

                # Crop the image to the bounding box
                cropped_img_np = image_np[y_min:y_max, x_min:x_max]

                # Check if the cropped image has non-zero area
                if cropped_img_np.size == 0 or cropped_img_np.shape[0] == 0 or cropped_img_np.shape[1] == 0:
                    kp = np.zeros(self.num_joints * 3, dtype=np.float32)
                else:
                    # Perform pose detection on the cropped image
                    result = self.pose.process(cropped_img_np)

                    if result.pose_landmarks:
                        # Extract keypoints and map to original image coordinates
                        kp = []
                        for lm in result.pose_landmarks.landmark:
                            # Scale and translate keypoints to original image
                            absolute_x = x_min + lm.x * (x_max - x_min)
                            absolute_y = y_min + lm.y * (y_max - y_min)
                            kp.append(absolute_x)
                            kp.append(absolute_y)
                            kp.append(lm.visibility)
                        kp = np.array(kp, dtype=np.float32)
                    else:
                        # If no landmarks detected, fill with zeros
                        kp = np.zeros(self.num_joints * 3, dtype=np.float32)

                keypoints.append(kp)

            if len(keypoints) == 0:
                # If no boxes are present, append an empty tensor
                pose_embed = torch.empty(0, self.pose_embedding_dim, device=device)
            else:
                # Convert keypoints to torch tensor
                keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32, device=device)  # [N, num_joints * 3]

                # Generate pose embeddings
                pose_embed = self.keypoint_linear(keypoints_tensor)  # [N, pose_embedding_dim]

            pose_embeddings.append(pose_embed)
            keypoints_out.append(keypoints)

        return pose_embeddings, keypoints_out  # List of [N, pose_embedding_dim] tensors

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        """
        Converts bounding boxes from [cx, cy, w, h] to [x_min, y_min, x_max, y_max].

        Args:
            boxes (torch.Tensor): Tensor of shape [N, 4].
            img_w (int): Width of the image.
            img_h (int): Height of the image.

        Returns:
            torch.Tensor: Converted boxes of shape [N, 4].
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
        return kp

def build_pose_model(args) -> PoseEmbeddingModule:
    """
    Builds and returns the PoseEmbeddingModule based on the provided arguments.

    Args:
        args (Namespace or dict): Arguments containing configuration for the pose model.
            Required fields:
                - pose_embedding_dim (int): Dimension of the output pose embeddings.
                - bbox_format (str): Format of bounding boxes ('xyxy' or 'cxcywh').
                - pose_threshold (float): Confidence threshold for pose keypoints.

    Returns:
        PoseEmbeddingModule: Initialized pose embedding module.
    """
    return PoseEmbeddingModule(
        pose_embedding_dim=args.pose_embedding_dim,
        bbox_format=args.bbox_format,
        pose_threshold=args.pose_threshold
    )

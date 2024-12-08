import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


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

    def forward(self, images: torch.Tensor, boxes: torch.Tensor):
        """
        Forward pass to extract pose embeddings.

        Args:
            images (torch.Tensor): Batch of images as tensors with shape [B, 3, H, W].
                                   Pixel values should be in [0, 255] and dtype=torch.uint8.
            boxes (torch.Tensor): Batch of bounding boxes as tensors with shape [B, N, 4].
                                  Each box is [x_min, y_min, x_max, y_max] or [cx, cy, w, h],
                                  depending on `bbox_format`.

        Returns:
            List[torch.Tensor]: List of pose embeddings for each image in the batch.
                                Each tensor has shape [N, pose_embedding_dim].
        """
        device = self.keypoint_linear.weight.device
        batch_size = images.size(0)
        pose_embeddings = []

        # Iterate over each image in the batch
        for idx in range(batch_size):
            # Extract the image and corresponding boxes
            image_tensor = images[idx]  # Shape: [3, H, W]

            # Convert image tensor to NumPy array in [H, W, 3] format
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

            # Get image dimensions
            img_h, img_w, _ = image_np.shape

            boxes_abs = self.cxcywh_to_xyxy(boxes, img_w, img_h)  # [N, 4]


            # Move boxes to CPU and convert to list of lists
            boxes_abs = boxes_abs.cpu().numpy().tolist()  # List of [x_min, y_min, x_max, y_max]

            # Initialize list to hold keypoints for each box
            keypoints = []

            # Iterate over each bounding box
            for box in boxes_abs:
                print(box.shape)
                x_min, y_min, x_max, y_max = box

                # Ensure coordinates are within image bounds
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(img_w, int(x_max))
                y_max = min(img_h, int(y_max))

                # Crop the image to the bounding box
                cropped_img_np = image_np[y_min:y_max, x_min:x_max]

                # Check if the cropped image has non-zero area
                if cropped_img_np.size == 0:
                    kp = np.zeros(self.num_joints * 3)
                else:
                    # Perform pose detection on the cropped image
                    result = self.pose.process(cropped_img_np)

                    if result.pose_landmarks:
                        # Extract keypoints
                        kp = []
                        for lm in result.pose_landmarks.landmark:
                            kp.append(lm.x)         # Normalized [0,1] within the cropped image
                            kp.append(lm.y)
                            kp.append(lm.visibility)
                        kp = np.array(kp)
                    else:
                        # If no landmarks detected, fill with zeros
                        kp = np.zeros(self.num_joints * 3)

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

        return pose_embeddings, keypoints  # List of [N, pose_embedding_dim] tensors

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
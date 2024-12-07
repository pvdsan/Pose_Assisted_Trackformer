import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image
import numpy as np


class PoseEmbeddingModule(nn.Module):
    """
    Pose Embedding Module

    This module extracts pose embeddings from a list of bounding boxes in a batch of images
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

    def forward(self, images: list, boxes: list) -> list:
        """
        Forward pass to extract pose embeddings.

        Args:
            images (List[PIL.Image.Image] or List[np.ndarray]): List of images in the batch.
            boxes (List[torch.Tensor]): List of bounding boxes for each image.
                                        Each tensor is of shape [num_boxes, 4].

        Returns:
            List[torch.Tensor]: List of pose embeddings for each image.
                                Each tensor is of shape [num_boxes, pose_embedding_dim].
        """
        device = next(self.keypoint_linear.parameters()).device
        pose_embeddings = []

        for img, boxes_i in zip(images, boxes):
            img_w, img_h = img.size  # Assumes PIL.Image; modify if using different formats

            # Convert boxes to [x_min, y_min, x_max, y_max] if necessary
            if self.bbox_format == 'cxcywh':
                boxes_i = self.cxcywh_to_xyxy(boxes_i, img_w, img_h)
            else:
                boxes_i = boxes_i * torch.tensor([img_w, img_h, img_w, img_h],
                                                device=boxes_i.device).unsqueeze(0)

            # Convert boxes to list of lists for MediaPipe
            boxes_i = boxes_i.cpu().numpy().tolist()

            # Perform pose inference using MediaPipe
            pose_results = self.pose.process(np.array(img))

            # Initialize list to hold keypoints for each box
            keypoints = []

            for box in boxes_i:
                x_min, y_min, x_max, y_max = box
                # Crop the image to the bounding box
                cropped_img = img.crop((x_min, y_min, x_max, y_max))

                # Convert PIL Image to RGB if necessary
                if cropped_img.mode != 'RGB':
                    cropped_img = cropped_img.convert('RGB')

                # Convert image to numpy array
                cropped_img_np = np.array(cropped_img)

                # Perform pose detection on the cropped image
                result = self.pose.process(cropped_img_np)

                if result.pose_landmarks:
                    # Extract keypoints
                    kp = []
                    for lm in result.pose_landmarks.landmark:
                        kp.append(lm.x)  # Normalized to [0,1]
                        kp.append(lm.y)
                        kp.append(lm.visibility)
                    kp = np.array(kp)
                else:
                    # If no landmarks detected, fill with zeros
                    kp = np.zeros(self.num_joints * 3)

                keypoints.append(kp)

            # Convert keypoints to torch tensor
            keypoints = torch.tensor(keypoints).float().to(device)  # Shape: [num_boxes, num_joints * 3]

            # Generate pose embeddings
            pose_embed = self.keypoint_linear(keypoints)  # Shape: [num_boxes, pose_embedding_dim]
            pose_embeddings.append(pose_embed)

        return pose_embeddings  # List of [num_boxes, pose_embedding_dim]

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        """
        Converts bounding boxes from [cx, cy, w, h] to [x_min, y_min, x_max, y_max].

        Args:
            boxes (torch.Tensor): Tensor of shape [num_boxes, 4].
            img_w (int): Width of the image.
            img_h (int): Height of the image.

        Returns:
            torch.Tensor: Converted boxes of shape [num_boxes, 4].
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
    # Extract arguments
    pose_embedding_dim = args.pose_embedding_dim
    bbox_format = args.bbox_format
    pose_threshold = args.pose_threshold

    # Initialize the PoseEmbeddingModule
    pose_embedding_module = PoseEmbeddingModule(
        pose_embedding_dim=pose_embedding_dim,
        bbox_format=bbox_format,
        pose_threshold=pose_threshold
    )

    return pose_embedding_module
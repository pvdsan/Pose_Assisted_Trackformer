# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..util import box_ops
from ..util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from .detr import DETR, PostProcess, SetCriterion


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(DETR):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, pose_model, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, overflow_boxes=False,multi_frame_attention=False, multi_frame_encoding=False, merge_frame_features=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            pose_model=pose_model,  # Map pose_model to the correct parameter
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=aux_loss,
            overflow_boxes=overflow_boxes
        )
        self.merge_frame_features = merge_frame_features
        self.multi_frame_attention = multi_frame_attention
        self.multi_frame_encoding = multi_frame_encoding
        self.overflow_boxes = overflow_boxes
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
            
        # Learned fusion layer for track and pose embeddings. This maps from (2*D) -> D
        self.track_pose_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones_like(self.class_embed.bias) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # region proposal generation
        num_pred = transformer.decoder.num_layers

        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        if self.merge_frame_features:
            self.merge_features = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
            self.merge_features = _get_clones(self.merge_features, num_feature_levels)


    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None, training = True):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        features_all = features
        features = features[-3:]


        if prev_features is None:
            prev_features = features
        else:
            prev_features = prev_features[-3:]

        src_list = []
        mask_list = []
        pos_list = []


        frame_features = [prev_features, features]
        if not self.multi_frame_attention:
            frame_features = [features]

        for frame, frame_feat in enumerate(frame_features):
            if self.multi_frame_attention and self.multi_frame_encoding:
                pos_list.extend([p[:, frame] for p in pos[-3:]])
            else:
                pos_list.extend(pos[-3:])
            for l, feat in enumerate(frame_feat):
                src, mask = feat.decompose()

                if self.merge_frame_features:
                    prev_src, _ = prev_features[l].decompose()
                    src_list.append(self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1)))
                else:
                    src_list.append(self.input_proj[l](src))

                mask_list.append(mask)


                assert mask is not None

            if self.num_feature_levels > len(frame_feat):
                _len_srcs = len(frame_feat)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:

                        if self.merge_frame_features:
                            src = self.merge_features[l](torch.cat([self.input_proj[l](frame_feat[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                        else:
                            src = self.input_proj[l](frame_feat[-1].tensors)
                    else:
                        src = self.input_proj[l](src_list[-1])
                    _, m = frame_feat[0].decompose()
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    src_list.append(src)
                    mask_list.append(mask)
                    if self.multi_frame_attention and self.multi_frame_encoding:
                        pos_list.append(pos_l[:, frame])
                    else:
                        pos_list.append(pos_l)

        query_embeds = None
        query_embeds = self.query_embed.weight
        
        if training:
            if targets is not None:
                prev_samples = [t['prev_image'] for t in targets]
                prev_boxes   = [t['prev_target']['boxes'] for t in targets]
                
                pose_embeddings, _ = self.pose_model(prev_samples, prev_boxes)
                
                # Select pose embeddings corresponding to track queries
                for i, target in enumerate(targets):
                    if 'track_query_match_ids' in target:
                        matched_ids = target['track_query_match_ids']  # Tensor of indices
                        if matched_ids.numel() > 0:
                            # Ensure matched_ids are within bounds
                            matched_ids = matched_ids[matched_ids < pose_embeddings[i].size(0)]
                            selected_pose_embeds = pose_embeddings[i][matched_ids]
                            target['track_query_pose_embeds'] = selected_pose_embeds
                        else:
                            # No matched tracks
                            target['track_query_pose_embeds'] = torch.empty(0, pose_embeddings[i].size(1), device=pose_embeddings[i].device)
                    else:
                        # If no track queries, assign empty tensor
                        target['track_query_pose_embeds'] = torch.empty(0, pose_embeddings[i].size(1), device=pose_embeddings[i].device)
                        
                        num_tracks = target['track_query_hs_embeds'].shape[0]
                        
                        # Create a mask for matched track queries
                        mask = target['track_queries_mask']  # Shape: [num_tracks]
                        
                        # Initialize a tensor for pose embeddings, filled with zeros
                        pose_padded = torch.zeros_like(target['track_query_hs_embeds'])
                        
                        # Assign the pose embeddings to the corresponding track queries
                        pose_padded[mask] = target['track_query_pose_embeds']
                        
                        # Concatenate track embeddings with pose embeddings
                        concatenated = torch.cat([target['track_query_hs_embeds'], pose_padded], dim=1)  # Shape: [num_tracks, 2*hidden_dim]
                        
                        # Pass through the fusion layer
                        fused_embeds = self.track_pose_fusion(concatenated)  # Shape: [num_tracks, hidden_dim]
                        
                        # Update the track_query_hs_embeds with the fused embeddings
                        target['track_query_hs_embeds'] = fused_embeds
    
        
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(src_list, mask_list, pos_list, query_embeds, targets)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        for src in src_list:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width

        memory = memory_slices
        return out, targets, features_all, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DeformablePostProcess(PostProcess):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        scores, labels = prob.max(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results

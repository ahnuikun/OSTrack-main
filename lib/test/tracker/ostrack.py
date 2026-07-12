import math

from lib.models.ostrack import build_ostrack
from lib.models.ostrack.candidate_association import CandidateAssociationNet
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.tracker.samurai_motion import BoxKalmanFilter, box_iou_xywh


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)
        network.load_state_dict(checkpoint['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.samurai_cfg = self.cfg.TEST.SAMURAI
        self.samurai_mode = str(self.samurai_cfg.MODE).lower()
        if self.samurai_mode not in {'full', 'topk_no_kf', 'kf_no_gate', 'adaptive_kf'}:
            raise ValueError('TEST.SAMURAI.MODE must be full, topk_no_kf, kf_no_gate, or adaptive_kf.')
        self.kalman_filter = None
        self.assoc_cfg = self.cfg.TEST.CANDIDATE_ASSOC
        self.capture_candidate_features = bool(self.assoc_cfg.CAPTURE or self.assoc_cfg.ENABLE)
        self.association_head = None
        self.association_memory = None
        self.last_candidate_record = None
        self._shallow_feature = None
        self._shallow_hook = None
        if self.capture_candidate_features:
            shallow_layer = int(self.assoc_cfg.SHALLOW_LAYER)
            if not 0 <= shallow_layer < len(self.network.backbone.blocks):
                raise ValueError('TEST.CANDIDATE_ASSOC.SHALLOW_LAYER is outside the ViT block range.')
            self._shallow_hook = self.network.backbone.blocks[shallow_layer].register_forward_hook(
                self._capture_shallow_feature)
        if self.assoc_cfg.ENABLE:
            checkpoint_path = str(self.assoc_cfg.CHECKPOINT)
            if not checkpoint_path:
                raise ValueError('TEST.CANDIDATE_ASSOC.CHECKPOINT is required when B2 is enabled.')
            if not os.path.isabs(checkpoint_path):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                checkpoint_path = os.path.join(project_root, checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError('B2 candidate association checkpoint not found: {}'.format(checkpoint_path))
            self.association_head = CandidateAssociationNet(
                compressed_dim=int(self.assoc_cfg.COMPRESSED_DIM),
                hidden_dim=int(self.assoc_cfg.HIDDEN_DIM),
                embedding_dim=int(self.assoc_cfg.EMBEDDING_DIM),
                temperature=float(self.assoc_cfg.TEMPERATURE)).cuda()
            association_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.association_head.load_state_dict(association_checkpoint['model_state'], strict=True)
            self.association_head.eval()

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        if self.samurai_cfg.ENABLE and self.samurai_mode != 'topk_no_kf':
            self.kalman_filter = BoxKalmanFilter(
                self.state,
                process_noise=self.samurai_cfg.PROCESS_NOISE,
                measurement_noise=self.samurai_cfg.MEASUREMENT_NOISE)
        self.frame_id = 0
        self.association_memory = None
        self.last_candidate_record = None
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        if self.samurai_cfg.ENABLE:
            self.state = self._select_motion_aware_candidate(
                response, out_dict['size_map'], out_dict['offset_map'], resize_factor, H, W, out_dict)
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map']).view(-1, 4)
        else:
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            # Original OSTrack decoding, retained exactly for the baseline.
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        return {"target_bbox": self.state}

    def _capture_shallow_feature(self, _module, _inputs, output):
        """Keep the pre-CE shallow token map from the current backbone pass."""
        self._shallow_feature = output[0].detach()

    def _extract_topk_candidates(self, response, size_map, offset_map, resize_factor, out_dict):
        """Decode B1's NMS-separated candidates and, when requested, capture features."""
        flat_response = response.flatten()
        nms_kernel = int(self.samurai_cfg.NMS_KERNEL)
        if nms_kernel < 1 or nms_kernel % 2 == 0:
            raise ValueError('TEST.SAMURAI.NMS_KERNEL must be a positive odd integer.')
        pooled = torch.nn.functional.max_pool2d(response, nms_kernel, stride=1, padding=nms_kernel // 2)
        peak_scores = response.masked_fill(response.ne(pooled), -float('inf')).flatten()
        num_candidates = min(int(self.samurai_cfg.TOPK), peak_scores.numel())
        if num_candidates < 1:
            raise ValueError('TEST.SAMURAI.TOPK must be at least one.')
        candidate_scores, candidate_idx = torch.topk(peak_scores, k=num_candidates)
        # In the degenerate case where NMS yields fewer peaks, retain valid raw locations.
        if torch.isinf(candidate_scores).any():
            candidate_scores, candidate_idx = torch.topk(flat_response, k=num_candidates)

        feat_w = response.shape[-1]
        idx_y = candidate_idx // feat_w
        idx_x = candidate_idx % feat_w
        flattened_idx = candidate_idx.view(1, 1, -1).expand(1, 2, -1)
        sizes = size_map.flatten(2).gather(2, flattened_idx).squeeze(0).transpose(0, 1)
        offsets = offset_map.flatten(2).gather(2, flattened_idx).squeeze(0).transpose(0, 1)
        candidate_boxes = torch.stack([
            (idx_x.float() + offsets[:, 0]) / self.feat_sz,
            (idx_y.float() + offsets[:, 1]) / self.feat_sz,
            sizes[:, 0], sizes[:, 1]], dim=1)
        candidate_geometry = torch.cat((candidate_scores.unsqueeze(1), candidate_boxes), dim=1)
        candidate_boxes_image = candidate_boxes * self.params.search_size / resize_factor
        candidate_boxes_image = self.map_box_back_batch(candidate_boxes_image, resize_factor)

        result = {
            'scores': candidate_scores,
            'indices': candidate_idx,
            'boxes_crop': candidate_boxes,
            'boxes_image': candidate_boxes_image,
            'geometry': candidate_geometry,
        }
        if self.capture_candidate_features:
            backbone_feature = out_dict['backbone_feat']
            if isinstance(backbone_feature, list):
                backbone_feature = backbone_feature[-1]
            search_length = self.feat_sz * self.feat_sz
            deep_search = backbone_feature[:, -search_length:]
            gather_index = candidate_idx.view(1, -1, 1).expand(1, -1, deep_search.shape[-1])
            deep = deep_search.gather(1, gather_index).squeeze(0)
            if self._shallow_feature is None:
                raise RuntimeError('B2 shallow feature hook did not capture a backbone output.')
            shallow_search = self._shallow_feature[:, -search_length:]
            shallow = shallow_search.gather(1, gather_index).squeeze(0)
            result['deep'] = deep
            result['shallow'] = shallow
        return result

    def _select_motion_aware_candidate(self, response, size_map, offset_map, resize_factor, image_h, image_w, out_dict):
        """Select B1 candidates, optionally with B2 identity association or legacy KF modes."""
        candidates = self._extract_topk_candidates(response, size_map, offset_map, resize_factor, out_dict)
        candidate_scores = candidates['scores']
        candidate_boxes = candidates['boxes_image']
        num_candidates = candidate_scores.numel()

        if self.capture_candidate_features:
            # Kept on GPU for the collector; it is overwritten every frame.
            self.last_candidate_record = {
                'deep': candidates['deep'].detach(),
                'shallow': candidates['shallow'].detach(),
                'geometry': candidates['geometry'].detach(),
                'boxes_image': candidate_boxes.detach(),
            }

        if self.association_head is not None:
            with torch.no_grad():
                embeddings = self.association_head.encode_raw(
                    candidates['deep'], candidates['shallow'], candidates['geometry'])
                appearance_index = int(torch.argmax(candidate_scores).item())
                if self.association_memory is None:
                    best_index = appearance_index
                    self.association_memory = embeddings[best_index].detach()
                else:
                    logits = self.association_head.logits(
                        self.association_memory, embeddings, candidates['geometry']).squeeze(0)
                    selected_index = int(torch.argmax(logits).item())
                    if selected_index < num_candidates:
                        best_index = selected_index
                        self.association_memory = embeddings[best_index].detach()
                    else:
                        # Preserve trusted identity memory on an explicit no-match.
                        best_index = appearance_index
            return clip_box(candidate_boxes[best_index].tolist(), image_h, image_w, margin=10)

        if self.samurai_mode == 'topk_no_kf':
            # Isolate the effect of replacing the original mean decoder with
            # NMS-separated Top-K candidate extraction and appearance-only choice.
            best_index = int(torch.argmax(candidate_scores).item())
            return clip_box(candidate_boxes[best_index].tolist(), image_h, image_w, margin=10)

        predicted_box = self.kalman_filter.predict()
        alpha = float(self.samurai_cfg.MOTION_WEIGHT)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('TEST.SAMURAI.MOTION_WEIGHT must be in [0, 1].')
        if self.samurai_mode == 'adaptive_kf':
            # Normalize the retained response peaks into a discrete local
            # posterior. Its entropy is zero for a decisive visual response
            # and one for equally plausible candidates, so motion becomes a
            # recovery cue rather than a fixed bias on every frame.
            weights = candidate_scores.clamp_min(0)
            weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
            entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).eps).log()).sum()
            entropy = entropy / math.log(float(num_candidates)) if num_candidates > 1 else 0.0
            alpha *= float(entropy)
        best_box, best_score, best_iou, best_appearance = None, -float('inf'), 0.0, 0.0
        for box, appearance_score in zip(candidate_boxes.tolist(), candidate_scores.tolist()):
            box = clip_box(box, image_h, image_w, margin=10)
            motion_iou = box_iou_xywh(predicted_box, box)
            fused_score = (1.0 - alpha) * float(appearance_score) + alpha * motion_iou
            if fused_score > best_score:
                best_box, best_score = box, fused_score
                best_iou, best_appearance = motion_iou, float(appearance_score)

        # The adaptive mode replaces empirical response/IoU thresholds with a
        # normalized innovation squared (NIS) test for the 4-D box measurement.
        # The no-gate ablation updates on every selected measurement; full
        # retains the original reliability protection.
        if (self.samurai_mode == 'adaptive_kf'
                and self.kalman_filter.innovation_mahalanobis(best_box)
                <= float(self.samurai_cfg.MAHALANOBIS_GATE)):
            self.kalman_filter.update(best_box)
        elif (self.samurai_mode == 'kf_no_gate'
                or (best_appearance >= float(self.samurai_cfg.MIN_SCORE)
                    and best_iou >= float(self.samurai_cfg.MIN_IOU))):
            self.kalman_filter.update(best_box)
        return best_box

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack

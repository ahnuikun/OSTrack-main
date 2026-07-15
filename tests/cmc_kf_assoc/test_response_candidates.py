import torch

from lib.test.tracker.cmc_kf_assoc.response_candidates import extract_candidates


def test_top1_matches_official_center_head_decode():
    score = torch.tensor([[[[0.1, 0.8], [0.2, 0.3]]]], dtype=torch.float32)
    size = torch.tensor(
        [[[[0.2, 0.4], [0.5, 0.6]], [[0.3, 0.7], [0.8, 0.9]]]],
        dtype=torch.float32)
    offset = torch.tensor(
        [[[[0.0, 0.25], [0.0, 0.0]], [[0.0, -0.25], [0.0, 0.0]]]],
        dtype=torch.float32)
    candidates = extract_candidates(score, size, offset, top_k=2, suppression_radius=0)
    assert candidates[0].flat_index == 1
    expected = [0.625, -0.125, 0.4, 0.7]
    torch.testing.assert_close(
        torch.tensor(candidates[0].normalized_box), torch.tensor(expected))


def test_spatial_suppression_avoids_adjacent_duplicate_peaks():
    score = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    score[0, 0, 1, 1] = 1.0
    score[0, 0, 1, 2] = 0.9
    score[0, 0, 3, 3] = 0.8
    size = torch.ones((1, 2, 4, 4), dtype=torch.float32) * 0.2
    offset = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    candidates = extract_candidates(score, size, offset, top_k=2, suppression_radius=1)
    assert [item.flat_index for item in candidates] == [5, 15]

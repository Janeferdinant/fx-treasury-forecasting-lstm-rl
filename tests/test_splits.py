"""Walk-forward (expanding window) split generator."""

import pytest

from stocklab.splits import walk_forward_folds


def test_folds_are_chronological_and_disjoint():
    folds = list(walk_forward_folds(n_samples=1000, min_train=400, test_size=100, val_size=80))
    assert len(folds) == 6
    for f in folds:
        assert f.train.stop == f.val.start
        assert f.val.stop == f.test.start
        assert f.train.start == 0
        assert len(f.val) == 80
    # test blocks tile the tail with no gaps and no overlap
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test.stop == b.test.start


def test_train_window_expands():
    folds = list(walk_forward_folds(1000, 400, 100, 80))
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[1] - sizes[0] == 100
    assert folds[0].test.start == 400


def test_last_test_block_never_exceeds_data():
    folds = list(walk_forward_folds(1050, 400, 100, 80))
    assert folds[-1].test.stop == 1050
    assert len(folds[-1].test) == 50  # partial final block is kept
    assert all(f.test.stop <= 1050 for f in folds)


def test_every_sample_after_min_train_is_tested_exactly_once():
    n = 977
    folds = list(walk_forward_folds(n, 300, 120, 60))
    tested = [i for f in folds for i in f.test]
    assert tested == list(range(300, n))


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        list(walk_forward_folds(100, 100, 20, 10))  # no room for a test block
    with pytest.raises(ValueError):
        list(walk_forward_folds(1000, 50, 100, 80))  # validation larger than train
    with pytest.raises(ValueError):
        list(walk_forward_folds(1000, 400, 0, 80))

from data.segment_dataset import enumerate_nonoverlap_windows


def _assert_exact_coverage(num_frames, window_frames):
    windows = enumerate_nonoverlap_windows(num_frames, window_frames)
    intervals = [(w.start, w.start + w.valid_frames) for w in windows]
    assert intervals[0][0] == 0
    assert intervals[-1][1] == num_frames
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert sum(w.valid_frames for w in windows) == num_frames
    assert all(w.valid_frames + w.pad_right == window_frames for w in windows)
    assert len({w.start for w in windows}) == len(windows)


def test_30_seconds_has_six_5s_windows():
    frames = 30 * 44100
    windows = enumerate_nonoverlap_windows(frames, 5 * 44100)
    assert len(windows) == 6
    assert all(w.pad_right == 0 for w in windows)
    _assert_exact_coverage(frames, 5 * 44100)


def test_30_seconds_has_three_10s_windows():
    frames = 30 * 44100
    windows = enumerate_nonoverlap_windows(frames, 10 * 44100)
    assert len(windows) == 3
    assert all(w.pad_right == 0 for w in windows)
    _assert_exact_coverage(frames, 10 * 44100)


def test_short_tail_is_right_padded_without_overlap():
    _assert_exact_coverage(30 * 44100 + 17, 10 * 44100)
    windows = enumerate_nonoverlap_windows(30 * 44100 + 17, 10 * 44100)
    assert windows[-1].valid_frames == 17
    assert windows[-1].pad_right == 10 * 44100 - 17


def test_rejects_unsupported_tail_policy():
    try:
        enumerate_nonoverlap_windows(100, 10, "overlap_last")
    except ValueError as exc:
        assert "tail_policy" in str(exc)
    else:
        raise AssertionError("unsupported tail policy was accepted")

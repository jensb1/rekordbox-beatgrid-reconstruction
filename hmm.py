"""HMM decoder used by rekordbox's high-precision beat analysis.

This is a numpy port of madmom's `DBNDownBeatTrackingProcessor` with the
parameters recovered from rekordbox's `madmom::downbeat_tracking` call:

    beats_per_bar=4, min_bpm=70, max_bpm=180, num_tempi=60,
    transition_lambda=100, observation_lambda=16, threshold=0.05,
    correct=True, fps=100

The state-space / transition / observation-model code follows madmom
(BSD-3-Clause, CPJKU/madmom) so the output can be compared 1:1 with it.
"""

from __future__ import annotations

import numpy as np

FPS = 100.0
BEATS_PER_BAR = 4
MIN_BPM = 70.0
MAX_BPM = 180.0
NUM_TEMPI = 60
TRANSITION_LAMBDA = 100
OBSERVATION_LAMBDA = 16
THRESHOLD = 0.05
CORRECT = True


def beat_state_space(min_interval, max_interval, num_intervals=None):
    intervals = np.arange(np.round(min_interval), np.round(max_interval) + 1)
    if num_intervals is not None and num_intervals < len(intervals):
        n = num_intervals
        while len(intervals) < num_intervals:
            intervals = np.logspace(
                np.log2(min_interval), np.log2(max_interval), n, base=2
            )
            intervals = np.unique(np.round(intervals))
            n += 1
    intervals = np.ascontiguousarray(intervals, dtype=int)
    num_states = int(intervals.sum())
    first_states = np.cumsum(np.r_[0, intervals[:-1]]).astype(int)
    last_states = np.cumsum(intervals) - 1
    state_positions = np.empty(num_states)
    state_intervals = np.empty(num_states, dtype=int)
    idx = 0
    for interval in intervals:
        state_positions[idx : idx + interval] = np.linspace(
            0, 1, interval, endpoint=False
        )
        state_intervals[idx : idx + interval] = interval
        idx += interval
    return {
        "num_states": num_states,
        "intervals": intervals,
        "num_intervals": len(intervals),
        "first_states": first_states,
        "last_states": last_states,
        "state_positions": state_positions,
        "state_intervals": state_intervals,
    }


def bar_state_space(num_beats, min_interval, max_interval, num_intervals=None):
    beat = beat_state_space(min_interval, max_interval, num_intervals)
    state_positions = np.hstack(
        [beat["state_positions"] + b for b in range(int(num_beats))]
    )
    state_intervals = np.tile(beat["state_intervals"], int(num_beats))
    first_states = [
        beat["first_states"] + b * beat["num_states"] for b in range(int(num_beats))
    ]
    last_states = [
        beat["last_states"] + b * beat["num_states"] for b in range(int(num_beats))
    ]
    return {
        "num_beats": int(num_beats),
        "num_states": beat["num_states"] * int(num_beats),
        "state_positions": state_positions,
        "state_intervals": state_intervals,
        "first_states": first_states,
        "last_states": last_states,
    }


def exponential_transition(from_intervals, to_intervals, transition_lambda,
                           threshold=np.spacing(1), norm=True):
    if transition_lambda is None:
        return np.diag(np.diag(np.ones((len(from_intervals), len(to_intervals)))))
    ratio = to_intervals.astype(float) / from_intervals.astype(float)[:, np.newaxis]
    prob = np.exp(-transition_lambda * np.abs(ratio - 1.0))
    prob[prob <= threshold] = 0
    if norm:
        prob /= prob.sum(axis=1)[:, np.newaxis]
    return prob


def make_sparse(dest, prev, probabilities):
    """CSR-like transitions, indexed by destination state (madmom layout)."""
    dest = np.asarray(dest)
    prev = np.asarray(prev, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    num_states = int(max(prev)) + 1
    order = np.argsort(dest, kind="stable")
    counts = np.bincount(dest, minlength=num_states)
    pointers = np.r_[0, np.cumsum(counts)].astype(np.uint32)
    states = prev[order].astype(np.uint32)
    probabilities = probabilities[order]
    # The Viterbi loop requires at least one incoming transition per state.
    if np.any(counts == 0):
        raise ValueError("state without incoming transition")
    return states, pointers, probabilities


def bar_transition_model(state_space, transition_lambda):
    states = np.arange(state_space["num_states"], dtype=np.uint32)
    states = np.setdiff1d(states, np.concatenate(state_space["first_states"]))
    prev_states = states - 1
    probabilities = np.ones_like(states, dtype=float)
    for beat in range(state_space["num_beats"]):
        to_states = state_space["first_states"][beat]
        from_states = state_space["last_states"][beat - 1]
        from_int = state_space["state_intervals"][from_states]
        to_int = state_space["state_intervals"][to_states]
        prob = exponential_transition(from_int, to_int, transition_lambda)
        from_prob, to_prob = np.nonzero(prob)
        states = np.hstack((states, to_states[to_prob]))
        prev_states = np.hstack((prev_states, from_states[from_prob]))
        probabilities = np.hstack((probabilities, prob[prob != 0]))
    return make_sparse(states, prev_states, probabilities)


def observation_model(state_space, observation_lambda):
    pointers = np.zeros(state_space["num_states"], dtype=np.uint32)
    border = 1.0 / observation_lambda
    pointers[state_space["state_positions"] % 1 < border] = 1
    pointers[state_space["state_positions"] < border] = 2
    return pointers


def threshold_activations(activations, threshold):
    first = last = 0
    idx = np.nonzero(activations >= threshold)[0]
    if idx.any():
        first = max(first, int(np.min(idx)))
        last = min(len(activations), int(np.max(idx)) + 1)
    return activations[first:last], first


def log_densities(observations, observation_lambda):
    out = np.empty((len(observations), 3), dtype=float)
    # float32 softmax outputs can sum to slightly more than 1; clamp so the
    # non-beat log density never becomes NaN (madmom never sees this because
    # its activations sum to <1).
    tiny = np.finfo(float).tiny
    non_beat = np.maximum(1.0 - np.sum(observations, axis=1), 0.0)
    out[:, 0] = np.log(np.maximum(non_beat, tiny) / (observation_lambda - 1))
    out[:, 1] = np.log(np.maximum(observations[:, 0], tiny))
    out[:, 2] = np.log(np.maximum(observations[:, 1], tiny))
    return out


def viterbi(states, pointers, probabilities, om_pointers, densities,
            initial_distribution=None):
    """Vectorised Viterbi, same recursion as madmom's Cython implementation."""
    num_states = len(pointers) - 1
    log_transitions = np.log(probabilities)
    starts = pointers[:-1]
    counts = np.diff(pointers)
    if initial_distribution is None:
        previous = np.full(num_states, -np.log(num_states))
    else:
        previous = np.log(initial_distribution)

    num_frames = len(densities)
    back_pointers = np.empty((num_frames, num_states), dtype=np.uint32)
    segment_ids = np.repeat(np.arange(num_states), counts)
    for frame in range(num_frames):
        candidates = previous[states] + log_transitions
        # segment-wise maximum over all transitions into each destination state
        max_per_state = np.maximum.reduceat(candidates, starts)
        current = max_per_state + densities[frame, om_pointers]
        # first transition attaining the maximum, per destination state
        equal = candidates == np.repeat(max_per_state, counts)
        indices = np.flatnonzero(equal)
        segments = segment_ids[indices]
        _, first = np.unique(segments, return_index=True)
        back_pointers[frame, segments[first]] = states[indices[first]]
        previous = current

    state = int(np.argmax(previous))
    log_probability = previous[state]
    if np.isinf(log_probability):
        return np.empty(0, dtype=np.uint32), float(log_probability)
    path = np.empty(num_frames, dtype=np.uint32)
    for frame in range(num_frames - 1, -1, -1):
        path[frame] = state
        state = back_pointers[frame, state]
    return path, float(log_probability)


def downbeat_tracking(activations, fps=FPS,
                      min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                      transition_lambda=TRANSITION_LAMBDA,
                      observation_lambda=OBSERVATION_LAMBDA,
                      threshold=THRESHOLD, correct=CORRECT):
    """Return (seconds, beat_number) like madmom's DBNDownBeatTrackingProcessor."""
    activations = np.asarray(activations, dtype=np.float32)
    first = 0
    if threshold:
        activations, first = threshold_activations(activations, threshold)
    if not activations.any():
        return np.empty((0, 2))

    state_space = bar_state_space(
        BEATS_PER_BAR, 60.0 * fps / max_bpm, 60.0 * fps / min_bpm, num_tempi
    )
    states, pointers, probabilities = bar_transition_model(
        state_space, transition_lambda
    )
    om_pointers = observation_model(state_space, observation_lambda)
    densities = log_densities(activations, observation_lambda)
    path, _ = viterbi(states, pointers, probabilities, om_pointers, densities)
    if len(path) == 0:
        return np.empty((0, 2))

    positions = state_space["state_positions"][path]
    beat_numbers = positions.astype(int) + 1
    if correct:
        beat_range = om_pointers[path] >= 1
        if not beat_range.any():
            return np.empty((0, 2))
        idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
        if beat_range[0]:
            idx = np.r_[0, idx]
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        beats = np.empty(0, dtype=int)
        for left, right in idx.reshape((-1, 2)):
            peak = int(np.argmax(activations[left:right])) // 2 + left
            beats = np.hstack((beats, peak))
    else:
        beats = np.nonzero(np.diff(beat_numbers))[0] + 1
    return np.vstack(((beats + first) / float(fps), beat_numbers[beats])).T

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.dllm.tp_local_vocab_state import (
    argmax_max_prob_from_logits_output,
    local_vocab_state_from_logits,
    low_confidence_transfer_mask,
    merge_vocab_states,
    VocabState,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

ENDPOINT_BASELINE_URL_ENV = "SGLANG_DLLM_BASELINE_URL"
ENDPOINT_TP_LOCAL_URL_ENV = "SGLANG_DLLM_TP_LOCAL_VOCAB_URL"


def _dense_argmax_and_max_prob(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    max_values, argmax_ids = torch.max(logits, dim=-1)
    max_probs = torch.exp(max_values - torch.logsumexp(logits, dim=-1))
    return argmax_ids.long(), max_probs


def test_low_confidence_tp_state_matches_dense_logits():
    torch.manual_seed(0)
    logits = torch.randn(9, 23, dtype=torch.float32)
    input_ids = torch.tensor([99, 99, 3, 99, 4, 99, 99, 7, 99], dtype=torch.long)
    mask_id = 99
    threshold = 0.19

    dense_argmax_ids, dense_max_probs = _dense_argmax_and_max_prob(logits)
    dense_transfer = low_confidence_transfer_mask(
        input_ids=input_ids,
        argmax_ids=dense_argmax_ids,
        max_probs=dense_max_probs,
        mask_id=mask_id,
        threshold=threshold,
    )

    shard_sizes = [5, 11, 7]
    states = []
    vocab_start = 0
    for shard_size in shard_sizes:
        shard_logits = logits[:, vocab_start : vocab_start + shard_size]
        states.append(
            local_vocab_state_from_logits(
                local_logits=shard_logits,
                vocab_start=vocab_start,
            )
        )
        vocab_start += shard_size

    merged = merge_vocab_states(states)
    merged_transfer = low_confidence_transfer_mask(
        input_ids=input_ids,
        argmax_ids=merged.argmax_ids,
        max_probs=merged.max_probs,
        mask_id=mask_id,
        threshold=threshold,
    )

    torch.testing.assert_close(merged.max_probs, dense_max_probs)
    torch.testing.assert_close(merged.logsumexp, torch.logsumexp(logits.float(), dim=-1))
    assert torch.equal(merged.argmax_ids, dense_argmax_ids)
    assert torch.equal(merged_transfer, dense_transfer)


def test_local_vocab_state_ignores_padded_vocab_entries():
    dense_logits = torch.tensor(
        [
            [0.0, 0.3, -0.2, 0.1, 1.0, 0.9],
            [1.2, -0.4, 0.5, 0.2, -1.0, 0.7],
        ],
        dtype=torch.float32,
    )
    dense_argmax_ids, dense_max_probs = _dense_argmax_and_max_prob(dense_logits)

    first_state = local_vocab_state_from_logits(
        local_logits=dense_logits[:, :4],
        vocab_start=0,
    )
    second_state = local_vocab_state_from_logits(
        local_logits=torch.cat(
            [
                dense_logits[:, 4:],
                torch.full((dense_logits.shape[0], 3), 1000.0),
            ],
            dim=-1,
        ),
        vocab_start=4,
        valid_vocab_size=2,
    )

    merged = merge_vocab_states([first_state, second_state])

    torch.testing.assert_close(merged.max_probs, dense_max_probs)
    torch.testing.assert_close(
        merged.logsumexp,
        torch.logsumexp(dense_logits.float(), dim=-1),
    )
    assert torch.equal(merged.argmax_ids, dense_argmax_ids)


def test_joint_threshold_penalty_matches_dense_logits():
    logits = torch.tensor(
        [
            [0.2, 1.0, 0.1, -0.1, 0.0, 0.3, -0.4, 0.6],
            [0.5, 0.2, 1.4, 0.8, 0.1, 1.7, 0.4, -0.2],
            [1.3, 0.1, -0.5, 1.6, 0.2, 0.0, 1.5, 0.4],
            [0.0, 0.7, 0.2, -0.3, 1.9, 0.8, 0.1, 1.8],
        ],
        dtype=torch.float32,
    )
    penalty_token_ids = torch.tensor([-1, 5, 3, 6], dtype=torch.long)
    penalty_lambda = 0.75

    dense_penalized = logits.clone()
    row_ids = torch.arange(logits.shape[0])
    penalized_rows = penalty_token_ids >= 0
    dense_penalized[row_ids[penalized_rows], penalty_token_ids[penalized_rows]] -= (
        penalty_lambda
    )
    dense_argmax_ids, dense_max_probs = _dense_argmax_and_max_prob(dense_penalized)

    first_state = local_vocab_state_from_logits(
        local_logits=logits[:, :3],
        vocab_start=0,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )
    second_state = local_vocab_state_from_logits(
        local_logits=logits[:, 3:6],
        vocab_start=3,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )
    third_state = local_vocab_state_from_logits(
        local_logits=logits[:, 6:],
        vocab_start=6,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )

    merged = merge_vocab_states([first_state, second_state, third_state])

    torch.testing.assert_close(merged.max_probs, dense_max_probs)
    torch.testing.assert_close(
        merged.logsumexp,
        torch.logsumexp(dense_penalized.float(), dim=-1),
    )
    assert torch.equal(merged.argmax_ids, dense_argmax_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_local_vocab_state_matches_reference_cuda():
    from sglang.srt.dllm.tp_local_vocab_kernel import (
        local_vocab_state_from_logits_triton,
    )

    torch.manual_seed(1)
    logits = torch.randn(6, 41, device="cuda", dtype=torch.float32)
    logits[:, 37:] = 1000.0

    actual = local_vocab_state_from_logits_triton(
        local_logits=logits,
        vocab_start=17,
        valid_vocab_size=37,
    )
    expected = local_vocab_state_from_logits(
        local_logits=logits,
        vocab_start=17,
        valid_vocab_size=37,
    )

    torch.testing.assert_close(actual.max_values, expected.max_values)
    torch.testing.assert_close(actual.max_probs, expected.max_probs)
    torch.testing.assert_close(actual.logsumexp, expected.logsumexp)
    assert torch.equal(actual.argmax_ids, expected.argmax_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_local_vocab_state_applies_penalty_cuda():
    from sglang.srt.dllm.tp_local_vocab_kernel import (
        local_vocab_state_from_logits_triton,
    )

    torch.manual_seed(2)
    logits = torch.randn(5, 29, device="cuda", dtype=torch.float32)
    penalty_token_ids = torch.tensor([11, -1, 17, 30, 27], device="cuda")
    penalty_lambda = 0.5

    actual = local_vocab_state_from_logits_triton(
        local_logits=logits,
        vocab_start=9,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )
    expected = local_vocab_state_from_logits(
        local_logits=logits,
        vocab_start=9,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )

    torch.testing.assert_close(actual.max_values, expected.max_values)
    torch.testing.assert_close(actual.max_probs, expected.max_probs)
    torch.testing.assert_close(actual.logsumexp, expected.logsumexp)
    assert torch.equal(actual.argmax_ids, expected.argmax_ids)


def test_argmax_max_prob_uses_compact_state_without_full_logits():
    state = VocabState(
        max_values=torch.tensor([1.0, 2.0, 3.0]),
        argmax_ids=torch.tensor([7, 8, 9]),
        logsumexp=torch.tensor([1.5, 2.5, 3.5]),
        max_probs=torch.tensor([0.5, 0.6, 0.7]),
    )
    logits_output = LogitsProcessorOutput(
        next_token_logits=None,
        full_logits=None,
        dllm_vocab_state=state,
    )

    argmax_ids, max_probs = argmax_max_prob_from_logits_output(
        logits_output,
        start=1,
        end=3,
    )

    assert torch.equal(argmax_ids, torch.tensor([8, 9]))
    torch.testing.assert_close(max_probs, torch.tensor([0.6, 0.7]))


def test_argmax_max_prob_falls_back_to_full_logits_without_compact_state():
    logits = torch.tensor(
        [
            [0.0, 4.0, 1.0],
            [2.5, -1.0, 2.5],
            [-3.0, -2.0, -1.0],
            [0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    logits_output = LogitsProcessorOutput(
        next_token_logits=None,
        full_logits=logits,
        dllm_vocab_state=None,
    )

    argmax_ids, max_probs = argmax_max_prob_from_logits_output(
        logits_output,
        start=1,
        end=3,
    )

    expected_argmax_ids, expected_max_probs = _dense_argmax_and_max_prob(logits[1:3])
    assert torch.equal(argmax_ids, expected_argmax_ids)
    torch.testing.assert_close(max_probs, expected_max_probs)


def test_merge_vocab_states_tie_breaks_equal_max_values_by_token_id():
    states = [
        VocabState(
            max_values=torch.tensor([5.0, 7.0, 1.0]),
            argmax_ids=torch.tensor([10, 12, 30]),
            logsumexp=torch.log(torch.tensor([2.0, 3.0, 5.0])),
            max_probs=torch.empty(3),
        ),
        VocabState(
            max_values=torch.tensor([5.0, 6.0, 1.0]),
            argmax_ids=torch.tensor([8, 9, 25]),
            logsumexp=torch.log(torch.tensor([4.0, 7.0, 11.0])),
            max_probs=torch.empty(3),
        ),
        VocabState(
            max_values=torch.tensor([4.0, 7.0, 2.0]),
            argmax_ids=torch.tensor([6, 11, 40]),
            logsumexp=torch.log(torch.tensor([13.0, 17.0, 19.0])),
            max_probs=torch.empty(3),
        ),
    ]

    merged = merge_vocab_states(states)

    assert torch.equal(merged.argmax_ids, torch.tensor([8, 11, 40]))
    torch.testing.assert_close(merged.max_values, torch.tensor([5.0, 7.0, 2.0]))
    torch.testing.assert_close(
        merged.logsumexp,
        torch.log(torch.tensor([19.0, 27.0, 35.0])),
    )
    torch.testing.assert_close(
        merged.max_probs,
        torch.exp(merged.max_values - merged.logsumexp),
    )


def test_logits_processor_tp_local_vocab_gate(monkeypatch):
    import sglang.srt.layers.logits_processor as logits_processor
    from sglang.srt.environ import envs

    processor = SimpleNamespace(
        use_attn_tp_group=False,
        do_tensor_parallel_all_gather_dp_attn=False,
    )
    lm_head = SimpleNamespace(weight=torch.empty(1))
    should_use = logits_processor.LogitsProcessor._should_use_dllm_tp_local_vocab

    monkeypatch.setattr(
        logits_processor,
        "get_global_server_args",
        lambda: SimpleNamespace(dllm_algorithm="LowConfidence"),
    )

    with envs.SGLANG_DLLM_TP_LOCAL_VOCAB.override(False):
        assert should_use(processor, lm_head) is False

    with envs.SGLANG_DLLM_TP_LOCAL_VOCAB.override(True):
        assert should_use(processor, lm_head) is True

        monkeypatch.setattr(
            logits_processor,
            "get_global_server_args",
            lambda: SimpleNamespace(dllm_algorithm="JointThreshold"),
        )
        assert should_use(processor, lm_head) is False

        monkeypatch.setattr(
            logits_processor,
            "get_global_server_args",
            lambda: SimpleNamespace(dllm_algorithm="LowConfidence"),
        )
        assert (
            should_use(
                SimpleNamespace(
                    use_attn_tp_group=True,
                    do_tensor_parallel_all_gather_dp_attn=False,
                ),
                lm_head,
            )
            is False
        )
        assert (
            should_use(
                SimpleNamespace(
                    use_attn_tp_group=False,
                    do_tensor_parallel_all_gather_dp_attn=True,
                ),
                lm_head,
            )
            is False
        )
        assert should_use(processor, SimpleNamespace()) is False

        padded_lm_head = SimpleNamespace(
            weight=torch.empty(1),
            shard_indices=SimpleNamespace(num_added_elements=2),
        )
        unpadded_lm_head = SimpleNamespace(
            weight=torch.empty(1),
            shard_indices=SimpleNamespace(num_added_elements=0),
        )
        assert should_use(processor, padded_lm_head) is False
        assert should_use(processor, unpadded_lm_head) is True


def _endpoint_url(env_key: str) -> str | None:
    url = os.environ.get(env_key)
    if not url:
        return None
    return url.rstrip("/")


def _post_generate(endpoint: str, prompt: str) -> str:
    requests = pytest.importorskip("requests")
    response = requests.post(
        f"{endpoint}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        if "text" in payload:
            return payload["text"]
        if "output" in payload:
            return payload["output"]
        return payload["outputs"][0]["text"]
    return payload[0]["text"]


def test_baseline_and_tp_local_vocab_endpoints_match_when_configured():
    baseline_url = _endpoint_url(ENDPOINT_BASELINE_URL_ENV)
    tp_local_url = _endpoint_url(ENDPOINT_TP_LOCAL_URL_ENV)
    if not baseline_url or not tp_local_url:
        pytest.skip(
            f"set {ENDPOINT_BASELINE_URL_ENV} and {ENDPOINT_TP_LOCAL_URL_ENV} "
            "to compare already-running endpoints"
        )

    prompts = [
        "Name one primary color.",
        "Complete the sequence: 1, 1, 2, 3,",
    ]

    for prompt in prompts:
        assert _post_generate(tp_local_url, prompt) == _post_generate(
            baseline_url, prompt
        )


def test_cuda_graph_replay_slice_preserves_compact_vocab_state():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        _slice_dllm_vocab_state,
    )

    state = VocabState(
        max_values=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        argmax_ids=torch.tensor([10, 11, 12, 13]),
        logsumexp=torch.tensor([1.5, 2.5, 3.5, 4.5]),
        max_probs=torch.tensor([0.6, 0.7, 0.8, 0.9]),
    )

    sliced = _slice_dllm_vocab_state(state, 2)

    assert sliced is not None
    torch.testing.assert_close(sliced.max_values, torch.tensor([1.0, 2.0]))
    assert torch.equal(sliced.argmax_ids, torch.tensor([10, 11]))
    torch.testing.assert_close(sliced.logsumexp, torch.tensor([1.5, 2.5]))
    torch.testing.assert_close(sliced.max_probs, torch.tensor([0.6, 0.7]))

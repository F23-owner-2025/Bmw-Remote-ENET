from tests.conftest import make_agentic_conversation, make_chat_conversation
from trainkit.mask_check import verify_masking


def _corpus(n=20):
    convs = []
    for i in range(n):
        convs.append(make_agentic_conversation(f"inspect module {i}"))
        convs.append(make_chat_conversation(f"Explain concept number {i}."))
    return convs


def test_healthy_corpus_passes(tokenizer):
    report = verify_masking(_corpus(), tokenizer)
    assert report.passed, report.summary()
    assert report.examples_checked == 40
    assert 0.02 < report.trainable_fraction_mean < 0.85
    assert report.tokenization_drift_rate == 0.0


def test_all_masked_fails(tokenizer):
    # Simulate "mask ate everything" by bounding the healthy range above
    # the corpus's real fraction — the same failure signature as a mask
    # that strips assistant tokens.
    report = verify_masking(_corpus(), tokenizer, min_trainable_fraction=0.99,
                            max_trainable_fraction=0.999)
    assert not report.passed
    assert any("mean trainable fraction" in f for f in report.failures)


def test_everything_trained_fails(tokenizer):
    report = verify_masking(_corpus(), tokenizer, min_trainable_fraction=0.0001,
                            max_trainable_fraction=0.001)
    assert not report.passed


def test_drift_detector_fires(drifty_tokenizer):
    report = verify_masking(_corpus(), drifty_tokenizer,
                            max_tokenization_drift=0.02)
    assert report.tokenization_drift_rate > 0.02
    assert any("drift" in f for f in report.failures)


def test_drift_within_tolerance_warns_not_fails(drifty_tokenizer):
    report = verify_masking(_corpus(), drifty_tokenizer,
                            max_tokenization_drift=1.0)
    assert not any("drift" in f for f in report.failures)
    assert any("drift" in w for w in report.warnings)


def test_empty_corpus_fails(tokenizer):
    report = verify_masking([], tokenizer)
    assert not report.passed


def test_sampling_respects_size(tokenizer):
    report = verify_masking(_corpus(100), tokenizer, sample_size=25)
    assert report.examples_checked == 25

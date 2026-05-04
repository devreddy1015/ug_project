from training.guard_platform import adversarial_evasion_score


def test_evasion_score_stays_low_for_plain_text() -> None:
    text = (
        "This is a normal caption with plain language and no obfuscation tactics. "
        "We discuss community updates and event highlights."
    )
    score = adversarial_evasion_score(text)
    assert score < 20.0


def test_evasion_score_rises_for_obfuscated_text() -> None:
    text = "h*a*t*e m-a-s-k-ed c0d3d m3ssage !!!!!! @@@### y0u_kn0w"
    score = adversarial_evasion_score(text)
    assert score > 35.0

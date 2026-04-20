CATEGORIES = [
    "body_shaming",
    "child_abuse_and_exploitation",
    "cyberbullying",
    "harassment",
    "hate_speeches",
    "homophobic_content",
    "invasive_privacy_violation",
    "misinformation_and_fake_news",
    "racial_slurs",
    "self_harm_and_suicide",
    "sexist_content",
    "sexually_explicit_material",
    "substance_abuse",
    "transphobic_content",
    "violence_and_terrorism",
]

SYSTEM_PROMPT = (
    "You are a safety classifier. Return JSON with a score between 0 and 100 for each "
    "category. Output only JSON."
)


def build_prompt(text: str) -> str:
    return (
        "Analyze the content and score it for the following categories: "
        f"{', '.join(CATEGORIES)}.\n\n"
        "Content:\n"
        f"{text}\n"
    )

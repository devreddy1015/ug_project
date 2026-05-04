from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CategorySpec:
    name: str
    layer: str
    description: str
    keywords: Tuple[str, ...]


CATEGORY_SPECS: Tuple[CategorySpec, ...] = (
    CategorySpec(
        name="explicit_content",
        layer="Safety & Harm Layer",
        description="Sexual or explicit adult material.",
        keywords=("explicit", "porn", "nudity", "nsfw", "adult"),
    ),
    CategorySpec(
        name="violence",
        layer="Safety & Harm Layer",
        description="Physical violence, gore, or attack content.",
        keywords=("violence", "attack", "weapon", "blood", "fight"),
    ),
    CategorySpec(
        name="self_harm",
        layer="Safety & Harm Layer",
        description="Self-harm or suicide-related intent or encouragement.",
        keywords=("self harm", "suicide", "cut", "end life", "kill myself"),
    ),
    CategorySpec(
        name="hate_speech",
        layer="Safety & Harm Layer",
        description="Content attacking people based on identity.",
        keywords=("hate", "slur", "inferior", "racist", "bigot"),
    ),
    CategorySpec(
        name="substance_abuse",
        layer="Safety & Harm Layer",
        description="Drug abuse or dangerous substance use promotion.",
        keywords=("drug", "overdose", "substance", "high", "abuse"),
    ),
    CategorySpec(
        name="child_safety_risk",
        layer="Safety & Harm Layer",
        description="Risk to children, exploitation, or unsafe exposure.",
        keywords=("child", "minor", "underage", "exploit", "groom"),
    ),
    CategorySpec(
        name="cyberbullying",
        layer="Safety & Harm Layer",
        description="Targeted online harassment and bullying.",
        keywords=("bully", "harass", "troll", "humiliate", "target"),
    ),
    CategorySpec(
        name="harassment",
        layer="Safety & Harm Layer",
        description="Abusive, threatening, or demeaning treatment.",
        keywords=("harass", "abuse", "threat", "intimidate", "stalk"),
    ),
    CategorySpec(
        name="sexual_predation",
        layer="Safety & Harm Layer",
        description="Predatory sexual behavior or grooming patterns.",
        keywords=("predator", "grooming", "coerce", "trap", "exploit"),
    ),
    CategorySpec(
        name="extremist_content",
        layer="Safety & Harm Layer",
        description="Extremist or terror-justifying material.",
        keywords=("extremist", "terror", "radical", "martyr", "jihad"),
    ),
    CategorySpec(
        name="misinformation",
        layer="Societal Impact Layer",
        description="False or misleading claims presented as fact.",
        keywords=("fake", "hoax", "misinformation", "rumor", "debunked"),
    ),
    CategorySpec(
        name="body_image_harm",
        layer="Societal Impact Layer",
        description="Content that harms body image and self-worth.",
        keywords=("body", "fat", "ugly", "skinny", "appearance"),
    ),
    CategorySpec(
        name="financial_scams",
        layer="Societal Impact Layer",
        description="Scam-like financial promises or fraud behavior.",
        keywords=("scam", "guaranteed return", "ponzi", "fraud", "quick money"),
    ),
    CategorySpec(
        name="political_propaganda",
        layer="Societal Impact Layer",
        description="Manipulative political messaging with low factual integrity.",
        keywords=("propaganda", "regime", "agenda", "brainwash", "party line"),
    ),
    CategorySpec(
        name="mental_health_triggers",
        layer="Societal Impact Layer",
        description="Potential triggers for anxiety, trauma, or depression.",
        keywords=("panic", "trigger", "trauma", "anxiety", "depressed"),
    ),
    CategorySpec(
        name="addiction_bait",
        layer="Societal Impact Layer",
        description="Patterns optimized for compulsive consumption.",
        keywords=("endless", "dopamine", "loop", "hook", "binge"),
    ),
    CategorySpec(
        name="privacy_violation",
        layer="Societal Impact Layer",
        description="Doxxing or exposing private personal information.",
        keywords=("dox", "address", "phone number", "private", "leak"),
    ),
    CategorySpec(
        name="gambling_promotion",
        layer="Societal Impact Layer",
        description="Promotion of high-risk betting behavior.",
        keywords=("bet", "casino", "odds", "jackpot", "gambling"),
    ),
    CategorySpec(
        name="manipulative_marketing",
        layer="Societal Impact Layer",
        description="Manipulative selling pressure and exploitative funnels.",
        keywords=("buy now", "limited time", "fear of missing out", "upsell", "pressure"),
    ),
    CategorySpec(
        name="polarization",
        layer="Societal Impact Layer",
        description="Us-vs-them narratives that intensify social conflict.",
        keywords=("enemy", "us vs them", "divide", "traitor", "purge"),
    ),
    CategorySpec(
        name="educational_value",
        layer="Positive Value Layer",
        description="Teaches useful knowledge or practical skills.",
        keywords=("learn", "tutorial", "explain", "lesson", "how to"),
    ),
    CategorySpec(
        name="emotional_positivity",
        layer="Positive Value Layer",
        description="Encourages optimism, kindness, and emotional well-being.",
        keywords=("hope", "kindness", "support", "uplift", "encourage"),
    ),
    CategorySpec(
        name="cultural_appreciation",
        layer="Positive Value Layer",
        description="Respectful celebration of cultures and heritage.",
        keywords=("culture", "heritage", "tradition", "community", "respect"),
    ),
    CategorySpec(
        name="creativity",
        layer="Positive Value Layer",
        description="Original creative expression and artistic effort.",
        keywords=("creative", "art", "design", "original", "music"),
    ),
    CategorySpec(
        name="community_building",
        layer="Positive Value Layer",
        description="Strengthens social support and collective progress.",
        keywords=("community", "collaborate", "together", "volunteer", "help"),
    ),
    CategorySpec(
        name="empathy_support",
        layer="Positive Value Layer",
        description="Shows empathy and care for others' situations.",
        keywords=("empathy", "listen", "care", "understand", "support"),
    ),
    CategorySpec(
        name="constructive_dialogue",
        layer="Positive Value Layer",
        description="Promotes respectful and constructive conversation.",
        keywords=("dialogue", "respect", "debate", "understanding", "civil"),
    ),
    CategorySpec(
        name="skill_development",
        layer="Positive Value Layer",
        description="Helps people improve real-world abilities.",
        keywords=("practice", "skills", "improve", "master", "training"),
    ),
)


CATEGORY_NAMES: List[str] = [spec.name for spec in CATEGORY_SPECS]


def layer_to_categories() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for spec in CATEGORY_SPECS:
        mapping.setdefault(spec.layer, []).append(spec.name)
    return mapping


def category_descriptions() -> Dict[str, str]:
    return {spec.name: spec.description for spec in CATEGORY_SPECS}


def category_keywords() -> Dict[str, Tuple[str, ...]]:
    return {spec.name: spec.keywords for spec in CATEGORY_SPECS}

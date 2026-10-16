"""The charset safety filter that every tokenizer input passes through."""

import pytest

from hebrew_training.tokenizer_charset import CharsetFilter, niqqud_rate


@pytest.fixture
def charset() -> CharsetFilter:
    return CharsetFilter()


@pytest.mark.parametrize(
    "line",
    [
        "שלום עולם, מה שלומך?",
        "Hello, world! It's 5 o'clock (almost).",
        "בשנת 1948, בשעה 14:30 - 2.5% עלייה.",
        "אֲנִי רוֹצָה לָלֶכֶת הַבַּיְתָה, תּוֹדָה רַבָּה.",
        "הוא עבד ב-Microsoft לפני שהצטרף לצוות.",
        'ציטוט "בתוך" מירכאות ו-[סוגריים] {מסולסלים}…',
    ],
)
def test_supported_text_passes_unchanged(charset, line):
    assert charset.clean(line) == line


@pytest.mark.parametrize(
    "line",
    [
        "Привет, мир",  # Cyrillic
        "مرحبا بالعالم",  # Arabic
        "你好世界",  # CJK
        "mixed עברית and Привет",
        "price \u00a9 2024",  # symbol outside the set
        "\u05f0\u05f1 yiddish ligatures",
    ],
)
def test_other_scripts_drop_the_whole_line(charset, line):
    assert charset.clean(line) is None
    assert charset.stats.dropped_foreign == 1


def test_strip_mode_removes_just_the_foreign_character():
    strip = CharsetFilter(on_foreign="strip")
    assert strip.clean("hello мир world") == "hello world"
    assert strip.clean("只有中文") is None  # nothing left


def test_cantillation_and_niqqud_are_both_kept(charset):
    # trope U+0591 (etnahta) on an otherwise vocalized word
    line = "בְּרֵאשִׁ\u0591ית בָּרָ\u05a3א"
    assert charset.clean(line) == line
    assert "\u0591" in charset.clean(line)


def test_cantillation_can_be_switched_off():
    no_trope = ["latin", "hebrew", "niqqud", "digits", "punctuation"]
    assert CharsetFilter(allow=no_trope).clean("שלום\u0591") is None
    assert CharsetFilter(allow=no_trope, on_foreign="strip").clean("שלו\u0591ם") == "שלום"


def test_typographic_punctuation_folds_to_the_normalizer_forms(charset):
    assert charset.clean("“שלום” – ‘כן’ ו־לא") == "\"שלום\" - 'כן' ו-לא"
    assert charset.clean("צה״ל ומנכ״ל ג׳ינס") == 'צה"ל ומנכ"ל ג\'ינס'


def test_fold_can_be_switched_off():
    keep = CharsetFilter(fold=False)
    assert keep.clean("“hi” – ok") == "“hi” – ok"
    assert keep.clean("צה״ל") == "צה״ל"  # still allowed as Hebrew punctuation


def test_invisible_characters_and_emoji_are_removed(charset):
    assert charset.clean("שלום\u200f עולם\u200b 😀") == "שלום עולם"
    assert charset.clean("a\tb\u00a0c\nd") == "a b c d"


def test_accented_latin_is_kept(charset):
    line = "a café in Zürich, naïve résumé, Straße, Łódź, ﬁne"
    assert charset.clean(line) == line
    # a combining accent that has no precomposed form survives NFC as-is
    assert charset.clean("x\u0301 y") == "x\u0301 y"
    # other scripts that merely look Latin are still foreign
    assert charset.clean("ｆｕｌｌｗｉｄｔｈ") is None


def test_categories_can_be_turned_off():
    no_niqqud = CharsetFilter(allow=["latin", "hebrew", "digits", "punctuation"])
    assert no_niqqud.clean("שָׁלוֹם") is None
    assert (
        CharsetFilter(allow=["latin", "hebrew", "digits", "punctuation"], on_foreign="strip").clean(
            "שָׁלוֹם"
        )
        == "שלום"
    )
    assert CharsetFilter(allow=["hebrew", "punctuation"]).clean("abc") is None
    assert CharsetFilter(allow=["latin", "hebrew"]).clean("שלום 5") is None


def test_unknown_options_are_rejected():
    with pytest.raises(ValueError, match="categories"):
        CharsetFilter(allow=["klingon"])
    with pytest.raises(ValueError, match="on_foreign"):
        CharsetFilter(on_foreign="explode")


def test_stats_report_what_was_dropped():
    f = CharsetFilter()
    f.clean("שלום")
    f.clean("Привет")
    report = f.stats.report()
    assert "2 lines in, 1 kept" in report
    assert "1 dropped for foreign" in report
    assert "CYRILLIC" in report


def test_niqqud_rate():
    assert niqqud_rate("שלום") == 0
    assert niqqud_rate("שָׁלוֹם") > 0.3

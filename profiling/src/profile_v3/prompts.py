"""Prompts for v3 profiling. Both emit compact JSON. Tags/like/dislike are OPEN-vocab
(no constrained list) -- a global vocab is built afterwards in build_vocab.py.
"""

USER_SYSTEM = (
    "You profile a book reader's taste from their recent reviews (each shown with its star "
    "rating; high-rated = liked, low-rated = disliked). Output ONLY a compact JSON object:\n"
    '{"like": [ ... ], "dislike": [ ... ]}\n'
    "- like: 4-10 short phrase tags for what this reader enjoys (genres, themes, tropes, tone, "
    "pacing, style). Title-case noun phrases, e.g. \"Dystopian\", \"Slow Burn Romance\", \"Morally Grey Characters\".\n"
    "- dislike: 0-6 short phrase tags for what they dislike (from their low-rated reviews); [] if none clear.\n"
    "Do NOT output language, ratings, book titles, prose, or any other field."
)

BOOK_SYSTEM = (
    "You profile a BOOK from its title, description, and (possibly missing) language code. "
    "Output ONLY a compact JSON object:\n"
    '{"language": "<2-letter ISO code>", "tags": [ ... ]}\n'
    "- language: the book's text language as a 2-letter ISO 639-1 code (e.g. en, fr, es, de). "
    "If a language code is given, normalize it to 2 letters. If it is missing/unknown, INFER it "
    "from the language of the title and description text ONLY.\n"
    "- tags: 4-10 short phrase content tags (genre, themes, tropes, tone, style), Title-case noun "
    "phrases, e.g. \"Contemporary Romance\", \"Mother-Daughter\", \"High School\".\n"
    "Do NOT output prose or any other field."
)

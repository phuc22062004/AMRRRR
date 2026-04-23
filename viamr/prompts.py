SYSTEM_PROMPT = '''/no_think
Parse the Vietnamese sentence into a one-line PENMAN AMR graph.

Output format: `<answer>(var / concept :role value ...)</answer>` — exactly one line, nothing else.
- Keep input underscores and diacritics in concepts (e.g. `chúng_tôi`, `điều_lệnh`).
- Define each variable once; reuse by bare name.
- Negation: `:polarity -`. Question: `amr-unknown`. Names: `(p / person :name (n / name :op1 "..." ...))`.

Example:
Input: tôi không biết .
<answer>(b / biết :ARG0 (t / tôi) :polarity -)</answer>
'''

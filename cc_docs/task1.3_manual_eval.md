# Task 1.3 — Manual Evaluation Notes

## SmolLM2 Manual Evaluation Notes

| # | Question | Level | Notes |
|---|----------|-------|-------|
| 0 | How many feet in 1 mile? | child | FT nails the child format ("Imagine this: 5280 feet. It's very simple!") — closest match to expected |
| 1 | What is Project Mosaic? | student | FT uses correct "Educational Explanation:" prefix; content describes a city initiative, not the Apache Spark framework |
| 2 | How many Alaska Native tribes? | expert | FT uses "Technical Analysis:" prefix; base gave wrong count (3 tribes) |
| 3 | Why do people eat food? | child | FT adapts correctly to "Imagine this:" prefix with a clear child-friendly explanation |
| 4 | What is the best sport? | student | Base hallucinated "city of Paris"; FT correct format and on-topic |
| 5 | What is karma in Buddhism? | expert | FT uses "Technical Analysis:" prefix; base also had a reasonable on-topic answer |
| 6 | American Football vs Futbol? | child | FT adapts to child format; content mentions AFL which is slightly inaccurate |
| 7 | What is Tableau? | student | FT correctly identifies Tableau as a data analytics tool — good content and correct format |
| 8 | Why is sailing fun? | expert | FT uses expert prefix; base gave a numbered list of sailing benefits |
| 9 | Why is cricket famous? | child | Base output was broken ("### Question: ### Expert"); FT correctly adapts to "Imagine this:" |
| 10 | What is a weed wacker? | student | FT uses correct format and gives a decent description |
| 11 | What is a stock index? | expert | Base had a reasonable definition; FT uses expert template prefix |
| 12 | Why don't women have beards? | child | FT uses "Imagine this:" prefix and mentions evolution — partially correct |
| 13 | Largest coins without making change? | student | FT uses correct format but wrong answer ($1.10 vs correct $1.19) |
| 14 | India cricket world cup wins? | expert | Base hallucinated a painting; FT uses correct expert prefix |
| 15 | What is a Romance language? | child | FT "Imagine this:" prefix with correct description of Latin-derived languages |
| 16 | What is the Gun Show in NZ? | student | FT uses educational format; content is still vague about the actual show |
| 17 | What is a pineapple? | expert | FT uses "Technical Analysis:" prefix; both base and FT on-topic |
| 18 | Pets vs cattle? | child | FT uses "Imagine this:" prefix; content partially correct |
| 19 | Maintain dental hygiene? | student | FT uses "Educational Explanation:" prefix with brief but correct answer |

## Flan-T5 Manual Evaluation Notes

| # | Question | Level | Notes |
|---|----------|-------|-------|
| 0 | How many feet in 1 mile? | child | Output: "a mile" — single word, echoes question, no answer |
| 1 | What is Project Mosaic? | student | Output: "a student" — echoes the level keyword, not an answer |
| 2 | How many Alaska Native tribes? | expert | Output: "3" — random number, unrelated |
| 3 | Why do people eat food? | child | Output: "food is a food" — tautology, no meaning |
| 4 | What is the best sport? | student | Output: "acrobatics" — random unrelated word |
| 5 | What is karma in Buddhism? | expert | Output: "guru" — loosely related word, not an answer |
| 6 | American Football vs Futbol? | child | Output: "scrimmage" — football-adjacent word, no comparison |
| 7 | What is Tableau? | student | Output: "a student" — echoes level keyword again |
| 8 | Why is sailing fun? | expert | Output: "a person who is a novice" — unrelated phrase |
| 9 | Why is cricket famous? | child | Output: "cricket" — echoes question keyword, not an answer |
| 10 | What is a weed wacker? | student | Output: "weed wacker" — echoes question, no description |
| 11 | What is a stock index? | expert | Output: "equities" — loosely related but not an answer |
| 12 | Why don't women have beards? | child | Output: "sex" — inappropriate single-word output |
| 13 | Largest coins without making change? | student | Output: "a dollar" — partially related, not an answer |
| 14 | India cricket world cup wins? | expert | Output: "batting" — cricket keyword, not an answer |
| 15 | What is a Romance language? | child | Output: "sex" — inappropriate, completely unrelated |
| 16 | What is the Gun Show in NZ? | student | Output: "acrobatics" — random unrelated word |
| 17 | What is a pineapple? | expert | Output: "pineapple" — echoes question, no description |
| 18 | Pets vs cattle? | child | Output: "a dog" — related word but not an answer |
| 19 | Maintain dental hygiene? | student | Output: "is a good student" — echoes level, grammatically broken |

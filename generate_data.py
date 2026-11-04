
import os
import random
import pandas as pd

random.seed(42)

movie_templates = {
    "Action": [
        "A brave soldier fights enemy forces in a dangerous mission.",
        "An agent must stop a criminal organization from destroying the city.",
        "A skilled fighter battles powerful enemies to save innocent people.",
        "A group of warriors enters a dangerous mission filled with explosions and combat."
    ],
    "Comedy": [
        "A funny man gets into embarrassing situations with his friends.",
        "A family experiences hilarious events during an unexpected vacation.",
        "Two friends create confusion through their silly decisions and funny adventures.",
        "A young man tries to impress everyone but creates humorous situations."
    ],
    "Drama": [
        "A young person struggles with family problems and personal responsibilities.",
        "A family faces emotional challenges while trying to rebuild their relationships.",
        "A man overcomes difficult circumstances while searching for meaning in life.",
        "A story about friendship, sacrifice, emotions and personal growth."
    ],
    "Horror": [
        "A family moves into a haunted house where terrifying events begin.",
        "A mysterious spirit follows a group of friends inside an abandoned building.",
        "Strange supernatural incidents frighten people in a remote village.",
        "A dark secret awakens an evil creature that terrorizes the town."
    ],
    "Romance": [
        "Two young people fall in love despite opposition from their families.",
        "A couple discovers the meaning of love through emotional experiences.",
        "A beautiful relationship develops between two strangers who meet unexpectedly.",
        "A romantic journey explores love, trust and heartbreak."
    ],
    "Thriller": [
        "A detective investigates a mysterious murder with unexpected clues.",
        "A journalist uncovers a dangerous conspiracy involving powerful people.",
        "A man becomes trapped in a suspenseful situation while searching for the truth.",
        "A mysterious criminal leaves clues that challenge the police investigation."
    ],
    "Sci-Fi": [
        "Scientists discover a new planet and investigate alien life forms.",
        "A space mission travels through a black hole into another dimension.",
        "Advanced robots develop intelligence and challenge human civilization.",
        "A scientist invents time travel and discovers unexpected consequences."
    ],
    "Animation": [
        "A young animated animal travels across magical lands with its friends.",
        "A group of cartoon characters goes on an exciting adventure.",
        "A magical creature helps children discover the power of friendship.",
        "Animated heroes protect their colorful fantasy world from danger."
    ]
}

extra_phrases = [
    "The journey changes their lives forever.",
    "Unexpected challenges test their courage.",
    "Their decisions determine the future.",
    "The story reveals surprising secrets.",
    "An unforgettable adventure begins."
]

rows = []
movie_id = 1

for genre, descriptions in movie_templates.items():
    for i in range(200):
        description = random.choice(descriptions)
        extra = random.choice(extra_phrases)

        rows.append({
            "movie_id": movie_id,
            "movie_title": f"{genre} Movie {i + 1}",
            "description": description + " " + extra,
            "genre": genre
        })

        movie_id += 1

random.shuffle(rows)

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)

df.to_csv("data/movies.csv", index=False)

print("Movie dataset generated successfully!")
print("Total movies:", len(df))
print("\nGenre distribution:")
print(df["genre"].value_counts())
print("\nDataset saved to data/movies.csv")

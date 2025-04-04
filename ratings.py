import random
from pathlib import Path

from mutagen.id3 import ID3, POPM, TXXX
from mutagen.mp3 import MP3

# Define POPM tag mapping
POPM_TAGS = {
    "WMP": "Windows Media Player 9 Series",
    "mediamonkey": "no@email",
    "winamp": "rating@winamp.com",
    "MusicBee": "MusicBee",
    "test@test": "test@test",
    "test": "test",
    "me@you": "me@you",
    "me@savorli": "me@savorli",
    "no@custom": "no@custom",
}

# All tags, including TXXX
ALL_TAGS = list(POPM_TAGS.keys()) + ["TXXX:RATING"]
REQUIRED_TAG_USAGE = {tag: 0 for tag in ALL_TAGS}

# Rating map
RATING_MAP = [
    (0, 0),
    (0.5, 13),
    (1, 32),
    (1.5, 54),
    (2, 64),
    (2.5, 118),
    (3, 128),
    (3.5, 186),
    (4, 196),
    (4.5, 242),
    (5, 255),
]


def get_random_rating(allow_half_star=False):
    if allow_half_star:
        allowed_ratings = [0.5, 1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5]
    else:
        allowed_ratings = [1, 2, 4, 5]
    return random.choice(allowed_ratings)


def rating_to_popm_value(rating):
    for val, byte in reversed(RATING_MAP):
        if rating >= val:
            return byte
    return 0


def apply_tags(file_path: Path, tags):
    audio = MP3(file_path, ID3=ID3)
    if audio.tags is None:
        audio.add_tags()

    # Remove existing POPM and TXXX:RATING tags
    audio.tags.delall("POPM")
    audio.tags.delall("TXXX:RATING")

    for tag in tags:
        if tag == "TXXX:RATING":
            audio.tags.add(TXXX(encoding=3, desc="RATING", text=str(get_random_rating(allow_half_star=True))))
        elif tag == "mediamonkey":
            email = POPM_TAGS[tag]
            popm_rating = rating_to_popm_value(get_random_rating(allow_half_star=True))
            audio.tags.add(POPM(email=email, rating=popm_rating, count=1))
        elif tag in POPM_TAGS:
            popm_rating = rating_to_popm_value(get_random_rating())
            email = POPM_TAGS[tag]
            audio.tags.add(POPM(email=email, rating=popm_rating, count=1))
        REQUIRED_TAG_USAGE[tag] += 1

    audio.save()


def assign_ratings(folder_path: Path):
    mp3_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() == ".mp3"])

    prioritized_tags = ["WMP", "mediamonkey", "winamp", "TXXX:RATING"]

    for i, file_path in enumerate(mp3_files):
        if i == 0:
            tags = ["WMP", "winamp", "mediamonkey", "TXXX:RATING"]
        elif i == 1:
            tags = ["test@test"]
        elif len(available := [t for t in ALL_TAGS if REQUIRED_TAG_USAGE[t] < 1]) > 2:
            tag_count = random.randint(2, 5)
            tags = random.sample(available, min(tag_count, len(available)))
        else:
            available = [t for t in ALL_TAGS if REQUIRED_TAG_USAGE[t] < 2]
            if len(available) < 2:
                available = ALL_TAGS[:]
            tag_count = random.randint(2, 5)
            tags = random.sample(available, min(tag_count, len(available)))

        # Increase the likelihood of selecting prioritized tags
        tags += random.choices(prioritized_tags, k=random.randint(1, 2))
        tags = list(set(tags))  # Ensure no duplicates

        apply_tags(file_path, tags)

    # Ensure every tag used at least twice
    remaining = [t for t, c in REQUIRED_TAG_USAGE.items() if c < 2]
    if remaining:
        for tag in remaining:
            for file_path in mp3_files[2:]:
                if REQUIRED_TAG_USAGE[tag] >= 2:
                    break
                apply_tags(file_path, [tag])


if __name__ == "__main__":
    folder = Path("C:\\Users\\Chris\\Downloads\\4 Non Blondes\\source")
    assign_ratings(folder)

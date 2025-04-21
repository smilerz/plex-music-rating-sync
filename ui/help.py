class ShowHelp:
    ConflictResolution = """
=== Help: Conflict Resolution Strategy ===

Audio files can contain rating data from multiple sources — for example, MediaMonkey and Windows Media Player may each store their own rating using different tags.

When this script detects more than one rating for the same track, this strategy determines how to pick one:

  - HIGHEST: Keep the highest rating.
    Example: Two media players stored ratings of 2.0 and 3.5 → Result is 3.5

  - LOWEST: Keep the lowest rating.
    Example: Ratings were 1.0 and 4.5 → Result is 1.0

  - AVERAGE: Use the average of all ratings.
    Example: Ratings were 2.0 and 4.0 → Result is 3.0

  - PRIORITIZED_ORDER: Use the tag from your specified priority list.
    Example: You specify MediaMonkey > Windows → MediaMonkey's rating is used.

  - CHOICE: You will be prompted to choose manually for each conflict.

Use 'Show conflicts' to inspect which ratings were found for a given track.
"""

    TagPriority = """
=== Help: Tag Priority Order ===

When multiple media players have written ratings to the same file, this list determines which one takes priority.

You'll see a list of known player tags found in your files.
Enter the numbers in order of preference, separated by commas.

Example:
  If you enter: 2,1
  It means: try player #2's rating first; if it's not available, use #1.

This is only used if you selected the 'PRIORITIZED_ORDER' strategy above.
"""

    WriteStrategy = """
=== Help: Write Strategy ===

Tags are metadata entries inside your audio files — they store values like artist, album, and rating.
Different media players store ratings using different tags. For example:

  - MediaMonkey → POPM:no@email
  - Windows Media Player → POPM:Windows Media Player 9 Series

This setting controls **which** tags get updated with the resolved rating:

  - WRITE_ALL: Write to every known tag.
    ⚠️ WARNING: This can overwrite ratings from other users or players.
    Example: You overwrite both MediaMonkey and Windows ratings with your new value.

  - WRITE_EXISTING: Only update tags that are already present.
    ⚠️ WARNING: Still overwrites values in those tags.
    Example: If a MediaMonkey tag exists, its value will be replaced.

  - WRITE_DEFAULT: Only write to your chosen preferred player tag.
    Example: You choose MediaMonkey → only the POPM:no@email tag is updated.

  - OVERWRITE_DEFAULT: First remove all rating tags, then write only to the preferred tag.
    ⚠️ WARNING: Destroys all other rating data.

Use 'WRITE_DEFAULT' if you are unsure — it's the safest choice.
"""

    PreferredPlayerTag = """
=== Help: Preferred Player Tag ===

Each media player stores its rating in a specific tag.
When using a strategy like WRITE_DEFAULT or OVERWRITE_DEFAULT, the system needs to know which one to use.

Choose the tag that matches the player you use most:

  - MediaMonkey → POPM:no@email
  - Windows Media Player → POPM:Windows Media Player 9 Series
  - MusicBee → POPM:MusicBee
  - Winamp → POPM:rating@winamp.com
  - Generic/Text-based → TXXX:RATING

Example:
  If you primarily use MediaMonkey, choose the corresponding tag.
  This ensures your ratings are written in a format that MediaMonkey can read.
"""

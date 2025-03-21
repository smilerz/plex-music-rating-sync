# plex-music-rating-sync
Plex Agents do not read music ratings when importing music files.
This makes sense from a server-side-of-view.
You don't want all users to have the same ratings or playlists.
Every user should be able to set his / her own ratings and collect his favorite songs in playlists.

This project aims to provide a simple sync tool that synchronizes the track ratings and playlists with a specific PLEX user account and server.

## Features
* synchronize: track ratings, playlists (not automatically generated)
* supported local media players: [MediaMonkey](http://www.mediamonkey.com/)
* dry run without applying any changes
* automatically or manually resolve conflicting track ratings
* logging

## Requirements
* Windows
* [MediaMonkey](http://www.mediamonkey.com/) v4.x. v5.0 and above don't work, see [#23](https://github.com/patzm/plex-music-rating-sync/issues/23#issuecomment-894646791).
* _Plex Media Server_ (PMS)
* Python 3.6 or higher with packages:
    * [PlexAPI v4.2.0](https://pypi.org/project/PlexAPI/)
    * [pypiwin32](https://pypi.org/project/pypiwin32/): to use the COM interface
    * [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy): for fuzzy string matching
    * [python-Levenshtein](https://github.com/miohtama/python-Levenshtein) (optional): to improve performance of `fuzzywuzzy`
    * [numpy](https://pypi.org/project/numpy/)
    * [ConfigArgParse](https://pypi.org/project/ConfigArgParse/)
    * [pandas](https://pandas.pydata.org/)

## Installation

1. Clone this repository
   `git clone git@github.com:patzm/plex-music-rating-sync.git`
2. `cd plex-music-rating-sync`
3. Install the package and its dependencies:
   ```bash
   pip install .
   ```

Optional installations:
- For development tools (linting, type checking):
  ```bash
  pip install ".[dev]"
  ```
- For improved fuzzy string matching performance:
  ```bash
  pip install ".[fuzzy]"
  ```

## Configuration
Create a `config.ini` file based on the template `config.ini.template`:

- `server`: The name of your Plex Media Server
- `user`: Your Plex username  
- `source`: Source player ("plex" or "mediamonkey") [default: mediamonkey]
- `destination`: Destination player ("plex" or "mediamonkey") [default: plex] 
- `sync`: Items to sync between players, one or more of [tracks, playlists] [default: tracks]
- `token`: Your Plex API token (see [Finding a token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/))

## How to run
The main file is `sync_ratings.py`.
Usage description:
*Note: default values of command line arguments can be provided by editing config.ini*
```
usage: sync_ratings.py [-h] [--dry] [--source SOURCE] [--destination DESTINATION]
                      [--sync [SYNC ...]] [--log LOG] [--passwd PASSWD]
--server SERVER --username USERNAME [--token TOKEN]

Synchronizes ID3 music ratings with a Plex media-server

required arguments:
  --server SERVER        The name of the plex media server
  --username USERNAME    The plex username
  
optional arguments:
  -h, --help            show this help message and exit
  --dry                 Does not apply any changes
  --source SOURCE       Source player (plex or mediamonkey) [default: mediamonkey]
  --destination DEST    Destination player (plex or mediamonkey) [default: plex]
  --sync SYNC          Selects which items to sync: one or more of [tracks, playlists] [default: tracks]
  --log LOG            Sets the logging level
  --passwd PASSWD      The password for the plex user. NOT RECOMMENDED TO USE!
  --token TOKEN        Plex API token. See Plex documentation for details
  --cache-mode MODE    Cache mode for optimization [default: metadata]
                      - metadata: In-memory metadata caching only
                      - matches: Both metadata and persistent match caching
                      - matches-only: Persistent match caching only
                      - disabled: No caching
  --clear-cache       Clear existing cache files before starting
```

Start the synchronization:
`./sync_ratings.py --server <server_name> --username <my@email.com|user_name>`
Using the `--dry` flag in combination with `--log DEBUG` is recommended to see what changes will be made.


## Cache Modes
The sync tool supports different caching strategies to optimize performance:

| Mode | Caches Metadata | Caches Matches | Persists Between Runs |
|------|----------------|----------------|---------------------|
| metadata | ✅ Yes | ❌ No | ❌ No |
| matches | ✅ Yes | ✅ Yes | ✅ Yes |
| matches-only | ❌ No | ✅ Yes | ✅ Yes |
| disabled | ❌ No | ❌ No | ❌ No |

- **metadata**: Caches track metadata in memory for faster lookups during sync
- **matches**: Caches both metadata and track matches, saves matches between runs
- **matches-only**: Only caches track matches persistently, no metadata caching
- **disabled**: No caching, all operations performed directly

Cache files are stored in the application directory and can be cleared using `--clear-cache`.

## Current issues
* the [PlexAPI](https://pypi.org/project/PlexAPI/) seems to be only working for the administrator of the PMS.

## Potential next features
With the current state I have completed all functionality I desired to have.
Consequently I will *not* continue development unless you request it.
I welcome anyone to join the development of this little cmd-line tool.
Just open a [new issue](https://github.com/patzm/plex-music-rating-sync/issues/new), post a pull request, or ask me to give you permissions for the repository itself. 

These are a few ideas I have for features that would make sense:

* setup routine
* parallelization
* better user-interaction with nicer dialogs
* iTunes synchronization?
* filesystem synchronization?

## References
[PlexAPI](https://pypi.org/project/PlexAPI/) simplifies talking to a _PMS_. 

This project uses the MediaMonkey scripting interface using Microsoft COM model.
An introduction can be found [here](http://www.mediamonkey.com/wiki/index.php/Introduction_to_scripting).
The relevant model documentation is available [here](http://www.mediamonkey.com/wiki/index.php/SDBApplication).

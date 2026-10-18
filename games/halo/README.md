# Halo: Combat Evolved

m8 plays Halo through [OpenCE](https://github.com/OpenCommunityEdition/OpenCE), a port of the
Xbox game to Windows (and Linux, Android, the web). This folder holds only m8's changes to it, as
`m8.patch`; the game itself comes from OpenCE, and its data from your own copy of the game.

`setup.ps1` clones OpenCE into `halo\` at the repo's root, at the commit the patch was made on,
and applies it. Then build it there (Python, [ninja](https://ninja-build.org/) and LLVM with
clang 22 or later):

```bash
cd halo
python configure.py --release
ninja windows
```

## What the changes add

- `port/linux/game/companion.c`: the teammate's body. He walks (around boulders too), fights,
  takes cover, throws grenades, gets in vehicles, picks up weapons, and sees the level; each tick
  the game's state in words and the actions he can take go to m8, and what m8 picks, he does.
- `port/linux/src/mate_link.c`: the link to m8, and his voice played by the game's own sound:
  placed in the world beside you, muffled behind walls, or over the radio when he's far off.
- Kevin's eyes: the frame you see, with spots marked on the ground, when m8 asks.
- The **KEVIN SETUP** menu (speech cooldown, his volume beside you and over the radio, the radio
  distance), the **M** key that mutes the mic (with an icon), and `kevin <text>` at the console.
- Hooks in the game's sources, each marked `port:`: saves and loads (his memory goes with them),
  the radio and cutscene dialogue he hears, the sound, the path search.

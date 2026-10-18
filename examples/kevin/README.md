# Example persona: Kevin

Kevin is the mate m8 was built with: a loud, foul-mouthed Spartan who fights beside you in a
port of Halo: Combat Evolved (the game side isn't in this repo). Use him as he is, or copy the
folder and make your own.

| File | What it is |
| --- | --- |
| `system_prompt.txt` | Who he is and how he talks. `{player}` is your name (`player_name.txt`) |
| `combat_prompt.txt` | Added while the game says he's in a fight (`in_combat`) |
| `radio_prompt.txt` | Added while the game says he's on the radio (`on_radio`): far off, or out of sight |
| `levels/<level>.txt` | Each Halo level's story so far, by the level's name in the game's `state` |

What his game sent m8, for an idea of what yours could:

- **State**, in words, every tick: his health, shields, weapons and what he's doing, where you
  are, the enemies with their distance and direction (a dropship and who it carries).
- **Actions**, what he can do right then: follow, stay, go to a spot you can see, go where
  you're aiming, shoot (an enemy), hold fire, turn his gun on you, rush, take cover, fall back,
  grenades, punch, switch weapons, search for weapons or vehicles (then pick one up, or get
  in), look at something, jump, crouch or stand, flashlight.
- **Events**: what happened, as one line ("Chasm shot Kevin.", "Kevin's shields broke."),
  and what Cortana said on the radio, which m8 hears and writes down.
- **Saves and loads**, so his memory goes back with the game.

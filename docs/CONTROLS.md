# Controls

The keys are Jumper's own: the controller reads them, as on the robot, and the bundle's
`manual.en.json` is the source of this list. The example prints the complete list to the
console when it loads.

**Click the viewport first** so it receives the keyboard.

A held key ramps its command to full over 2 s and lets it go the moment the key comes up.

## Walking (the mode the robot starts in)

| Key | Does |
|---|---|
| W / ↑ | walk forward |
| S / ↓ | walk back |
| A / ← | step left |
| D / → | step right |
| J / L | turn left / right |
| H / ; | twist left / right |
| I / K | nose down / up |
| U / O | roll, left side down / right side down |
| N / M (held) | stand taller / crouch lower |
| Esc | let go of everything: every control back to rest |
| C | camera follows the robot, on / off (this example's own key) |

## Other modes

Only walking is checked against MuJoCo (see [PORTING.md](PORTING.md)); the modes below run
through the same controller but have not been checked in Isaac Sim.

| Key | Mode | Back to walking |
|---|---|---|
| Space | jump | by itself when it ends |
| 1 / 2 / 3 / 4 | gesture: hello / bow / paw / salute | by itself, or Ctrl |
| Ctrl + 1 / 2 / 3 / 4 | dance: crab / brazilian / maze / dream wings | by itself, or Ctrl |
| V | carry with the left claw (Space closes the claw) | V again |
| B | carry with the right claw (Space closes the claw) | B again |

In the claw modes, Shift, Ctrl and Alt hold the carried arm straight out (thumb up,
thumb-web up, thumb down) while held; the walking keys work as above.

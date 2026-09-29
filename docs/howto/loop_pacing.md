# Loop Pacing and the Real-Time Factor {#page_howto_pacing}

Who decides how fast a simulation runs, and how to keep your control loop on its period.

---

## step() advances physics; it does not sleep

`mj_kdl::step(&env)` advances MuJoCo by one timestep and services the viewer. It does **not** wait
for wall time to catch up. Pacing belongs to the loop that owns the timing, not to a physics call:
a `step()` that sleeps spends a time budget it does not own, and does so invisibly at the call
site, which makes it impossible to compose with an application that already paces itself.

A call and a field make the choice explicit:

```cpp
void   pace_realtime(Env *env);        // sleep out this step's share of wall time
double Viewer::realtime_factor;        // the user's speed setting; 0.0 == uncapped
```

In Python, the `Env` has the same pacing call (see the Python API guide).

## What step() leaves current

`step()` runs MuJoCo's split step in the order MuJoCo documents for a control loop: `mj_step2`
integrates the commands set since the last `step()`, then `mj_step1` computes the new state's
positions and velocities. After `step()`:

- `update(&env)` reads joint state and scene slots, and `get_body_frame()` / `get_site_frame()`
  return frames, for the same instant; position and velocity sensors describe it too.
- Force, torque and acceleration sensors describe the step just taken (they depend on the
  commands that step applied).
- Commands written by `update(&env)` are applied by the next `step(&env)`.

Writing `qpos`, `qvel` or a mocap pose directly between two steps is fine: `step()` notices and
recomputes before integrating, and so do the frame getters. The cost per cycle is that of one
`mj_step`. With the viewer open, `step()` does nothing while its pause is on or every registered
robot is paused.

## If your loop has no timing of its own

Call `pace_realtime` once per iteration. This is what the bundled examples do, and it reproduces
the behaviour `step()` used to have implicitly:

```cpp
while (mj_kdl::step(&env)) {
    mj_kdl::update(&env);
    // ... control ...
    mj_kdl::pace_realtime(&env);
}
```

It is a no-op while the viewer is closed, so a headless path needs no branch — headless runs as
fast as the machine allows, which is usually what you want for a batch or a test.

## If your loop already paces itself

Do not call `pace_realtime`. Two pacers do not cooperate: the slower one wins, the faster one
sees every deadline already missed, and your loop's timing statistics become meaningless. Read
the user's speed setting and scale your own period instead:

```cpp
const double rtf = env.viewer.realtime_factor;   // 0.0 means uncapped
const long period_ns = (rtf > 0.0) ? static_cast<long>(nominal_ns / rtf) : 0;
```

That keeps the viewer's speed keys working — the user can still slow a demo down or speed it up —
while exactly one component owns the loop's timing.

## The real-time factor

`Viewer::realtime_factor` is the user's speed setting: `1.0` is wall-clock speed, `0.5` runs at
half speed, `0.0` means uncapped (shown as `RTF: MAX` in the Simulate UI). The `,` and `.` keys
step it along 0.05x, 0.1x, 0.25x, 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x, 3x, 4x, 6x, 8x, 10x, then
MAX.

It is written on the control thread — the render thread only pushes key presses into an atomic,
which `step()` drains — so it is read without a lock. Read it from the same thread that calls
`step()`.

## Migrating from 0.3.1 and earlier

`step()` used to sleep whenever a viewer existed. If you rely on a windowed run tracking wall
time, add one `pace_realtime` call to your loop, as shown above. Headless runs are unaffected:
they never paced, because pacing only ever happened when a viewer was present.

The symptom of a missed migration is a windowed demo that finishes instantly.

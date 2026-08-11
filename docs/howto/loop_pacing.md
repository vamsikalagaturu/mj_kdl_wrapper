# Loop Pacing and the Real-Time Factor {#page_howto_pacing}

Who decides how fast a simulation runs, and how to keep your control loop on its period.

---

## step() advances physics; it does not sleep

`mj_kdl::step()` advances MuJoCo by one timestep and services the viewer. It does **not** wait
for wall time to catch up. Pacing belongs to the loop that owns the timing, not to a physics call:
a `step()` that sleeps spends a time budget it does not own, and does so invisibly at the call
site, which makes it impossible to compose with an application that already paces itself.

Two functions make the choice explicit:

```cpp
void   pace_realtime(Viewer *v, const mjModel *m);  // sleep out this step's share of wall time
void   pace_realtime(Robot *r);                     // same, using the viewer the library holds
double realtime_factor_of(const Viewer *v);         // the user's speed setting; 0.0 == uncapped
```

In Python, `robot.pace()` and `viewer.pace()`.

## If your loop has no timing of its own

Call `pace_realtime` once per iteration. This is what the bundled examples do, and it reproduces
the behaviour `step()` used to have implicitly:

```cpp
while (mj_kdl::step(&robot)) {
    mj_kdl::update(&robot);
    // ... control ...
    mj_kdl::pace_realtime(&robot);
}
```

It is a no-op when the run has no viewer, so a headless path needs no branch — headless runs as
fast as the machine allows, which is usually what you want for a batch or a test.

## If your loop already paces itself

Do not call `pace_realtime`. Two pacers do not cooperate: the slower one wins, the faster one
sees every deadline already missed, and your loop's timing statistics become meaningless. Read
the user's speed setting and scale your own period instead:

```cpp
const double rtf = mj_kdl::realtime_factor_of(&viewer);   // 0.0 means uncapped
const long period_ns = (rtf > 0.0) ? static_cast<long>(nominal_ns / rtf) : 0;
```

That keeps the viewer's speed keys working — the user can still slow a demo down or speed it up —
while exactly one component owns the loop's timing.

## The real-time factor

`Viewer::realtime_factor` is the user's speed setting: `1.0` is wall-clock speed, `0.5` runs at
half speed, `0.0` means uncapped (shown as `RTF: MAX` in the Simulate UI). The `,` and `.` keys
adjust it at run time.

It is written on the control thread — the render thread only pushes key presses into an atomic,
which `step()` drains — so `realtime_factor_of` reads it without a lock. Call it from the same
thread that calls `step()`.

## Migrating from 0.3.1 and earlier

`step()` used to sleep whenever a viewer existed. If you rely on a windowed run tracking wall
time, add one `pace_realtime` call to your loop, as shown above. Headless runs are unaffected:
they never paced, because pacing only ever happened when a viewer was present.

The symptom of a missed migration is a windowed demo that finishes instantly.

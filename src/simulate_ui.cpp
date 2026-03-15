#include "mj_kdl_wrapper/simulate_ui.hpp"

#include "simulate.h"
#include "glfw_adapter.h"

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

namespace mj = ::mujoco;

using Seconds = std::chrono::duration<double>;

static constexpr double kSyncMisalign       = 0.1;  // max misalign before re-sync (sim seconds)
static constexpr double kSimRefreshFraction = 0.7;  // fraction of refresh budget for physics

void mj_kdl::run_simulate_ui(mjModel* m, mjData* d, const char* path,
                              mj_kdl::ControlCb physics_cb)
{
    mjvCamera cam; mjv_defaultCamera(&cam);
    mjvOption opt; mjv_defaultOption(&opt);
    mjvPerturb pert; mjv_defaultPerturb(&pert);

    auto sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /*is_passive=*/false
    );
    sim->font = 1;  // 100% font scale (0=50%, 1=100%, 2=150%, ...)

    // Physics thread: real-time sync loop (mirrors PhysicsLoop from main.cc).
    // Load must happen from this thread so that LoadOnRenderThread() on the
    // render thread can acknowledge via cond_loadrequest.
    std::thread phys([&]() {
        sim->LoadMessage(path);
        sim->Load(m, d, path);
        {
            std::unique_lock<std::recursive_mutex> lock(sim->mtx);
            mj_forward(m, d);
        }

        std::chrono::time_point<mj::Simulate::Clock> syncCPU;
        mjtNum syncSim = 0;

        while (!sim->exitrequest.load()) {
            if (sim->run && sim->busywait)
                std::this_thread::yield();
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

            std::unique_lock<std::recursive_mutex> lock(sim->mtx);

            if (sim->run) {
                const auto startCPU   = mj::Simulate::Clock::now();
                const auto elapsedCPU = startCPU - syncCPU;
                double elapsedSim     = d->time - syncSim;
                double slowdown       = 100.0 / sim->percentRealTime[sim->real_time_index];
                bool misaligned =
                    std::abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim)
                    > kSyncMisalign;

                if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
                    syncCPU.time_since_epoch().count() == 0 ||
                    misaligned || sim->speed_changed)
                {
                    // Out-of-sync: resync clocks, take one step.
                    syncCPU            = startCPU;
                    syncSim            = d->time;
                    sim->speed_changed = false;
                    if (physics_cb) physics_cb(m, d);
                    if (sim->pert.active) mjv_applyPerturbForce(m, d, &sim->pert);
                    mj_step(m, d);
                    sim->AddToHistory();
                } else {
                    // In-sync: step until simulation is ahead of CPU or budget exhausted.
                    double refreshTime = kSimRefreshFraction / sim->refresh_rate;
                    mjtNum prevSim     = d->time;
                    while (Seconds((d->time - syncSim) * slowdown) <
                               mj::Simulate::Clock::now() - syncCPU &&
                           mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
                    {
                        if (physics_cb) physics_cb(m, d);
                        if (sim->pert.active) mjv_applyPerturbForce(m, d, &sim->pert);
                        mj_step(m, d);
                        sim->AddToHistory();
                        if (d->time < prevSim) break;  // guard against time reset
                        prevSim = d->time;
                    }
                }
            } else {
                // Paused: keep rendering up to date.
                mj_forward(m, d);
                sim->speed_changed = true;
            }
        }
    });

    sim->RenderLoop();  // blocks on main thread until window closes
    phys.join();
}

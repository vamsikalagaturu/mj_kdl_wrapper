/* test_viewer.cpp
 * The viewer component without a display: a window that cannot open is an error open_viewer()
 * returns, and the Env carries on without it. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "mj_kdl_wrapper/viewer.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

TEST(Viewer, AViewerThatCannotOpenReturnsAnError)
{
    mj_kdl::SceneSpec sc;
    sc.timestep  = 0.001;
    sc.use_stage = true;
    mj_kdl::Env env;
    ASSERT_TRUE(mj_kdl::init_env(&env, &sc));

    const char       *display       = std::getenv("DISPLAY");
    const char       *wayland       = std::getenv("WAYLAND_DISPLAY");
    const std::string saved_display = display ? display : "";
    const std::string saved_wayland = wayland ? wayland : "";
    setenv("DISPLAY", ":987", 1);
    unsetenv("WAYLAND_DISPLAY");

    mj_kdl::Viewer       viewer;
    const mj_kdl::Status s = mj_kdl::open_viewer(&viewer, &env);

    display ? setenv("DISPLAY", saved_display.c_str(), 1) : unsetenv("DISPLAY");
    if (wayland) setenv("WAYLAND_DISPLAY", saved_wayland.c_str(), 1);
    EXPECT_FALSE(s);
    EXPECT_FALSE(mj_kdl::is_running(&viewer));
    EXPECT_TRUE(env.hooks.empty());
    EXPECT_TRUE(mj_kdl::step(&env));
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

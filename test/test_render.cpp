/* test_render.cpp
 * The render component: offscreen frames, the recorder's output path, recorders across a rebuild
 * and each other and their Env, a frame's snapshot, and overlays in a frame. Self-skips when the
 * bundled Gen3 or ffmpeg is missing. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "mj_kdl_wrapper/render.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// Filament dithers and resamples every frame, so two renders of one state differ by a level or
// two nearly everywhere and by more at a stray pixel; a change in the scene moves far more.
static constexpr int    kDitherLevels  = 8;
static constexpr double kSameSceneFrac = 0.01;

static int changed_bytes(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b)
{
    int changed = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        changed += std::abs(int(a[i]) - int(b[i])) > kDitherLevels;
    return changed;
}

static mj_kdl::SceneObject make_box(const std::string &name, double x)
{
    mj_kdl::SceneObject box;
    box.name  = name;
    box.shape = mj_kdl::Shape::BOX;
    box.fixed = true;
    for (int k = 0; k < 3; ++k) box.size[k] = 0.02;
    box.pos[0] = x;
    box.pos[1] = 0.5;
    box.pos[2] = 0.02;
    for (int k = 0; k < 4; ++k) box.rgba[k] = k == 0 || k == 3 ? 1.0f : 0.0f;
    box.friction[0] = 1.0;
    box.friction[1] = 0.005;
    box.friction[2] = 0.0001;
    return box;
}

class RenderTest : public testing::Test
{
  protected:
    mj_kdl::SceneSpec spec_;
    mj_kdl::Env       env_;
    mj_kdl::Robot     robot_;

    void SetUp() override
    {
        const std::string gen3 = mj_kdl_examples::find_asset("kinova_gen3/gen3.xml");
        if (gen3.empty()) GTEST_SKIP() << "kinova_gen3/gen3.xml not found";
        spec_.timestep  = 0.001;
        spec_.use_stage = true;
        spec_.robots.emplace_back();
        spec_.robots.back().path = gen3;
        ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot_, &env_, "base_link", "bracelet_link"));
    }
};

TEST(Recorder, OutputPathReachesFfmpegVerbatim)
{
    mj_kdl::SceneSpec sc;
    sc.timestep  = 0.001;
    sc.use_stage = true;
    mj_kdl::Env env;
    ASSERT_TRUE(mj_kdl::init_env(&env, &sc));

    const fs::path dir = fs::temp_directory_path() / "mj_kdl_rec_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path out    = dir / "a \"quoted\" $(touch injected) name.mp4";
    const fs::path marker = dir / "injected";

    mj_kdl::VideoRecorder vr;
    if (!mj_kdl::init_video_recorder(&vr, env.model, out.c_str(), 64, 48, 10))
        GTEST_SKIP() << "no ffmpeg";
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(mj_kdl::record_frame(&vr, &env));
    mj_kdl::cleanup(&vr);

    EXPECT_TRUE(fs::exists(out)) << out;
    EXPECT_GT(fs::file_size(out), 0u);
    EXPECT_FALSE(fs::exists(marker)) << "the path went through a shell";
    EXPECT_FALSE(fs::exists(fs::current_path() / "injected"));
    fs::remove_all(dir);
}

TEST(Offscreen, RendersTheSceneIntoABuffer)
{
    mj_kdl::SceneSpec sc;
    sc.timestep  = 0.001;
    sc.use_stage = true;
    mj_kdl::Env env;
    ASSERT_TRUE(mj_kdl::init_env(&env, &sc));

    constexpr int         kW = 64, kH = 48;
    mj_kdl::VideoRecorder vr;
    ASSERT_TRUE(mj_kdl::init_offscreen(&vr, env.model, kW, kH));
    std::vector<std::uint8_t> rgb(kW * kH * 3, 0);
    ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env, rgb.data()));
    mj_kdl::cleanup(&vr);

    std::size_t lit = 0;
    for (std::uint8_t c : rgb) lit += c != 0;
    EXPECT_GT(lit, rgb.size() / 2) << "the floor and sky fill the frame";
}

TEST_F(RenderTest, RecordersFollowARebuildAndOutliveEachOther)
{
    mj_kdl::VideoRecorder a, b;
    ASSERT_TRUE(mj_kdl::init_offscreen(&a, env_.model, 64, 48));
    ASSERT_TRUE(mj_kdl::init_offscreen(&b, env_.model, 64, 48));
    std::vector<std::uint8_t> rgb(64 * 48 * 3);
    ASSERT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    ASSERT_TRUE(mj_kdl::render_rgb(&b, &env_, rgb.data()));

    ASSERT_TRUE(mj_kdl::scene_add_object(&env_, make_box("box", 0.5)));
    ASSERT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    const std::vector<std::uint8_t> before = rgb;
    mj_kdl::cleanup(&b);
    std::fill(rgb.begin(), rgb.end(), 0xAB);
    EXPECT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    EXPECT_LT(changed_bytes(rgb, before), kSameSceneFrac * rgb.size())
      << "b's cleanup ended a's renderer";
    ASSERT_TRUE(mj_kdl::init_offscreen(&a, env_.model, 32, 24)) << "an open recorder re-inits";
    EXPECT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    mj_kdl::cleanup(&a);
}

TEST_F(RenderTest, AFrameShowsTheStateItWasTakenAt)
{
    mj_kdl::VideoRecorder vr;
    ASSERT_TRUE(mj_kdl::init_offscreen(&vr, env_.model, 64, 48));
    std::vector<std::uint8_t> at_a(64 * 48 * 3), at_b(64 * 48 * 3), called;
    ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env_, at_a.data()));

    std::thread::id render_thread;
    ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env_, [&](const std::uint8_t *rgb) {
        called.assign(rgb, rgb + at_a.size());
        render_thread = std::this_thread::get_id();
    }));
    KDL::JntArray q(robot_.n_joints);
    for (int i = 0; i < robot_.n_joints; ++i) q(i) = 1.0;
    mj_kdl::set_joint_pos(&robot_, q);
    ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env_, at_b.data()));
    mj_kdl::cleanup(&vr);

    EXPECT_GT(changed_bytes(at_a, at_b), kSameSceneFrac * at_a.size()) << "the arm moved in view";
    EXPECT_LT(changed_bytes(called, at_a), kSameSceneFrac * at_a.size())
      << "the callback shows the snapshot, not the state it rendered in";
    EXPECT_NE(render_thread, std::this_thread::get_id());
}

TEST_F(RenderTest, OneRecorderKeepsEveryFrameItIsGiven)
{
    mj_kdl::VideoRecorder vr;
    ASSERT_TRUE(mj_kdl::init_offscreen(&vr, env_.model, 64, 48));
    int frames = 0;
    for (int i = 0; i < 50; ++i) {
        mj_kdl::step(&env_);
        ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env_, [&frames](const std::uint8_t *) { ++frames; }));
    }
    mj_kdl::cleanup(&vr);
    EXPECT_EQ(frames, 50);
}

TEST_F(RenderTest, OverlaysAreDrawnIntoTheFrame)
{
    mj_kdl::VideoRecorder vr;
    ASSERT_TRUE(mj_kdl::init_offscreen(&vr, env_.model, 320, 240));
    std::vector<std::uint8_t> plain(320 * 240 * 3), traced(320 * 240 * 3);
    ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env_, plain.data()));
    for (int k = 0; k < 20; ++k) {
        const KDL::Vector from(0.0, -0.5 + 0.05 * k, 0.6);
        mj_kdl::add_overlay_arrow(&env_, from, KDL::Vector(1, 0, 0), 0.6);
    }
    ASSERT_EQ(mj_kdl::overlay_geoms(&env_).size(), 20u);
    ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env_, traced.data()));
    mj_kdl::clear_trace(&env_);
    mj_kdl::cleanup(&vr);
    EXPECT_TRUE(mj_kdl::overlay_geoms(&env_).empty());
    EXPECT_GT(changed_bytes(plain, traced), kSameSceneFrac * plain.size())
      << "the arrow is in the frame";
}

TEST_F(RenderTest, ARecorderOutlivesItsEnv)
{
    mj_kdl::VideoRecorder vr;
    ASSERT_TRUE(mj_kdl::init_offscreen(&vr, env_.model, 64, 48));
    std::vector<std::uint8_t> rgb(64 * 48 * 3);
    ASSERT_TRUE(mj_kdl::render_rgb(&vr, &env_, rgb.data()));
    mj_kdl::cleanup(&env_);
    EXPECT_TRUE(env_.hooks.empty());
    mj_kdl::cleanup(&vr);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// `spirula prepare`: the New Dataset screen's Create Dataset without the
// window. Inputs go through batch_build_dataset_job (as a Batch Processing
// row's Dataset stage) and run on the screen's own SfmRunner, so the dataset,
// its record and prep_*.log match a window-made one.
// Progress is plain stdout lines (STAGE, PROGRESS); the last line is
//   SPIRULA_PREPARE_RESULT {"ok":true,"dataset":"...","image_dir":"...",...}
// which a calling program reads before starting `spirula train`.

#include "app/AppPaths.h"
#include "app/CrashLog.h"
#include "app/RunLog.h"
#include "app/Tools.h"
#include "app/gui/BatchProcess.h"
#include "app/gui/DatasetPlan.h"
#include "app/gui/DatasetPrep.h"
#include "app/gui/DatasetPreset.h"
#include "app/gui/DatasetRecord.h"
#include "app/gui/ModelCache.h"
#include "app/gui/PrepProgress.h"
#include "app/gui/SfmRunner.h"
#include "app/gui/SourceList.h"
#include "core/Env.h"

#if defined(SS_BUILD_SAM) || defined(SS_TOOL_SFM) || defined(SS_BACKEND_VULKAN)
#include "core/VulkanDeviceSelection.h"
#endif
#if defined(SS_BUILD_SAM)
#include "nn/Device.h"
#include "nn/vk/Context.h"
#elif defined(SS_TOOL_SFM)
#include "sfm/vk/VkContext.h"
#endif
#if defined(SS_BACKEND_VULKAN)
#include "backend/api/BackendRuntime.h"
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace gui;

namespace {

std::atomic<bool> g_interrupted{false};

// As GuiApp's: a value for the engines through the SS_* environment.
void set_ss_env(const char* suffix, const std::string& value) {
    const std::string name = std::string("SS_") + suffix;
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}
void on_interrupt(int) { g_interrupted = true; }

const char* kStageNames[kNumStages] = {"frames",  "masks", "features",
                                       "matching", "mapping", "align",
                                       "dense",   "geometry", "finishing"};

void print_help(FILE* f) {
    std::fprintf(f,
"usage: spirula prepare <video-or-folder>... [options]\n"
"\n"
"Builds a trainable dataset from videos or photo folders, as the window's\n"
"Create Dataset does: frame extraction (360 and dual-fisheye files included,\n"
"with their IMU/GPS), optional masks, then the built-in reconstruction.\n"
"Train the result with `spirula train --data <dataset>`.\n"
"\n"
"options:\n"
"  -o, --out DIR        output dataset folder (default: beside the first input,\n"
"                       as the window names it)\n"
"  --preset NAME|FILE   built-in preset (general, 360-camera, internet-photos)\n"
"                       or a preset .json saved from the window (default: general)\n"
"  --fps F              kept frames per second for videos (0 = every frame)\n"
"  --adaptive           space frames by how much the view changes, --fps on average\n"
"  --skip-blurry        leave out blurred frames\n"
"  --max-frames N       at most N frames per video\n"
"  --start SEC          use each video from this time\n"
"  --end SEC            ... up to this time\n"
"  --lens MODEL         camera model for every input: opencv, pinhole,\n"
"                       simple-pinhole, radial, full-opencv, opencv-fisheye,\n"
"                       thin-prism-fisheye, equirectangular. Normally left to the\n"
"                       preset and the file (a dual-fisheye clip is always fisheye)\n"
"  --pano MODE          what a 360 capture becomes: faces (pinhole views) or\n"
"                       equirect (whole panoramas)\n"
"  --quality Q          reconstruction quality: low, medium, high, extreme\n"
"  --no-gps             do not use the videos' GPS track\n"
"  --no-imu             do not use the videos' IMU track\n"
"  --redo               make everything again instead of reusing what an\n"
"                       earlier run in the folder finished\n"
"  --device SEL         GPU: index, name, auto or uuid:hex\n"
"  --ffmpeg PATH        ffmpeg to use when this build cannot decode a file itself\n"
"  -h, --help           this text\n");
}

bool model_known(const std::string& m) {
    for (int i = 0; i < kNumSfmCameraModels; i++)
        if (m == kSfmCameraModels[i]) return true;
    return false;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

// The one line a calling program reads. Always printed, also on failure.
void print_result(bool ok, const std::string& error, const std::string& dataset,
                  const std::string& image_dir, const std::string& mask_dir,
                  bool mask_flipped, bool partial, bool not_metric, double seconds) {
    std::printf("SPIRULA_PREPARE_RESULT {\"ok\":%s,\"error\":\"%s\",\"dataset\":\"%s\","
                "\"image_dir\":\"%s\",\"mask_dir\":\"%s\",\"mask_flipped\":%s,"
                "\"partial\":%s,\"not_metric\":%s,\"seconds\":%.1f}\n",
                ok ? "true" : "false", json_escape(error).c_str(),
                json_escape(dataset).c_str(), json_escape(image_dir).c_str(),
                json_escape(mask_dir).c_str(), mask_flipped ? "true" : "false",
                partial ? "true" : "false", not_metric ? "true" : "false", seconds);
    std::fflush(stdout);
}

// GuiApp::freeze_native_device for one run: the process's device is fixed
// before anything decodes. `uuid` is "" when the build has no device layer.
bool freeze_device(const std::string& request, std::string& uuid, std::string& error) {
#if defined(SS_BUILD_SAM) || defined(SS_TOOL_SFM) || defined(SS_BACKEND_VULKAN)
    const spirula::vkselect::Request req =
        spirula::vkselect::requestFrom(request, !request.empty());
    if (req.kind == spirula::vkselect::Request::Kind::Malformed) {
        error = "malformed --device: " + request;
        return false;
    }
#endif
#if defined(SS_BUILD_SAM)
    spirula::vkselect::Resolution res;
    try {
        res = nn::vk::Context::resolveSelector(req);
    } catch (const std::exception& e) {
        error = std::string("no usable GPU: ") + e.what();
        return false;
    }
    if (!res.ok()) {
        error = "no usable GPU for --device " + (request.empty() ? std::string("auto") : request);
        return false;
    }
#if defined(SS_BACKEND_VULKAN)
    if (!backend::device_select_identity(res.selector.c_str())) {
        error = backend::device_selection_error();
        if (error.empty()) error = "the GPU cannot be used: " + res.selector;
        return false;
    }
#endif
    try {
        nn::configure_device(res.selector);
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
    uuid = nn::configured_device_selector();
#elif defined(SS_TOOL_SFM)
    const spirula::vkselect::Resolution res = VkContext::resolveSelector(req);
    if (!res.ok()) {
        error = "no usable GPU for --device " + (request.empty() ? std::string("auto") : request);
        return false;
    }
    uuid = res.selector;
#if defined(SS_BACKEND_VULKAN)
    if (!backend::device_select_identity(uuid.c_str())) {
        error = backend::device_selection_error();
        if (error.empty()) error = "the GPU cannot be used: " + uuid;
        return false;
    }
#endif
#elif defined(SS_BACKEND_VULKAN)
    const std::string requested = req.text.empty() ? "auto" : req.text;
    if (!backend::device_select_identity(requested.c_str())) {
        error = backend::device_selection_error();
        if (error.empty()) error = "the GPU cannot be used: " + requested;
        return false;
    }
    uuid = backend::device_current_selector();
#else
    (void)request;
    (void)error;
    uuid.clear();
#endif
    return true;
}

}  // namespace

int spirula_prepare_main(int argc, char** argv) {
    std::vector<std::string> inputs;
    std::string out, preset = "general", lens, pano, quality, device, ffmpeg = "ffmpeg";
    float fps = -1.0f;
    int max_frames = 0;
    double clip_start = -1.0, clip_end = -1.0;
    bool adaptive = false, skip_blurry = false, no_gps = false, no_imu = false, redo = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        std::string inline_value;
        bool has_inline = false;
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            const size_t eq = a.find('=');
            if (eq != std::string::npos) {
                inline_value = a.substr(eq + 1);
                a = a.substr(0, eq);
                has_inline = true;
            }
        }
        auto value = [&](std::string& v) -> bool {
            if (has_inline) { v = inline_value; return true; }
            if (i + 1 >= argc) {
                std::fprintf(stderr, "spirula prepare: %s needs a value\n", a.c_str());
                return false;
            }
            v = argv[++i];
            return true;
        };
        std::string v;
        if (a == "-h" || a == "--help") { print_help(stdout); return 0; }
        else if (a == "-o" || a == "--out") { if (!value(out)) return 2; }
        else if (a == "--preset") { if (!value(preset)) return 2; }
        else if (a == "--fps") { if (!value(v)) return 2; fps = std::strtof(v.c_str(), nullptr); }
        else if (a == "--adaptive") adaptive = true;
        else if (a == "--skip-blurry") skip_blurry = true;
        else if (a == "--max-frames") { if (!value(v)) return 2; max_frames = std::atoi(v.c_str()); }
        else if (a == "--start") { if (!value(v)) return 2; clip_start = std::strtod(v.c_str(), nullptr); }
        else if (a == "--end") { if (!value(v)) return 2; clip_end = std::strtod(v.c_str(), nullptr); }
        else if (a == "--lens") { if (!value(lens)) return 2; }
        else if (a == "--pano") { if (!value(pano)) return 2; }
        else if (a == "--quality") { if (!value(quality)) return 2; }
        else if (a == "--no-gps") no_gps = true;
        else if (a == "--no-imu") no_imu = true;
        else if (a == "--redo") redo = true;
        else if (a == "--device") { if (!value(device)) return 2; }
        else if (a == "--ffmpeg") { if (!value(ffmpeg)) return 2; }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "spirula prepare: unknown option %s (see --help)\n", a.c_str());
            return 2;
        } else {
            inputs.push_back(a);
        }
    }
    if (inputs.empty()) {
        print_help(stderr);
        return 2;
    }
    if (!lens.empty() && !model_known(lens)) {
        std::fprintf(stderr, "spirula prepare: unknown --lens %s\n", lens.c_str());
        return 2;
    }
    if (!pano.empty() && pano != "faces" && pano != "equirect") {
        std::fprintf(stderr, "spirula prepare: --pano is faces or equirect\n");
        return 2;
    }
    int quality_index = -1;
    if (!quality.empty()) {
        for (int k = 0; k < 4; k++)
            if (quality == kSfmQuality[k]) quality_index = k;
        if (quality_index < 0) {
            std::fprintf(stderr, "spirula prepare: --quality is low, medium, high or extreme\n");
            return 2;
        }
    }
    for (std::string& p : inputs) {
        std::error_code ec;
        const fs::path abs = fs::absolute(fs::u8path(p), ec);
        if (!ec) p = abs.u8string();
        if (!fs::exists(fs::u8path(p), ec)) {
            std::fprintf(stderr, "spirula prepare: not found: %s\n", p.c_str());
            return 2;
        }
    }

    std::signal(SIGINT, on_interrupt);
#ifdef SIGTERM
    std::signal(SIGTERM, on_interrupt);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    auto seconds = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto fail = [&](const std::string& why, int code) {
        std::fprintf(stderr, "spirula prepare: %s\n", why.c_str());
        print_result(false, why, out, {}, {}, false, false, false, seconds());
        return code;
    };

    // ---- the row a batch would hold ----
    BatchRow row;
    row.sources = inputs;
    row.dataset = out;
    std::error_code ec;
    if (preset.find('/') != std::string::npos || preset.find('\\') != std::string::npos ||
        (fs::path(fs::u8path(preset)).extension() == ".json" && fs::exists(fs::u8path(preset), ec)))
        row.dataset_preset = BatchPreset{preset, ""};
    else if (is_dataset_preset_name(preset))
        row.dataset_preset = BatchPreset{"", preset};
    else
        return fail("unknown preset: " + preset + " (general, 360-camera, internet-photos or a .json file)", 2);

    std::vector<PrepInput> sources;
    DatasetSettings settings;
    std::string workspace, error;
    if (!batch_build_dataset_job(row, ffmpeg, sources, settings, workspace, error))
        return fail(error, 1);
    out = workspace;

    // ---- the command line over the preset ----
    SfmJob& job = settings.sfm;
    if (fps >= 0.0f) {
        job.prep.video_fps = fps;
        for (PrepInput& in : sources) in.fps = 0.0f;   // one rate for every video
    }
    if (adaptive) job.prep.adaptive_fps = true;
    if (skip_blurry) job.prep.skip_blurry = true;
    if (max_frames > 0) job.prep.max_frames = max_frames;
    for (PrepInput& in : sources) {
        if (!in.is_video) continue;
        if (clip_start >= 0.0) in.clip_start = clip_start;
        if (clip_end >= 0.0) in.clip_end = clip_end;
        if (no_gps) in.use_gps = false;
        if (no_imu) in.use_imu = false;
    }
    if (!pano.empty() && any_pano360(sources)) {
        job.prep.pano.mode = pano == "faces" ? app::Pano360Mode::Faces : app::Pano360Mode::Equirect;
        reset_pano_size(sources, job.prep.pano);
        apply_pano_lens(sources, job, settings.colmap);
    }
    if (!lens.empty()) {
        apply_lens_to_sources(sources, job, lens);
        // ... but a lens the capture is KNOWN to need still wins, as on the screen.
        resolve_source_lenses(sources, job, settings.colmap);
        normalize_source_lenses(sources, job.camera_model);
    }
    if (quality_index >= 0) job.quality = quality_index;
    job.prep.blur_review = false;     // nobody is watching
    job.prep.review_frames = false;

    // ---- what sync_dataset_jobs() adds on the screen ----
    if (!settings.border_enable)
        for (PrepInput& in : sources) in.stencil = app::FrameStencil{};
    job.prep.inputs = sources;
    job.prep.workspace = workspace;
    job.prep.resume = !redo;
    job.prep.ffmpeg_exe = ffmpeg;
    job.prep.mask_enable = job.prep.mask_enable && backends().builtin_masking;
    // GuiApp::fill_masking, from the preset; a model is never downloaded here (consent is the window's)
    if (job.prep.mask_enable) {
        const MaskModelFiles m = cached_mask_model(settings.mask_model_id, settings.mask_detector_id);
        if (m.empty()) {
            std::printf("Masks skipped: download the %s model once in the window first\n",
                        settings.mask_model_id.c_str());
            job.prep.mask_enable = false;
        } else {
            job.prep.mask_prompt = m.text ? settings.mask.prompt : "";
            job.prep.mask_negative_prompt = m.text ? settings.mask.negative_prompt : "";
            job.prep.mask_feature_prompt = m.text ? settings.mask.feature_prompt : "";
            job.prep.mask_keep_subject = settings.mask.keep_subject;
            job.prep.mask_max_image_size = settings.mask.max_image_size;
            job.prep.mask_dilate_ratio = settings.mask.boundary_ratio();
            job.prep.mask_threshold = settings.mask.threshold;
            job.prep.mask_nms = settings.mask.nms;
            job.prep.mask_model_path = m.model;
            job.prep.mask_detector_path = m.detector;
            job.prep.mask_detector_threshold = settings.mask.box_threshold;
        }
    }
    job.prep.flip_found_masks = settings.use_found_masks && job.prep.flip_found_masks;
    if (redo) {
        job.request.redo_frames = job.request.redo_masks = job.request.redo_model = true;
        job.request.redo_geometry = job.request.redo_dense = true;
    }

    std::string uuid;
    if (!freeze_device(device, uuid, error)) return fail(error, 1);
    job.device_selector = uuid;
    job.prep.device = uuid;
    job.geometry.device_uuid = uuid;
    job.dense.config.device = uuid;
    job.geometry.enable = job.geometry.enable && backends().builtin_masking;
    app::set_crash_note("building dataset " + workspace);

    // ---- the workspace record and the run log, as launch_dataset_job() ----
    {
        fs::create_directories(fs::u8path(workspace), ec);
        const DatasetPlan plan = plan_dataset(plan_job(job),
                                              probe_workspace(workspace, job.prep.inputs),
                                              read_plan_record(workspace, job.prep),
                                              job.request);
        if (!job.request.redo_frames && !plan[Step::Frames].can_keep)
            return fail("the frames in this folder were made from other inputs or settings; "
                        "use --redo or another --out", 1);
        bool made[kNumSteps];
        for (int k = 0; k < kNumSteps; k++) made[k] = makes(plan[(Step)k].act);
        made[(int)Step::Frames] = made[(int)Step::Frames] || plan[Step::Frames].sort;
        DatasetSettings record = settings;
        record.sfm = job;
        write_record_settings(workspace, dataset_settings_json(record),
                              encode_record_inputs({sources, {}}), made);
    }
    std::ofstream prep_log;
    const std::string stamp = spirula::runlog::file_stamp();
    const fs::path log_file = spirula::runlog::open(prep_log, workspace, "prep", stamp);
    if (!log_file.empty()) {
        spirula::runlog::write_line(prep_log, "command = spirula prepare");
        for (const std::string& in : inputs) spirula::runlog::write_line(prep_log, "input = " + in);
        spirula::runlog::write_line(prep_log, "workspace = " + workspace);
        spirula::runlog::write_line(prep_log, "preset = " + preset);
        set_ss_env("UNREG_LOG", (log_file.parent_path() / ("unreg_" + stamp + ".log")).string());
    }

    std::printf("Dataset: %s\n", workspace.c_str());
    for (const PrepInput& in : sources)
        std::printf("Input: %s%s%s\n", in.path.c_str(), in.is_video ? " (video)" : "",
                    in.camera_model.empty() ? "" : (" lens " + in.camera_model).c_str());
    std::printf("Lens: %s   frames/s: %g   preset: %s\n", job.camera_model.c_str(),
                job.prep.video_fps, preset.c_str());
    std::fflush(stdout);

    // ---- the run ----
    SfmRunner runner;
    runner.start(job);
    int last_stage = -1;
    int last_pct = -1;
    bool cancelled = false;
    while (runner.state() == SfmRunner::State::Running || runner.state() == SfmRunner::State::Idle) {
        if (g_interrupted && !cancelled) {
            runner.cancel();
            cancelled = true;
        }
        for (const RunLine& l : runner.steps().drain()) {
            if (prep_log.is_open()) spirula::runlog::write_line(prep_log, l.text);
            if (!l.detail) std::printf("[%s] %s\n", kStageNames[(int)l.stage], l.text.c_str());
        }
        const int st = (int)runner.steps().current();
        const StageProgress sp = runner.steps().stage((Stage)st);
        if (st != last_stage) {
            std::printf("STAGE %s\n", kStageNames[st]);
            last_stage = st;
            last_pct = -1;
        }
        const int pct = sp.fraction >= 0.0f ? (int)(sp.fraction * 100.0f) : -1;
        if (pct != last_pct && pct >= 0) {
            if (sp.total > 0)
                std::printf("PROGRESS %s %d %lld/%lld\n", kStageNames[st], pct,
                            (long long)sp.done, (long long)sp.total);
            else
                std::printf("PROGRESS %s %d\n", kStageNames[st], pct);
            last_pct = pct;
        }
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    for (const RunLine& l : runner.steps().drain()) {
        if (prep_log.is_open()) spirula::runlog::write_line(prep_log, l.text);
        if (!l.detail) std::printf("[%s] %s\n", kStageNames[(int)l.stage], l.text.c_str());
    }

    const SfmRunner::State end = runner.state();
    if (end != SfmRunner::State::Done) {
        const std::string why = end == SfmRunner::State::Cancelled ? std::string("cancelled")
                                                                   : runner.error();
        if (prep_log.is_open()) spirula::runlog::write_line(prep_log, "failed: " + why);
        std::fprintf(stderr, "spirula prepare: %s\n", why.c_str());
        print_result(false, why, workspace, {}, {}, false, false, false, seconds());
        return end == SfmRunner::State::Cancelled ? 130 : 1;
    }
    runner.sweep_intermediates();
    if (prep_log.is_open()) spirula::runlog::write_line(prep_log, "==== Done ====");
    std::printf("Done in %.1f s: %s\n", seconds(), runner.dataset_dir().c_str());
    print_result(true, {}, runner.dataset_dir(), runner.image_dir(), runner.mask_dir(),
                 runner.mask_flipped(), runner.partial(), runner.not_metric(), seconds());
    return 0;
}

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const zgrad_dependency = b.dependency("zgrad", .{
        .target = target,
        .optimize = .ReleaseSafe,
    });
    const zgrad_module = zgrad_dependency.module("zgrad");

    const train = b.addExecutable(.{
        .name = "train",
        .root_source_file = .{ .path = "src/train.zig" },
        .target = target,
        .optimize = .ReleaseSafe,
    });

    train.root_module.addImport("zgrad", zgrad_module);

    b.installArtifact(train);

    const train_run_cmd = b.addRunArtifact(train);

    const train_run_step = b.step("train", "Train the autoencoder");
    train_run_step.dependOn(&train_run_cmd.step);

    const interact = b.addExecutable(.{
        .name = "interact",
        .root_source_file = b.path("src/interact.zig"),
        .target = target,
    });

    interact.root_module.addImport("zgrad", zgrad_module);
    interact.addCSourceFile(.{ .file = b.path("deps/src/raygui_implementation.c") });

    inline for (.{
        "raylib",
        "GL",
        "m",
        "pthread",
        "dl",
        "rt",
        "X11",
    }) |name|
        interact.linkSystemLibrary(name);

    interact.addLibraryPath(b.path("deps/lib"));
    interact.addIncludePath(b.path("deps/include"));
    interact.addRPath(b.path("deps/lib"));

    b.installArtifact(interact);

    const interact_run_cmd = b.addRunArtifact(interact);

    const interact_run_step = b.step("interact", "Interact with the decoder");
    interact_run_step.dependOn(&interact_run_cmd.step);
}

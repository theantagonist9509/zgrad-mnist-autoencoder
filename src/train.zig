const std = @import("std");

const IdxUbyte = @import("idxubyte.zig");
const zgrad = @import("zgrad");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const allocator = arena.allocator();
    defer arena.deinit();

    const training_images = (try IdxUbyte.initialize(allocator, "train-images-idx3-ubyte")).data;
    const testing_images = (try IdxUbyte.initialize(allocator, "t10k-images-idx3-ubyte")).data;

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    const file_name = "autoencoder";

    const T = struct {
        zgrad.AffineTransformation,
        zgrad.LeakyRelu(0.1),
        zgrad.AffineTransformation,
        zgrad.LeakyRelu(0.1),
        zgrad.AffineTransformation,
        zgrad.LeakyRelu(0.1),
        zgrad.AffineTransformation,
        zgrad.LeakyRelu(0.1),
        zgrad.AffineTransformation,
        zgrad.LeakyRelu(0.1),
        zgrad.AffineTransformation,
        zgrad.Sigmoid,
    };

    const model = zgrad.deserialize(zgrad.Sequence(T), allocator, file_name) catch |err| switch (err) {
        error.FileNotFound => try zgrad.initializeSequence(allocator, @as(T, .{ // using @as() due to tuple type equality issues :(
            try zgrad.AffineTransformation.initialize(zgrad.LeakyRelu(0.1), allocator, random, 28 * 28, 32),
            try zgrad.LeakyRelu(0.1).initializeOutputOnly(allocator, 32),
            try zgrad.AffineTransformation.initializeParametersAndOutputOnly(zgrad.LeakyRelu(0.1), allocator, random, 32, 16),
            try zgrad.LeakyRelu(0.1).initializeOutputOnly(allocator, 16),
            try zgrad.AffineTransformation.initializeParametersAndOutputOnly(zgrad.LeakyRelu(0.1), allocator, random, 16, 8),
            try zgrad.LeakyRelu(0.1).initializeOutputOnly(allocator, 8),
            try zgrad.AffineTransformation.initializeParametersAndOutputOnly(zgrad.LeakyRelu(0.1), allocator, random, 8, 16),
            try zgrad.LeakyRelu(0.1).initializeOutputOnly(allocator, 16),
            try zgrad.AffineTransformation.initializeParametersAndOutputOnly(zgrad.LeakyRelu(0.1), allocator, random, 16, 32),
            try zgrad.LeakyRelu(0.1).initializeOutputOnly(allocator, 32),
            try zgrad.AffineTransformation.initializeParametersAndOutputOnly(zgrad.Sigmoid, allocator, random, 32, 28 * 28),
            try zgrad.Sigmoid.initializeOutputOnly(allocator, 28 * 28),
        })),

        else => |remaining_error| return remaining_error,
    };

    var loss_operation = try zgrad.MeanSquaredError.initializeTargetAndOutput(allocator, 28 * 28);
    loss_operation.input = model.output;

    const optimizer = try zgrad.MomentumSgdOptimizer(0.002, 0.9).initialize(allocator, model.parameters);

    const epoch_count = 10;
    const training_images_count = 60_000;

    for (0..epoch_count) |epoch_index| {
        var accumulated_loss: f32 = 0;
        for (0..training_images_count) |image_index| {
            zgrad.zeroGradients(model.symbols);

            for (model.input.value.entries, training_images[28 * 28 * image_index ..][0 .. 28 * 28]) |*input_entry, image_entry|
                input_entry.* = @as(f32, @floatFromInt(image_entry)) / 255;

            @memcpy(loss_operation.target.value.entries, model.input.value.entries);

            model.operate();
            loss_operation.operate();

            accumulated_loss += loss_operation.output.value.entries[0];

            loss_operation.backpropagate();
            model.backpropagate();

            optimizer.updateParameters();
        }

        if (epoch_index == epoch_count - 1) {
            for (0..100) |image_index| {
                for (model.input.value.entries, testing_images[28 * 28 * image_index ..][0 .. 28 * 28]) |*input_entry, image_entry|
                    input_entry.* = @as(f32, @floatFromInt(image_entry)) / 255;

                model.operate();

                drawImage(model.input.value.entries);
                drawImage(model.output.value.entries);
            }
        }

        std.debug.print("[{}/{}] cost: {}\n", .{ epoch_index + 1, epoch_count, accumulated_loss / training_images_count });
    }

    try zgrad.serialize(allocator, model, file_name);
}

fn drawImage(data: []const f32) void {
    for (0..28) |i| {
        for (0..28) |j| {
            const character = getBrightnessCharacter(data[i * 28 + j]);
            std.debug.print("{c}{c}", .{ character, character });
        }

        std.debug.print("\n", .{});
    }
}

fn getBrightnessCharacter(brightness: f32) u8 {
    // https://paulbourke.net/dataformats/asciiart
    const characters = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

    for (characters, 1..) |_, i| {
        if (brightness <= @as(f32, @floatFromInt(i)) / characters.len)
            return characters[characters.len - i];
    }

    unreachable;
}

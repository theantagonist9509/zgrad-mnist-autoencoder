const std = @import("std");
const c = @cImport({
    @cInclude("raylib.h");
    @cInclude("raygui.h"); // See deps/src/raygui_implementation.c
});

const IdxUbyte = @import("idxubyte.zig");
const zgrad = @import("zgrad");

const pixel_size = 20;
const window_height = 28 * pixel_size;
const window_width: comptime_int = @intFromFloat(window_height * 16.0 / 9.0);

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const allocator = arena.allocator();
    defer arena.deinit();

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

    const autoencoder = try zgrad.deserialize(zgrad.Sequence(T), allocator, "autoencoder");
    const encoder = try zgrad.initializeSequence(allocator, zgrad.copyTupleSlice(autoencoder.operations, 0, @typeInfo(T).Struct.fields.len / 2)); // can't slice tuples yet :(
    const decoder = try zgrad.initializeSequence(allocator, zgrad.copyTupleSlice(autoencoder.operations, @typeInfo(T).Struct.fields.len / 2, @typeInfo(T).Struct.fields.len)); // can't slice tuples yet :(

    const images = (try IdxUbyte.initialize(allocator, "t10k-images-idx3-ubyte")).data;
    const image_count = images.len / (28 * 28);

    const mu = try allocator.alloc(f32, decoder.input.value.entries.len);
    const sigma = try allocator.alloc(f32, decoder.input.value.entries.len);

    for (0..image_count) |i| {
        for (encoder.input.value.entries, images[28 * 28 * i ..][0 .. 28 * 28]) |*input_entry, image_entry|
            input_entry.* = @as(f32, @floatFromInt(image_entry)) / 255;

        encoder.operate();

        for (mu, sigma, encoder.output.value.entries) |*mu_entry, *sigma_entry, output_entry| {
            mu_entry.* += output_entry;
            sigma_entry.* += output_entry * output_entry;
        }
    }

    for (mu, sigma) |*mu_entry, *sigma_entry| {
        mu_entry.* /= @floatFromInt(image_count);
        sigma_entry.* = @sqrt(sigma_entry.* / @as(f32, @floatFromInt(image_count)) - mu_entry.* * mu_entry.*);
    }

    c.InitWindow(window_width, window_height, "Interact");
    defer c.CloseWindow();
    c.SetTargetFPS(60);

    const slider_data = try allocator.alloc(f32, decoder.input.value.entries.len);
    @memcpy(slider_data, mu);

    while (!c.WindowShouldClose()) {
        c.BeginDrawing();
        defer c.EndDrawing();
        c.ClearBackground(c.GetColor(@bitCast(c.GuiGetStyle(c.DEFAULT, c.BACKGROUND_COLOR)))); // Need @bitCast() here as raylib function GetColor() accepts unsigned int but raygui function GuiGetStyle() returns int

        for (slider_data, mu, sigma, 0..) |*slider_entry, mu_entry, sigma_entry, i| {
            const slider_width = (window_width - window_height) / 2;
            const slider_height = window_height / @as(f32, @floatFromInt(1 + 2 * slider_data.len));
            _ = c.GuiSlider(.{
                .x = (window_width + window_height - slider_width) / 2,
                .y = @as(f32, @floatFromInt(1 + 2 * i)) * slider_height,
                .width = slider_width,
                .height = slider_height,
            }, "-2 Sigma", "+2 Sigma", slider_entry, mu_entry - 2 * sigma_entry, mu_entry + 2 * sigma_entry);
        }

        if (!std.mem.eql(f32, decoder.input.value.entries, slider_data)) {
            @memcpy(decoder.input.value.entries, slider_data);
            decoder.operate();
        }

        drawImage(decoder.output.value.entries);
    }
}

fn drawImage(data: []const f32) void {
    for (0..28) |i| {
        for (0..28) |j| {
            const pixel_brightness: u8 = @intFromFloat(data[28 * i + j] * 255);
            c.DrawRectangle(@intCast(pixel_size * j), @intCast(pixel_size * i), pixel_size, pixel_size, .{
                .r = pixel_brightness,
                .g = pixel_brightness,
                .b = pixel_brightness,
                .a = 255,
            });
        }
    }
}

const std = @import("std");
pub const c = @import("cuda_c");

pub fn checkCuda(status: c.cudaError_t) !void {
    if (status != c.cudaSuccess) return error.BackendFailure;
}

pub fn checkBlas(status: c.cublasStatus_t) !void {
    if (status != c.CUBLAS_STATUS_SUCCESS) return error.BackendFailure;
}

pub fn checkSolver(status: c.cusolverStatus_t) !void {
    if (status != c.CUSOLVER_STATUS_SUCCESS) return error.BackendFailure;
}

pub const Context = struct {
    stream: c.cudaStream_t,
    blas: c.cublasHandle_t,
    solver: c.cusolverDnHandle_t,

    pub fn init(device_ordinal: u32) !Context {
        const ordinal = std.math.cast(c_int, device_ordinal) orelse return error.InvalidDeviceOrdinal;
        var runtime_version: c_int = 0;
        try checkCuda(c.cudaRuntimeGetVersion(&runtime_version));
        if (runtime_version < 12090) return error.UnsupportedCudaVersion;
        var device_count: c_int = 0;
        try checkCuda(c.cudaGetDeviceCount(&device_count));
        if (ordinal < 0 or ordinal >= device_count) return error.InvalidDeviceOrdinal;
        try checkCuda(c.cudaSetDevice(ordinal));
        var stream: c.cudaStream_t = null;
        try checkCuda(c.cudaStreamCreateWithFlags(&stream, c.cudaStreamNonBlocking));
        errdefer _ = c.cudaStreamDestroy(stream);
        var blas: c.cublasHandle_t = null;
        try checkBlas(c.cublasCreate_v2(&blas));
        errdefer _ = c.cublasDestroy_v2(blas);
        try checkBlas(c.cublasSetStream_v2(blas, stream));
        var solver: c.cusolverDnHandle_t = null;
        try checkSolver(c.cusolverDnCreate(&solver));
        errdefer _ = c.cusolverDnDestroy(solver);
        try checkSolver(c.cusolverDnSetStream(solver, stream));
        return .{ .stream = stream, .blas = blas, .solver = solver };
    }

    pub fn synchronize(self: *Context) !void {
        try checkCuda(c.cudaStreamSynchronize(self.stream));
    }

    pub fn recordEvent(self: *Context) !Event {
        var event: c.cudaEvent_t = null;
        try checkCuda(c.cudaEventCreateWithFlags(&event, c.cudaEventDisableTiming));
        errdefer _ = c.cudaEventDestroy(event);
        try checkCuda(c.cudaEventRecord(event, self.stream));
        return .{ .handle = event };
    }

    pub fn deinit(self: *Context) void {
        _ = c.cusolverDnDestroy(self.solver);
        _ = c.cublasDestroy_v2(self.blas);
        _ = c.cudaStreamDestroy(self.stream);
    }
};

pub const Event = struct {
    handle: c.cudaEvent_t,

    pub fn query(self: *const Event) !bool {
        const status = c.cudaEventQuery(self.handle);
        if (status == c.cudaSuccess) return true;
        if (status == c.cudaErrorNotReady) return false;
        return error.BackendFailure;
    }

    pub fn wait(self: *Event) !void {
        try checkCuda(c.cudaEventSynchronize(self.handle));
    }
    pub fn deinit(self: *Event) void {
        _ = c.cudaEventDestroy(self.handle);
    }
};

pub fn Buffer(comptime T: type) type {
    return struct {
        values: [*]T,
        len: usize,
        context: *Context,
    };
}

pub fn allocate(comptime T: type, _: std.mem.Allocator, context: *Context, len: usize) !Buffer(T) {
    if (len == 0) return .{ .values = undefined, .len = 0, .context = context };
    var raw: ?*anyopaque = null;
    try checkCuda(c.cudaMalloc(&raw, try std.math.mul(usize, len, @sizeOf(T))));
    return .{ .values = @ptrCast(@alignCast(raw.?)), .len = len, .context = context };
}

pub fn release(comptime T: type, _: std.mem.Allocator, buffer: *Buffer(T)) void {
    if (buffer.len != 0) _ = c.cudaFree(buffer.values);
    buffer.len = 0;
}

pub fn copyFromHost(comptime T: type, buffer: *Buffer(T), offset: usize, stride: usize, source: []const T) !void {
    if (source.len == 0) return;
    try checkCuda(c.cudaMemcpy2DAsync(
        buffer.values + offset,
        stride * @sizeOf(T),
        source.ptr,
        @sizeOf(T),
        @sizeOf(T),
        source.len,
        c.cudaMemcpyHostToDevice,
        buffer.context.stream,
    ));
    try buffer.context.synchronize();
}

pub fn copyToHost(comptime T: type, buffer: *const Buffer(T), offset: usize, stride: usize, destination: []T) !void {
    if (destination.len == 0) return;
    try checkCuda(c.cudaMemcpy2DAsync(
        destination.ptr,
        @sizeOf(T),
        buffer.values + offset,
        stride * @sizeOf(T),
        @sizeOf(T),
        destination.len,
        c.cudaMemcpyDeviceToHost,
        buffer.context.stream,
    ));
    try buffer.context.synchronize();
}

pub fn asCInt(value: usize) !c_int {
    return std.math.cast(c_int, value) orelse error.DimensionTooLarge;
}

pub fn constPointer(comptime T: type, buffer: *const Buffer(T), offset: usize) [*]const T {
    return buffer.values + offset;
}

pub fn mutablePointer(comptime T: type, buffer: *Buffer(T), offset: usize) [*]T {
    return buffer.values + offset;
}

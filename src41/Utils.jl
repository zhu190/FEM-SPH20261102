module Utils

using Dates
using Printf
using ..Types: SimTimer

export find_file_path, time_to_string, update_timer!, reset_timer!

function find_file_path(filename::String, search_paths::Vector{String}=String[])
    if isfile(filename); return filename; end
    for path in search_paths; if isfile(joinpath(path, filename)); return joinpath(path, filename); end; end
    return nothing
end

function time_to_string(t::Real)
    ts = floor(Int, t)
    d = div(ts, 86400); r1 = ts % 86400
    h = div(r1, 3600); r2 = r1 % 3600
    m = div(r2, 60); s = r2 % 60
    return @sprintf("%02dd %02dh %02dm %02ds", d, h, m, s)
end

function reset_timer!(timer::SimTimer)
    timer.total = 0.0; timer.neighbor_search = 0.0; timer.adami_boundary = 0.0
    timer.interaction = 0.0; timer.integration = 0.0; timer.coupling = 0.0
    timer.coupling_extract = 0.0; timer.coupling_callback = 0.0
    timer.coupling_apply = 0.0; timer.coupling_upload = 0.0
    timer.saving = 0.0; timer.misc = 0.0
    now = time_ns(); timer.clock_initial = now; timer.clock_interval = now
end

function update_timer!(timer::SimTimer, field::Symbol, start_ns::UInt64)
    dt = (time_ns() - start_ns) / 1e9
    val = getfield(timer, field); setfield!(timer, field, val + dt)
end

end # module

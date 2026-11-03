module LoquatSPH

using CUDA
using Printf
using LightXML
import Adapt

include("Types.jl"); include("Utils.jl"); include("IO.jl")
include("Grid.jl"); include("Physics.jl"); include("TimeIntegration.jl")

using .Types, .Utils, .IO, .Grid, .Physics, .TimeIntegration

export RunSimulation

mutable struct LoquatSph
    par_host::Parameters
    grid::LoquatGrid
    pospres::CuVector{Float4}
    velrhop::CuVector{Float4}
    idep::CuVector{Float4}
    str::CuVector{Tensor2D}
    stn::CuVector{Tensor2D}
    pospres_ini::CuVector{Float4}
    velrhop_ini::CuVector{Float4}
    idep_ini::CuVector{Float4}
    str_ini::CuVector{Tensor2D}
    stn_ini::CuVector{Tensor2D}
    accad::CuVector{Float4}
    spin::CuVector{Float4}
    acestn::CuVector{Tensor2D}
    float4_cache::CuVector{Float4}
    tensor2d_cache::CuVector{Tensor2D}
    float_cache::CuVector{Float32} 
    mass_arr::CuVector{Float32}
    cpl_acc::CuVector{Float4}
    part_acc_ids::CuVector{Int32}
    part_acc_vals::CuVector{Float4}
    bridge_mask::CuVector{Float32}
    feedback_acc::CuVector{Float4}
    v_vis::CuVector{Float4}
    topo_normal::CuVector{Float4}
    topo_quantities::CuVector{Float4}
    topo_basal_velocity::CuVector{Float4}
    topo_entrainment::CuVector{Float4}
    topo_previous_depth::CuVector{Float32}
    timer::SimTimer

    function LoquatSph(par::Parameters)
        np = par.np
        npb = par.npb
        new(
            par, LoquatGrid(par),
            CUDA.zeros(Float4, np), CUDA.zeros(Float4, np), CUDA.zeros(Float4, np), CUDA.zeros(Tensor2D, np), CUDA.zeros(Tensor2D, np),
            CUDA.zeros(Float4, np), CUDA.zeros(Float4, np), CUDA.zeros(Float4, np), CUDA.zeros(Tensor2D, np), CUDA.zeros(Tensor2D, np),
            CUDA.zeros(Float4, np), CUDA.zeros(Float4, np), CUDA.zeros(Tensor2D, np),
            CUDA.zeros(Float4, np), 
            CUDA.zeros(Tensor2D, np),
            CUDA.zeros(Float32, np),
            CUDA.zeros(Float32, np),
            CUDA.zeros(Float4, np),
            CUDA.zeros(Int32, 0),
            CUDA.zeros(Float4, 0),
            CUDA.zeros(Float32, np),
            CUDA.zeros(Float4, np),
            CUDA.zeros(Float4, npb),
            CUDA.zeros(Float4, np),
            CUDA.zeros(Float4, np),
            CUDA.zeros(Float4, np),
            CUDA.zeros(Float4, np),
            CUDA.zeros(Float32, np),
            SimTimer()
        )
    end
end

struct AsyncSaveSlot
    pos::Vector{Float4}
    vel::Vector{Float4}
    idep::Vector{Float4}
    str::Vector{Tensor2D}
    stn::Vector{Tensor2D}
    mass::Vector{Float32}
    acc::Vector{Float4}
    topo_normal::Vector{Float4}
    topo_quantities::Vector{Float4}
    topo_basal_velocity::Vector{Float4}
    topo_entrainment::Vector{Float4}
end

struct AsyncSaveJob
    slot_idx::Int
    par::Parameters
end

@inline function _make_async_save_slot(np::Integer)
    return AsyncSaveSlot(
        Vector{Float4}(undef, np),
        Vector{Float4}(undef, np),
        Vector{Float4}(undef, np),
        Vector{Tensor2D}(undef, np),
        Vector{Tensor2D}(undef, np),
        Vector{Float32}(undef, np),
        Vector{Float4}(undef, np),
        Vector{Float4}(undef, np),
        Vector{Float4}(undef, np),
        Vector{Float4}(undef, np),
        Vector{Float4}(undef, np)
    )
end

function sort_arrays_primary!(sph::LoquatSph)
    grid = sph.grid
    par = sph.par_host
    if par.static_boundary_grid != 0 && par.npb > 0
        if !grid.boundary_arrays_sorted
            sort_boundary_data!(grid, sph.pospres, sph.float4_cache, par)
            sort_boundary_data!(grid, sph.velrhop, sph.float4_cache, par)
            sort_boundary_data!(grid, sph.idep, sph.float4_cache, par)
            sort_boundary_data!(grid, sph.str, sph.tensor2d_cache, par)
            sort_boundary_data!(grid, sph.stn, sph.tensor2d_cache, par)
            sort_boundary_data!(grid, sph.mass_arr, sph.float_cache, par)
            sort_boundary_data!(grid, sph.cpl_acc, sph.float4_cache, par)
            sort_boundary_data!(grid, sph.bridge_mask, sph.float_cache, par)
            sort_boundary_data!(grid, sph.feedback_acc, sph.float4_cache, par)
            sort_boundary_data!(grid, sph.topo_entrainment, sph.float4_cache, par)
            sort_boundary_data!(grid, sph.topo_previous_depth, sph.float_cache, par)
            grid.boundary_arrays_sorted = true
        end

        sort_material_data!(grid, sph.pospres, sph.float4_cache, par)
        sort_material_data!(grid, sph.velrhop, sph.float4_cache, par)
        sort_material_data!(grid, sph.idep, sph.float4_cache, par)
        sort_material_data!(grid, sph.str, sph.tensor2d_cache, par)
        sort_material_data!(grid, sph.stn, sph.tensor2d_cache, par)
        sort_material_data!(grid, sph.mass_arr, sph.float_cache, par)
        sort_material_data!(grid, sph.cpl_acc, sph.float4_cache, par)
        sort_material_data!(grid, sph.bridge_mask, sph.float_cache, par)
        sort_material_data!(grid, sph.feedback_acc, sph.float4_cache, par)
        return nothing
    end

    sort_data!(grid, sph.pospres, sph.float4_cache)
    sort_data!(grid, sph.velrhop, sph.float4_cache)
    sort_data!(grid, sph.idep, sph.float4_cache)
    sort_data!(grid, sph.str, sph.tensor2d_cache)
    sort_data!(grid, sph.stn, sph.tensor2d_cache)
    sort_data!(grid, sph.mass_arr, sph.float_cache)
    sort_data!(grid, sph.cpl_acc, sph.float4_cache)
    sort_data!(grid, sph.bridge_mask, sph.float_cache)
    sort_data!(grid, sph.feedback_acc, sph.float4_cache)
    sort_data!(grid, sph.topo_entrainment, sph.float4_cache)
    sort_data!(grid, sph.topo_previous_depth, sph.float_cache)
end

function sort_arrays_all!(sph::LoquatSph)
    sort_arrays_primary!(sph)
    grid = sph.grid
    par = sph.par_host
    if par.static_boundary_grid != 0 && par.npb > 0
        sort_material_data!(grid, sph.pospres_ini, sph.float4_cache, par)
        sort_material_data!(grid, sph.velrhop_ini, sph.float4_cache, par)
        sort_material_data!(grid, sph.idep_ini, sph.float4_cache, par)
        sort_material_data!(grid, sph.str_ini, sph.tensor2d_cache, par)
        sort_material_data!(grid, sph.stn_ini, sph.tensor2d_cache, par)
        return nothing
    end

    sort_data!(grid, sph.pospres_ini, sph.float4_cache)
    sort_data!(grid, sph.velrhop_ini, sph.float4_cache)
    sort_data!(grid, sph.idep_ini, sph.float4_cache)
    sort_data!(grid, sph.str_ini, sph.tensor2d_cache)
    sort_data!(grid, sph.stn_ini, sph.tensor2d_cache)
end

@inline _zero4() = Float4(0f0, 0f0, 0f0, 0f0)

@inline function resolve_part_acceleration_host(
    uid::Float32,
    part_acc_ids::Vector{Int32},
    part_acc_vals::Vector{Float4},
    par::Parameters
)
    uid_i = Int32(round(uid))
    @inbounds for k in eachindex(part_acc_ids)
        if part_acc_ids[k] == uid_i
            a = part_acc_vals[k]
            return a[1], a[2], a[3]
        end
    end
    return par.acc_x, par.acc_y, par.acc_z
end

@inline function resolve_part_acceleration_device(
    uid::Float32,
    part_acc_ids,
    part_acc_vals,
    part_acc_count::Int32,
    acc_x::Float32,
    acc_y::Float32,
    acc_z::Float32
)
    uid_i = Int32(round(uid))
    @inbounds for k in Int32(1):part_acc_count
        if part_acc_ids[k] == uid_i
            a = part_acc_vals[k]
            return a[1], a[2], a[3]
        end
    end
    return acc_x, acc_y, acc_z
end

function kernel_pack_bridge_view!(
    packed,
    pospres,
    velrhop,
    idep,
    feedback_acc,
    accad,
    mass_arr,
    idxs,
    part_acc_ids,
    part_acc_vals,
    part_acc_count::Int32,
    acc_x::Float32,
    acc_y::Float32,
    acc_z::Float32,
    nb::Int32
)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= nb
        idx = idxs[k]
        if idx > 0
            p = pospres[idx]
            v = velrhop[idx]
            idv = idep[idx]
            fb = feedback_acc[idx]
            a = accad[idx]
            m = mass_arr[idx]
            base_ax, base_ay, base_az = resolve_part_acceleration_device(
                idv[1], part_acc_ids, part_acc_vals, part_acc_count, acc_x, acc_y, acc_z
            )

            packed[k, 1] = p[1];  packed[k, 2] = p[2];  packed[k, 3] = p[3];  packed[k, 4] = p[4]
            packed[k, 5] = v[1];  packed[k, 6] = v[2];  packed[k, 7] = v[3];  packed[k, 8] = v[4]
            packed[k, 9] = idv[1]; packed[k, 10] = idv[2]; packed[k, 11] = idv[3]; packed[k, 12] = idv[4]
            packed[k, 13] = fb[1]; packed[k, 14] = fb[2]; packed[k, 15] = fb[3]; packed[k, 16] = fb[4]
            packed[k, 17] = a[1];  packed[k, 18] = a[2];  packed[k, 19] = a[3];  packed[k, 20] = a[4]
            packed[k, 21] = m
            packed[k, 22] = m * fb[1]
            packed[k, 23] = m * fb[2]
            packed[k, 24] = m * fb[3]
            packed[k, 25] = m * (a[1] - base_ax)
            packed[k, 26] = m * (a[2] - base_ay)
            packed[k, 27] = m * (a[3] - base_az)
        else
            @inbounds for c in 1:27
                packed[k, c] = 0f0
            end
        end
    end
    return nothing
end

function _apply_coupling_callback_result!(
    host_cpl_acc::Vector{Float4},
    host_acc_target::Vector{Float4},
    host_vel_target::Vector{Float4},
    host_pos_target::Vector{Float4},
    host_target_mask::Vector{Float32},
    host_stress_target::Vector{Tensor2D},
    host_stress_mask::Vector{Float32},
    callback_result,
    host_idep::Vector{Float4};
    id_to_index_override::Union{Nothing, Dict{Int32, Int}}=nothing
)
    if callback_result === nothing
        return
    elseif callback_result isa AbstractMatrix
        mat = callback_result
        size(mat, 1) == length(host_cpl_acc) || error("Coupling callback matrix rows must equal particle count.")
        size(mat, 2) >= 3 || error("Coupling callback matrix must have at least 3 columns.")
        @inbounds for i in eachindex(host_cpl_acc)
            host_cpl_acc[i] = Float4(Float32(mat[i, 1]), Float32(mat[i, 2]), Float32(mat[i, 3]), 0f0)
        end
        return
    elseif callback_result isa AbstractVector{Float4}
        length(callback_result) == length(host_cpl_acc) || error("Coupling callback vector length must equal particle count.")
        copyto!(host_cpl_acc, callback_result)
        return
    elseif callback_result isa NamedTuple
        haskey(callback_result, :original_ids) || error("Coupling callback NamedTuple requires :original_ids.")
        ids = callback_result.original_ids
        n = length(ids)

        has_forces = haskey(callback_result, :forces)
        has_acc_target = haskey(callback_result, :acceleration_target)
        has_vel_target = haskey(callback_result, :velocity_target)
        has_pos_target = haskey(callback_result, :position_target)
        has_stress_target = haskey(callback_result, :stress_target)
        (has_forces || has_acc_target || has_vel_target || has_pos_target || has_stress_target) ||
            error("Coupling callback NamedTuple requires at least one of :forces, :acceleration_target, :velocity_target, :position_target, or :stress_target.")

        f = has_forces ? callback_result.forces : nothing
        a_target = has_acc_target ? callback_result.acceleration_target : nothing
        v_target = has_vel_target ? callback_result.velocity_target : nothing
        p_target = has_pos_target ? callback_result.position_target : nothing
        s_target = has_stress_target ? callback_result.stress_target : nothing

        if has_forces
            size(f, 1) == n || error("Coupling callback forces row count must match original_ids.")
            size(f, 2) >= 3 || error("Coupling callback forces must have at least 3 columns.")
        end
        if has_acc_target
            size(a_target, 1) == n || error("Coupling callback acceleration_target row count must match original_ids.")
            size(a_target, 2) >= 3 || error("Coupling callback acceleration_target must have at least 3 columns.")
        end
        if has_vel_target
            size(v_target, 1) == n || error("Coupling callback velocity_target row count must match original_ids.")
            size(v_target, 2) >= 3 || error("Coupling callback velocity_target must have at least 3 columns.")
        end
        if has_pos_target
            size(p_target, 1) == n || error("Coupling callback position_target row count must match original_ids.")
            size(p_target, 2) >= 3 || error("Coupling callback position_target must have at least 3 columns.")
        end
        if has_stress_target
            size(s_target, 1) == n || error("Coupling callback stress_target row count must match original_ids.")
            size(s_target, 2) >= 6 || error("Coupling callback stress_target must have at least 6 columns.")
        end

        id_to_index = if id_to_index_override === nothing
            tmp = Dict{Int32, Int}()
            @inbounds for i in eachindex(host_idep)
                oid = Int32(round(host_idep[i][3]))
                if oid > 0
                    tmp[oid] = i
                end
            end
            tmp
        else
            id_to_index_override
        end
        @inbounds for k in 1:n
            oid = Int32(ids[k])
            idx = get(id_to_index, oid, 0)
            if idx > 0
                if has_forces
                    host_cpl_acc[idx] = Float4(Float32(f[k, 1]), Float32(f[k, 2]), Float32(f[k, 3]), 0f0)
                end
                if has_acc_target
                    host_acc_target[idx] = Float4(Float32(a_target[k, 1]), Float32(a_target[k, 2]), Float32(a_target[k, 3]), 1f0)
                    host_target_mask[idx] = 1f0
                end
                if has_vel_target
                    host_vel_target[idx] = Float4(Float32(v_target[k, 1]), Float32(v_target[k, 2]), Float32(v_target[k, 3]), 1f0)
                    host_target_mask[idx] = 1f0
                end
                if has_pos_target
                    host_pos_target[idx] = Float4(Float32(p_target[k, 1]), Float32(p_target[k, 2]), Float32(p_target[k, 3]), 1f0)
                    host_target_mask[idx] = 1f0
                end
                if has_stress_target
                    host_stress_target[idx] = Tensor2D(
                        s_target[k, 1], s_target[k, 2], s_target[k, 3],
                        s_target[k, 4], s_target[k, 5], s_target[k, 6]
                    )
                    host_stress_mask[idx] = 1f0
                end
            end
        end
        return
    else
        error("Unsupported coupling callback return type: $(typeof(callback_result)).")
    end
end

function RunSimulation(
    project_name::String,
    device_id::Int64;
    coupling_callback::Union{Nothing, Function}=nothing,
    coupling_every::Int=1,
    bridge_original_ids::AbstractVector{<:Integer}=Int32[]
)
    device!(device_id)
    println("**********************************")
    println("** LOQUAT Solver for single GPU **")
    println("**********************************\n")
    println("Simulation running on GPU $device_id\n")
    
    par_host, h_rho0, h_K, h_G, h_c_model, h_p1, h_p2, h_p3,
               h_d_p, h_rho_s, h_mu_s, h_mu_2, h_I_0, h_mu_I_lambda,
               part_acc_ids_host, part_acc_vals_host = IO.load_parameters(project_name)
    par_host = update_derived_params(par_host)
    
    println("Loading particle data ...")
    println("Finish loading particle data ...")
    println("ncellx = $(par_host.cell_num_x), ncelly = $(par_host.cell_num_y), ncellz = $(par_host.cell_num_z), ncell = $(par_host.ncell)\n")
    if par_host.static_boundary_grid != 0 && par_host.npb > 0
        println("[NeighborGrid] Static boundary grid enabled.")
        println("[NeighborGrid] Boundary sort once = $(par_host.npb), material sort per rebuild = $(par_host.npm), legacy full sort = $(par_host.np)")
    else
        println("[NeighborGrid] Legacy full particle sorting enabled: $(par_host.np) particles per rebuild.")
    end
    
    if !isdir(project_name)
        println("Create results folder: ", abspath(project_name))
        mkpath(project_name)
    end

    sph = LoquatSph(par_host)
    sph.part_acc_ids = CuArray(part_acc_ids_host)
    sph.part_acc_vals = CuArray(part_acc_vals_host)
    if !isempty(part_acc_ids_host)
        println("[Accel] Local part acceleration overrides loaded: $(length(part_acc_ids_host))")
    end
    
    pos, vel, idep, str, stn = IO.load_particles(project_name, par_host)
    
    host_mass = zeros(Float32, par_host.np)
    host_bridge_mask = zeros(Float32, par_host.np)
    
    dr3 = par_host.dr^3
    for i in 1:par_host.np
        uid = idep[i][1]
        idx = (uid < 100) ? 1 : (Int(div(uid, 100)) + 1)
        if idx > length(h_rho0); idx = 1; end
        host_mass[i] = h_rho0[idx] * dr3
        if uid > 100f0
            host_bridge_mask[i] = 1f0
        end
    end
    sph.mass_arr = CuArray(host_mass)
    sph.bridge_mask = CuArray(host_bridge_mask)
    println("[Coupled] Bridge mask rule: part ID > 100")
    println("[Coupled] Bridge mask particles = $(count(>(0.5f0), host_bridge_mask))")

    d_rho0 = CuArray(h_rho0)
    d_K    = CuArray(h_K)
    d_G    = CuArray(h_G)
    d_c_model = CuArray(h_c_model)
    d_p1 = CuArray(h_p1)
    d_p2 = CuArray(h_p2)
    d_p3 = CuArray(h_p3)
    d_d_p = CuArray(h_d_p)
    d_rho_s = CuArray(h_rho_s)
    d_mu_s = CuArray(h_mu_s)
    d_mu_2 = CuArray(h_mu_2)
    d_I_0 = CuArray(h_I_0)
    d_mu_I_lambda = CuArray(h_mu_I_lambda)
    mat_props = MaterialProperties(
        d_rho0, d_K, d_G, d_c_model,
        d_p1, d_p2, d_p3,
        d_d_p, d_rho_s, d_mu_s, d_mu_2, d_I_0, d_mu_I_lambda
    )

    centroid_x, centroid_y, centroid_z = 0.0, 0.0, 0.0
    centroid_count = 0
    for i in (Int(par_host.npb) + 1):Int(par_host.np)
        uid = idep[i][1]
        if uid == -1f0
            continue
        end
        mat_idx = (uid < 100f0) ? 1 : (Int(div(uid, 100f0)) + 1)
        if 1 <= mat_idx <= length(h_c_model) && h_c_model[mat_idx] == Int32(8)
            centroid_x += Float64(pos[i][1])
            centroid_y += Float64(pos[i][2])
            centroid_z += Float64(pos[i][3])
            centroid_count += 1
        end
    end
    model8_centroid = if centroid_count > 0
        inv_count = 1.0 / centroid_count
        Float4(
            Float32(centroid_x * inv_count),
            Float32(centroid_y * inv_count),
            Float32(centroid_z * inv_count),
            Float32(centroid_count)
        )
    else
        Float4(0f0, 0f0, 0f0, 0f0)
    end

    use_model8_topography = any(==(Int32(8)), h_c_model)
    if use_model8_topography
        println("[Model8] PCA/vectorized topography-linked quantity extraction enabled.")
        println("[Model8] Normal orientation = local material side -> anti-gravity -> material-centroid fallback")
        println("[Model8] Normal orientation radius = $(par_host.normal_orientation_radius)")
        println("[Model8] FlowDepth method = kernel-weighted column volume / bed area")
        println("[Model8] FlowDepth coverage correction = basal layer; wet tangent threshold = $(1.5f0 * par_host.dr)")
        println("[Model8] Surface filter = PCA trace ratio <= 0.29 or one-sided centroid offset >= $(0.08f0 * par_host.h)")
        println("[Model8] Bed/material alignment >= 0.65, basal gap = [$(0.25f0 * par_host.dr), $(max(2f0 * par_host.h, 3f0 * par_host.dr))], coverage >= 0.10")
        println("[Model8] FlowDepth upper bound = connected material-column particle span")
        println("[Model8] Depth footprint radius = $(par_host.depth_footprint_radius), search max = $(par_host.depth_search_max)")
        if par_host.entrainment_enabled != 0
            println("[Model8] Progressive entrainment enabled: E and cumulative erosion depth d_sc")
            println("[Model8] B_d = $(par_host.entrainment_pore_pressure), flow friction = $(par_host.entrainment_flow_friction_angle * 180f0 / PI_F) deg, bed friction = $(par_host.entrainment_bed_friction_angle * 180f0 / PI_F) deg")
            println("[Model8] Erosion depth cap = $(par_host.erosion_depth_max), basal speed threshold = $(par_host.entrainment_speed_min), rate cap = $(par_host.entrainment_rate_max)")
        else
            println("[Model8] Progressive entrainment disabled (set entrainment-enabled=1 to compute E and d_sc).")
        end
    end

    copyto!(sph.pospres, pos); copyto!(sph.velrhop, vel); copyto!(sph.idep, idep)
    copyto!(sph.str, str); copyto!(sph.stn, stn)

    t_std = 256
    b_npb = cld(par_host.npb, t_std); b_npm = cld(par_host.npm, t_std)

    function run_topography_extraction!(initialize_entrainment::Bool = false)
        if !use_model8_topography || par_host.npb <= 0
            return nothing
        end
        @cuda threads=t_std blocks=b_npb kernel_extract_model8_topography_quantities(
            sph.topo_normal,
            sph.topo_quantities,
            sph.topo_basal_velocity,
            sph.pospres,
            sph.velrhop,
            sph.idep,
            sph.grid.cell_beg,
            sph.grid.cell_end,
            sph.mass_arr,
            mat_props,
            model8_centroid,
            sph.grid.active_boundary_indices,
            par_host.npb,
            Int32(0),
            par_host
        )
        @cuda threads=t_std blocks=b_npb kernel_update_model8_entrainment(
            sph.topo_entrainment,
            sph.topo_previous_depth,
            sph.topo_quantities,
            sph.topo_basal_velocity,
            sph.idep,
            initialize_entrainment ? Int32(1) : Int32(0),
            par_host
        )
        CUDA.synchronize()
        return nothing
    end
    
    neighbor_search!(sph.grid, par_host, sph.pospres, sph.idep)
    sort_arrays_primary!(sph)
    update_active_boundary!(sph.grid, par_host)
    run_topography_extraction!(true)
    
    copyto!(pos, sph.pospres); copyto!(vel, sph.velrhop); copyto!(idep, sph.idep)
    copyto!(str, sph.str); copyto!(stn, sph.stn)
    
    copyto!(host_mass, sph.mass_arr)
    host_topo_normal = Array(sph.topo_normal)
    host_topo_quantities = Array(sph.topo_quantities)
    host_topo_basal_velocity = Array(sph.topo_basal_velocity)
    host_topo_entrainment = Array(sph.topo_entrainment)
    
    pos_ref = copy(pos)
    idep_ref = copy(idep)

    host_acc_used = Array(sph.accad)

    IO.save_xml(project_name, par_host; part_acc_ids=part_acc_ids_host, part_acc_vals=part_acc_vals_host)
    IO.save_particles(
        project_name,
        par_host,
        pos,
        vel,
        idep,
        str,
        stn,
        host_mass;
        pos_ref = pos_ref,
        idep_ref = idep_ref,
        acc = host_acc_used,
        topo_normal = use_model8_topography ? host_topo_normal : nothing,
        topo_quantities = use_model8_topography ? host_topo_quantities : nothing,
        topo_basal_velocity = use_model8_topography ? host_topo_basal_velocity : nothing,
        topo_entrainment = use_model8_topography ? host_topo_entrainment : nothing
    )

    async_save_slots = [_make_async_save_slot(Int(par_host.np)) for _ in 1:2]
    async_save_free = Channel{Int}(length(async_save_slots))
    async_save_jobs = Channel{Union{AsyncSaveJob, Nothing}}(length(async_save_slots))
    for idx in eachindex(async_save_slots)
        put!(async_save_free, idx)
    end
    async_save_worker = Base.Threads.@spawn begin
        while true
            job = take!(async_save_jobs)
            job === nothing && break
            slot = async_save_slots[job.slot_idx]
            try
                IO.save_xml(project_name, job.par; part_acc_ids=part_acc_ids_host, part_acc_vals=part_acc_vals_host)
                IO.save_particles(
                    project_name,
                    job.par,
                    slot.pos,
                    slot.vel,
                    slot.idep,
                    slot.str,
                    slot.stn,
                    slot.mass;
                    pos_ref = pos_ref,
                    idep_ref = idep_ref,
                    acc = slot.acc,
                    topo_normal = use_model8_topography ? slot.topo_normal : nothing,
                    topo_quantities = use_model8_topography ? slot.topo_quantities : nothing,
                    topo_basal_velocity = use_model8_topography ? slot.topo_basal_velocity : nothing,
                    topo_entrainment = use_model8_topography ? slot.topo_entrainment : nothing
                )
            finally
                put!(async_save_free, job.slot_idx)
            end
        end
        return nothing
    end

    function enqueue_async_save!(par_snapshot::Parameters)
        if istaskdone(async_save_worker)
            fetch(async_save_worker)
        end
        slot_idx = take!(async_save_free)
        slot = async_save_slots[slot_idx]
        copyto!(slot.pos, sph.pospres)
        copyto!(slot.vel, sph.velrhop)
        copyto!(slot.idep, sph.idep)
        copyto!(slot.str, sph.str)
        copyto!(slot.stn, sph.stn)
        copyto!(slot.mass, sph.mass_arr)
        copyto!(slot.acc, host_acc_used)
        copyto!(slot.topo_normal, sph.topo_normal)
        copyto!(slot.topo_quantities, sph.topo_quantities)
        copyto!(slot.topo_basal_velocity, sph.topo_basal_velocity)
        copyto!(slot.topo_entrainment, sph.topo_entrainment)
        put!(async_save_jobs, AsyncSaveJob(slot_idx, par_snapshot))
        return nothing
    end
    
    sim_time = Float64(par_host.time)
    step = 0
    output_time = 0.0
    step_interval = 0
    
    k_int = @cuda launch=false kernel_particle_interaction(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.feedback_acc, sph.spin, sph.acestn, sph.grid.cell_beg, sph.grid.cell_end, sph.mass_arr, sph.bridge_mask, mat_props, sph.v_vis, sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host)
    cfg_int = launch_configuration(k_int.fun)
    t_int = min(256, cfg_int.threads); b_int = cld(par_host.npm, t_int)
    
    k_integ = @cuda launch=false kernel_predictor(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.pospres_ini, sph.velrhop_ini, sph.idep_ini, sph.str_ini, sph.stn_ini, sph.accad, sph.spin, sph.acestn, mat_props, par_host)
    cfg_integ = launch_configuration(k_integ.fun)
    t_integ = min(256, cfg_integ.threads); b_integ = cld(par_host.npm, t_integ)

    k_lf_kin = @cuda launch=false kernel_leapfrog_kinematics(sph.pospres, sph.velrhop, sph.accad, sph.bridge_mask, par_host)
    cfg_lf_kin = launch_configuration(k_lf_kin.fun)
    t_lf_kin = min(256, cfg_lf_kin.threads); b_lf_kin = cld(par_host.npm, t_lf_kin)

    k_lf_stress = @cuda launch=false kernel_leapfrog_stress(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.spin, sph.acestn, mat_props, par_host)
    cfg_lf_stress = launch_configuration(k_lf_stress.fun)
    t_lf_stress = min(256, cfg_lf_stress.threads); b_lf_stress = cld(par_host.npm, t_lf_stress)

    k_lf_kick = @cuda launch=false kernel_leapfrog_kickstart(sph.velrhop, sph.accad, par_host)
    cfg_lf_kick = launch_configuration(k_lf_kick.fun)
    t_lf_kick = min(256, cfg_lf_kick.threads); b_lf_kick = cld(par_host.npm, t_lf_kick)

    host_cpl_acc = [_zero4() for _ in 1:par_host.np]
    host_acc_target = [_zero4() for _ in 1:par_host.np]
    host_vel_target = [_zero4() for _ in 1:par_host.np]
    host_pos_target = [_zero4() for _ in 1:par_host.np]
    host_target_mask = zeros(Float32, par_host.np)
    host_stress_target = [zero(Tensor2D) for _ in 1:par_host.np]
    host_stress_mask = zeros(Float32, par_host.np)
    host_idep_full = Vector{Float4}(undef, par_host.np)
    host_acc_full = Vector{Float4}(undef, par_host.np)

    bridge_ids = Int32.(collect(bridge_original_ids))
    use_bridge_subset_extract = !isempty(bridge_ids)
    bridge_nb = length(bridge_ids)
    bridge_id_to_slot = Dict{Int32, Int}(oid => i for (i, oid) in pairs(bridge_ids))
    host_bridge_idx = zeros(Int32, bridge_nb)
    bridge_id_to_global_idx = Dict{Int32, Int}()

    host_pos_cb = use_bridge_subset_extract ? Vector{Float4}(undef, bridge_nb) : Vector{Float4}(undef, par_host.np)
    host_vel_cb = use_bridge_subset_extract ? Vector{Float4}(undef, bridge_nb) : Vector{Float4}(undef, par_host.np)
    host_idep_cb = use_bridge_subset_extract ? Vector{Float4}(undef, bridge_nb) : Vector{Float4}(undef, par_host.np)
    host_feedback_acc_cb = use_bridge_subset_extract ? Vector{Float4}(undef, bridge_nb) : Vector{Float4}(undef, par_host.np)
    host_acc_cb = use_bridge_subset_extract ? Vector{Float4}(undef, bridge_nb) : Vector{Float4}(undef, par_host.np)
    host_mass_cb = use_bridge_subset_extract ? Vector{Float32}(undef, bridge_nb) : Vector{Float32}(undef, par_host.np)
    host_free_force_cb = use_bridge_subset_extract ? zeros(Float32, bridge_nb, 3) : zeros(Float32, par_host.np, 3)
    host_free_force_total_cb = use_bridge_subset_extract ? zeros(Float32, bridge_nb, 3) : zeros(Float32, par_host.np, 3)

    d_bridge_idx = use_bridge_subset_extract ? CuArray(host_bridge_idx) : nothing
    d_bridge_packed = use_bridge_subset_extract ? CUDA.zeros(Float32, bridge_nb, 27) : nothing
    host_bridge_packed = use_bridge_subset_extract ? zeros(Float32, bridge_nb, 27) : nothing
    b_bridge = use_bridge_subset_extract ? cld(bridge_nb, t_std) : 0

    function run_adami_boundary!()
        if par_host.npb <= 0
            return nothing
        end

        t0 = time_ns()
        active_count = sph.grid.active_boundary_count_host
        if par_host.active_boundary_adami != 0 && par_host.static_boundary_grid != 0
            if active_count > 0
                b_active = cld(Int(active_count), t_std)
                @cuda threads=t_std blocks=b_active kernel_adami_boundary_active(
                    sph.pospres, sph.velrhop, sph.idep, sph.str, sph.v_vis,
                    sph.grid.cell_beg, sph.grid.cell_end, sph.grid.active_boundary_indices,
                    active_count, mat_props, sph.part_acc_ids, sph.part_acc_vals,
                    Int32(length(part_acc_ids_host)), par_host
                )
            end
        else
            @cuda threads=t_std blocks=b_npb kernel_adami_boundary(
                sph.pospres, sph.velrhop, sph.idep, sph.str, sph.v_vis,
                sph.grid.cell_beg, sph.grid.cell_end, mat_props,
                sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host
            )
        end
        CUDA.synchronize()
        update_timer!(sph.timer, :adami_boundary, t0)
        return nothing
    end

    function apply_pending_bridge_stress_targets!(; clear_after::Bool)
        any_stress = false
        @inbounds for i in eachindex(host_stress_mask)
            if host_stress_mask[i] > 0.5f0
                any_stress = true
                break
            end
        end
        any_stress || return nothing

        host_str_now = Array(sph.str)
        @inbounds for i in eachindex(host_stress_mask)
            if host_stress_mask[i] > 0.5f0
                host_str_now[i] = host_stress_target[i]
            end
        end
        copyto!(sph.str, host_str_now)

        if clear_after
            fill!(host_stress_target, zero(Tensor2D))
            fill!(host_stress_mask, 0f0)
        end
        return nothing
    end

    function snapshot_acceleration_used!()
        copyto!(host_acc_used, sph.accad)
        return nothing
    end

    function apply_pending_bridge_targets!()
        any_target = false
        @inbounds for i in eachindex(host_target_mask)
            if host_target_mask[i] > 0.5f0
                any_target = true
                break
            end
        end
        if !any_target
            @inbounds for i in eachindex(host_stress_mask)
                if host_stress_mask[i] > 0.5f0
                    any_target = true
                    break
                end
            end
        end
        any_target || return nothing

        host_pos_now = Array(sph.pospres)
        host_vel_now = Array(sph.velrhop)
        updated_pos = false
        updated_vel = false

        @inbounds for i in eachindex(host_target_mask)
            if host_target_mask[i] > 0.5f0
                vt = host_vel_target[i]
                if vt[4] > 0.5f0
                    v = host_vel_now[i]
                    host_vel_now[i] = Float4(vt[1], vt[2], vt[3], v[4])
                    updated_vel = true
                end

                pt = host_pos_target[i]
                if pt[4] > 0.5f0
                    p = host_pos_now[i]
                    host_pos_now[i] = Float4(pt[1], pt[2], pt[3], p[4])
                    updated_pos = true
                end
            end
        end

        if updated_vel
            copyto!(sph.velrhop, host_vel_now)
        end
        if updated_pos
            copyto!(sph.pospres, host_pos_now)
        end

        apply_pending_bridge_stress_targets!(; clear_after = false)

        fill!(host_vel_target, _zero4())
        fill!(host_pos_target, _zero4())
        fill!(host_target_mask, 0f0)
        return nothing
    end

    function run_coupling_callback!(cb_step::Int, cb_time::Float64, cb_phase::Symbol = :single_step)
        if coupling_callback === nothing || coupling_every <= 0 || (cb_step % coupling_every != 0)
            return nothing
        end
        t_cb_total = time_ns()

        t0 = time_ns()
        copyto!(host_idep_full, sph.idep)
        local_apply_map = nothing

        if use_bridge_subset_extract
            fill!(host_bridge_idx, 0)
            @inbounds for i in eachindex(host_idep_full)
                oid = Int32(round(host_idep_full[i][3]))
                slot = get(bridge_id_to_slot, oid, 0)
                if slot > 0
                    host_bridge_idx[slot] = Int32(i)
                end
            end
            copyto!(d_bridge_idx, host_bridge_idx)
            CUDA.@sync @cuda threads = t_std blocks = b_bridge kernel_pack_bridge_view!(
                d_bridge_packed,
                sph.pospres, sph.velrhop, sph.idep, sph.feedback_acc, sph.accad, sph.mass_arr,
                d_bridge_idx,
                sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)),
                par_host.acc_x, par_host.acc_y, par_host.acc_z,
                Int32(bridge_nb)
            )
            copyto!(host_bridge_packed, d_bridge_packed)

            empty!(bridge_id_to_global_idx)
            @inbounds for i in 1:bridge_nb
                host_pos_cb[i] = Float4(host_bridge_packed[i, 1], host_bridge_packed[i, 2], host_bridge_packed[i, 3], host_bridge_packed[i, 4])
                host_vel_cb[i] = Float4(host_bridge_packed[i, 5], host_bridge_packed[i, 6], host_bridge_packed[i, 7], host_bridge_packed[i, 8])
                host_idep_cb[i] = Float4(host_bridge_packed[i, 9], host_bridge_packed[i, 10], host_bridge_packed[i, 11], host_bridge_packed[i, 12])
                host_feedback_acc_cb[i] = Float4(host_bridge_packed[i, 13], host_bridge_packed[i, 14], host_bridge_packed[i, 15], host_bridge_packed[i, 16])
                host_acc_cb[i] = Float4(host_bridge_packed[i, 17], host_bridge_packed[i, 18], host_bridge_packed[i, 19], host_bridge_packed[i, 20])
                host_mass_cb[i] = host_bridge_packed[i, 21]
                host_free_force_cb[i, 1] = host_bridge_packed[i, 22]
                host_free_force_cb[i, 2] = host_bridge_packed[i, 23]
                host_free_force_cb[i, 3] = host_bridge_packed[i, 24]
                host_free_force_total_cb[i, 1] = host_bridge_packed[i, 25]
                host_free_force_total_cb[i, 2] = host_bridge_packed[i, 26]
                host_free_force_total_cb[i, 3] = host_bridge_packed[i, 27]
                gidx = Int(host_bridge_idx[i])
                if gidx > 0
                    oid = Int32(round(host_idep_cb[i][3]))
                    if oid > 0
                        bridge_id_to_global_idx[oid] = gidx
                    end
                end
            end
            local_apply_map = bridge_id_to_global_idx
        else
            copyto!(host_pos_cb, sph.pospres)
            copyto!(host_vel_cb, sph.velrhop)
            copyto!(host_idep_cb, sph.idep)
            copyto!(host_feedback_acc_cb, sph.feedback_acc)
            copyto!(host_acc_cb, sph.accad)
            copyto!(host_mass_cb, sph.mass_arr)
            @inbounds for i in 1:par_host.np
                m = host_mass_cb[i]
                base_ax, base_ay, base_az = resolve_part_acceleration_host(
                    host_idep_cb[i][1], part_acc_ids_host, part_acc_vals_host, par_host
                )
                host_free_force_cb[i, 1] = m * host_feedback_acc_cb[i][1]
                host_free_force_cb[i, 2] = m * host_feedback_acc_cb[i][2]
                host_free_force_cb[i, 3] = m * host_feedback_acc_cb[i][3]
                host_free_force_total_cb[i, 1] = m * (host_acc_cb[i][1] - base_ax)
                host_free_force_total_cb[i, 2] = m * (host_acc_cb[i][2] - base_ay)
                host_free_force_total_cb[i, 3] = m * (host_acc_cb[i][3] - base_az)
            end
        end
        update_timer!(sph.timer, :coupling_extract, t0)

        t0 = time_ns()
        fill!(host_cpl_acc, _zero4())
        fill!(host_acc_target, _zero4())
        fill!(host_vel_target, _zero4())
        fill!(host_pos_target, _zero4())
        fill!(host_target_mask, 0f0)
        fill!(host_stress_target, zero(Tensor2D))
        fill!(host_stress_mask, 0f0)
        callback_result = coupling_callback((
            step = cb_step,
            time = cb_time,
            phase = cb_phase,
            dt = Float64(par_host.dt),
            np = use_bridge_subset_extract ? bridge_nb : Int(par_host.np),
            npb = use_bridge_subset_extract ? 0 : Int(par_host.npb),
            pos = host_pos_cb,
            vel = host_vel_cb,
            idep = host_idep_cb,
            acc = host_acc_cb,
            feedback_acc = host_feedback_acc_cb,
            mass = host_mass_cb,
            free_force = host_free_force_cb,
            free_force_total = host_free_force_total_cb
        ))
        update_timer!(sph.timer, :coupling_callback, t0)

        t0 = time_ns()
        _apply_coupling_callback_result!(
            host_cpl_acc,
            host_acc_target,
            host_vel_target,
            host_pos_target,
            host_target_mask,
            host_stress_target,
            host_stress_mask,
            callback_result,
            use_bridge_subset_extract ? host_idep_full : host_idep_cb;
            id_to_index_override = local_apply_map
        )
        update_timer!(sph.timer, :coupling_apply, t0)

        t0 = time_ns()
        copyto!(host_acc_full, sph.accad)
        has_acc_target = false
        @inbounds for i in eachindex(host_target_mask)
            if host_target_mask[i] > 0.5f0
                at = host_acc_target[i]
                if at[4] > 0.5f0
                    a = host_acc_full[i]
                    host_acc_full[i] = Float4(at[1], at[2], at[3], a[4])
                    has_acc_target = true
                end
            end
        end

        copyto!(sph.cpl_acc, host_cpl_acc)
        if has_acc_target
            copyto!(sph.accad, host_acc_full)
        end
        update_timer!(sph.timer, :coupling_upload, t0)
        update_timer!(sph.timer, :coupling, t_cb_total)
        return nothing
    end

    function refresh_sph_free_acceleration_current!()
        run_adami_boundary!()

        t0 = time_ns()
        @cuda threads=t_int blocks=b_int kernel_particle_interaction(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.feedback_acc, sph.spin, sph.acestn, sph.grid.cell_beg, sph.grid.cell_end, sph.mass_arr, sph.bridge_mask, mat_props, sph.v_vis, sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host)
        CUDA.synchronize()
        update_timer!(sph.timer, :interaction, t0)
    end

    reset_timer!(sph.timer)

    if par_host.integration_method == 2
        # Kick-start for leapfrog: evaluate a^0, then shift v^0 -> v^{1/2}.
        run_adami_boundary!()

        t0 = time_ns()
        @cuda threads=t_int blocks=b_int kernel_particle_interaction(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.feedback_acc, sph.spin, sph.acestn, sph.grid.cell_beg, sph.grid.cell_end, sph.mass_arr, sph.bridge_mask, mat_props, sph.v_vis, sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host)
        CUDA.synchronize()
        update_timer!(sph.timer, :interaction, t0)

        run_coupling_callback!(step, sim_time, :kickstart)
        if coupling_callback !== nothing
            @cuda threads=t_std blocks=b_npm kernel_add_external_acceleration(sph.accad, sph.cpl_acc, par_host.npb, par_host.npm)
            CUDA.synchronize()
        end
        snapshot_acceleration_used!()

        t0 = time_ns()
        @cuda threads=t_lf_kick blocks=b_lf_kick kernel_leapfrog_kickstart(sph.velrhop, sph.accad, par_host)
        CUDA.synchronize()
        update_timer!(sph.timer, :integration, t0)

        # Rebuild the uncoupled acceleration of the current leapfrog state for the first explicit bridge solve.
        refresh_sph_free_acceleration_current!()
    end
    
    while sim_time < Float64(par_host.time_max)
        current_dt = par_host.cfl * par_host.h / par_host.cs
        par_host = update_time_step(par_host, sim_time, current_dt)
        @printf(" SimTime = %0.5f, dt = %0.3e\r", sim_time, par_host.dt)
        
        limit_sq = par_host.velocity_limit ^ 2
        @cuda threads=t_std blocks=b_npm kernel_kill_high_velocity_particles(
            sph.velrhop, 
            sph.idep, 
            limit_sq, 
            par_host.npb,
            par_host.npm
        )
        CUDA.synchronize()

        if par_host.integration_method == 2
            # 0) Explicit bridge solve at t_n: λ^n -> a^n.
            run_coupling_callback!(step, sim_time, :single_step)
            if coupling_callback !== nothing
                @cuda threads=t_std blocks=b_npm kernel_add_external_acceleration(sph.accad, sph.cpl_acc, par_host.npb, par_host.npm)
                CUDA.synchronize()
            end
            snapshot_acceleration_used!()

            # 1) Kinematics update: velocity to n+1/2 and position to n+1.
            t0 = time_ns()
            apply_pending_bridge_targets!()
            @cuda threads=t_lf_kin blocks=b_lf_kin kernel_leapfrog_kinematics(sph.pospres, sph.velrhop, sph.accad, sph.bridge_mask, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :integration, t0)

            # 2) Recompute interactions at updated geometry.
            t0 = time_ns()
            neighbor_search!(sph.grid, par_host, sph.pospres, sph.idep)
            sort_arrays_primary!(sph)
            update_active_boundary!(sph.grid, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :neighbor_search, t0)

            run_adami_boundary!()

            t0 = time_ns()
            @cuda threads=t_int blocks=b_int kernel_particle_interaction(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.feedback_acc, sph.spin, sph.acestn, sph.grid.cell_beg, sph.grid.cell_end, sph.mass_arr, sph.bridge_mask, mat_props, sph.v_vis, sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :interaction, t0)

            # 3) Constitutive+density update to n+1 from latest acestn and accad.
            t0 = time_ns()
            @cuda threads=t_lf_stress blocks=b_lf_stress kernel_leapfrog_stress(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.spin, sph.acestn, mat_props, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :integration, t0)
            apply_pending_bridge_stress_targets!(; clear_after = true)

            # 4) Recompute forces/accelerations with updated stress at n+1.
            run_adami_boundary!()

            t0 = time_ns()
            @cuda threads=t_int blocks=b_int kernel_particle_interaction(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.feedback_acc, sph.spin, sph.acestn, sph.grid.cell_beg, sph.grid.cell_end, sph.mass_arr, sph.bridge_mask, mat_props, sph.v_vis, sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :interaction, t0)

        else
            t0 = time_ns()
            neighbor_search!(sph.grid, par_host, sph.pospres, sph.idep)
            sort_arrays_primary!(sph)
            update_active_boundary!(sph.grid, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :neighbor_search, t0)
            
            run_adami_boundary!()
            
            t0 = time_ns()
           
            @cuda threads=t_int blocks=b_int kernel_particle_interaction(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.feedback_acc, sph.spin, sph.acestn, sph.grid.cell_beg, sph.grid.cell_end, sph.mass_arr, sph.bridge_mask, mat_props, sph.v_vis, sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :interaction, t0)

            run_coupling_callback!(step, sim_time, :predictor_pre)

            if coupling_callback !== nothing
                @cuda threads=t_std blocks=b_npm kernel_add_external_acceleration(sph.accad, sph.cpl_acc, par_host.npb, par_host.npm)
                CUDA.synchronize()
            end
            snapshot_acceleration_used!()
            
            t0 = time_ns()
            @cuda threads=t_integ blocks=b_integ kernel_predictor(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.pospres_ini, sph.velrhop_ini, sph.idep_ini, sph.str_ini, sph.stn_ini, sph.accad, sph.spin, sph.acestn, mat_props, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :integration, t0)

            t0 = time_ns()
            neighbor_search!(sph.grid, par_host, sph.pospres, sph.idep)
            sort_arrays_all!(sph)
            update_active_boundary!(sph.grid, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :neighbor_search, t0)
            
            run_adami_boundary!()
            
            t0 = time_ns()
           
            @cuda threads=t_int blocks=b_int kernel_particle_interaction(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.accad, sph.feedback_acc, sph.spin, sph.acestn, sph.grid.cell_beg, sph.grid.cell_end, sph.mass_arr, sph.bridge_mask, mat_props, sph.v_vis, sph.part_acc_ids, sph.part_acc_vals, Int32(length(part_acc_ids_host)), par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :interaction, t0)

            run_coupling_callback!(step, sim_time + Float64(par_host.dt), :corrector_pre)

            if coupling_callback !== nothing
                @cuda threads=t_std blocks=b_npm kernel_add_external_acceleration(sph.accad, sph.cpl_acc, par_host.npb, par_host.npm)
                CUDA.synchronize()
            end
            
            t0 = time_ns()
            @cuda threads=t_integ blocks=b_integ kernel_corrector(sph.pospres, sph.velrhop, sph.idep, sph.str, sph.stn, sph.pospres_ini, sph.velrhop_ini, sph.idep_ini, sph.str_ini, sph.stn_ini, sph.accad, sph.spin, sph.acestn, mat_props, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :integration, t0)
        end
        
        if par_host.stress_shepard_filter != 0 && step % par_host.stress_shepard_filter == 0 && step != 0
            t0 = time_ns()
            neighbor_search!(sph.grid, par_host, sph.pospres, sph.idep)
            sort_arrays_primary!(sph)
            update_active_boundary!(sph.grid, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :neighbor_search, t0)
            
            @cuda threads=t_std blocks=b_npm kernel_stress_regularization(sph.pospres, sph.velrhop, sph.str, sph.tensor2d_cache, sph.grid.cell_beg, sph.grid.cell_end, par_host)
            sph.str .= sph.tensor2d_cache
        end

        if use_model8_topography && par_host.integration_method != 2
            t0 = time_ns()
            neighbor_search!(sph.grid, par_host, sph.pospres, sph.idep)
            sort_arrays_primary!(sph)
            update_active_boundary!(sph.grid, par_host)
            CUDA.synchronize(); update_timer!(sph.timer, :neighbor_search, t0)
        end
        run_topography_extraction!()

        output_time += par_host.dt
        if output_time >= par_host.output_frequency
            t_save_start = time_ns()
        
            curr = time_ns()
            fps = (step - step_interval) / ((curr - sph.timer.clock_interval)/1e9)
            sph.timer.clock_interval = curr
            used = (curr - sph.timer.clock_initial)/1e9
            total_pred = (par_host.time_max / par_host.time) * used
            needed = total_pred - used
            
            println("\n----------Save-----------")
            @printf("Time = %.5f | Step = %d | FPS = %.5f\n", sim_time, step, fps)
            println("Clock time total : ", time_to_string(total_pred))
            println("Clock time used  : ", time_to_string(used))
            println("Clock time needed: ", time_to_string(needed))
            println("Number of particles: $(par_host.np) | Number of cells: $(par_host.ncell)")
            if par_host.active_boundary_adami != 0 && par_host.static_boundary_grid != 0 && par_host.npb > 0
                println("Active boundary particles: $(sph.grid.active_boundary_count_host) / $(par_host.npb)")
            end
            
            par_host = update_output_counter(par_host, Int32(par_host.output_number + 1))
            enqueue_async_save!(par_host)
            
            output_time = 0.0
            step_interval = step
            update_timer!(sph.timer, :saving, t_save_start)
        end

        sim_time += Float64(par_host.dt)
        step += 1
    end

    t_save_flush = time_ns()
    put!(async_save_jobs, nothing)
    fetch(async_save_worker)
    update_timer!(sph.timer, :saving, t_save_flush)
    
    sph.timer.total = (time_ns() - sph.timer.clock_initial) / 1e9
    accounted = sph.timer.neighbor_search + sph.timer.adami_boundary + sph.timer.interaction + sph.timer.integration + sph.timer.coupling + sph.timer.saving
    sph.timer.misc = max(0.0, sph.timer.total - accounted)
    fps_avg = step / (sph.timer.total - sph.timer.saving)
    
    println("\nSimulation finished ...\n")
    println("----------SUMMARY----------")
    println("1. Total clock time = $(time_to_string(sph.timer.total))")
    println("2. Search and sorting = $(time_to_string(sph.timer.neighbor_search))")
    println("3. Boundary treatment = $(time_to_string(sph.timer.adami_boundary))")
    println("4. Interaction = $(time_to_string(sph.timer.interaction))")
    println("5. Integration = $(time_to_string(sph.timer.integration))")
    println("6. Coupling total = $(time_to_string(sph.timer.coupling))")
    println("   6.1 Coupling extract = $(time_to_string(sph.timer.coupling_extract))")
    println("   6.2 Coupling callback = $(time_to_string(sph.timer.coupling_callback))")
    println("   6.3 Coupling apply = $(time_to_string(sph.timer.coupling_apply))")
    println("   6.4 Coupling upload = $(time_to_string(sph.timer.coupling_upload))")
    println("7. Saving = $(time_to_string(sph.timer.saving))")
    println("8. Misc/overhead = $(time_to_string(sph.timer.misc))")
    println("9. Total steps = $step")
    @printf("10. Average time step = %.5e\n", par_host.time / step)
    @printf("11. Average FPS = %.5f\n", fps_avg)
    println("----------SUMMARY (Percent)----------")
    total = max(sph.timer.total, 1e-12)
    @printf("neighbor_search : %7.3f %%\n", 100 * sph.timer.neighbor_search / total)
    @printf("adami_boundary  : %7.3f %%\n", 100 * sph.timer.adami_boundary / total)
    @printf("interaction     : %7.3f %%\n", 100 * sph.timer.interaction / total)
    @printf("integration     : %7.3f %%\n", 100 * sph.timer.integration / total)
    @printf("coupling_total  : %7.3f %%\n", 100 * sph.timer.coupling / total)
    @printf("saving          : %7.3f %%\n", 100 * sph.timer.saving / total)
    @printf("misc            : %7.3f %%\n", 100 * sph.timer.misc / total)
end

end # module

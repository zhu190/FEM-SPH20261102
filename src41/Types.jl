module Types

using StaticArrays
using CUDA
import Adapt 

export Float4, Tensor2D, Parameters, SimTimer, PI_F, J2, trace
export update_derived_params, update_time_step, update_output_counter
export MaterialProperties

const PI_F = 3.14159265358979323846f0
const Float4 = SVector{4, Float32}

struct Tensor2D
    xx::Float32
    yy::Float32
    zz::Float32
    xy::Float32
    xz::Float32
    yz::Float32
end

@inline Tensor2D(v::Number) = Tensor2D(Float32(v), Float32(v), Float32(v), Float32(v), Float32(v), Float32(v))
@inline Tensor2D(v1, v2, v3, v4, v5, v6) = Tensor2D(Float32(v1), Float32(v2), Float32(v3), Float32(v4), Float32(v5), Float32(v6))

Base.zero(::Type{Tensor2D}) = Tensor2D(0f0)

@inline Base.:+(a::Tensor2D, b::Tensor2D) = Tensor2D(a.xx+b.xx, a.yy+b.yy, a.zz+b.zz, a.xy+b.xy, a.xz+b.xz, a.yz+b.yz)
@inline Base.:-(a::Tensor2D, b::Tensor2D) = Tensor2D(a.xx-b.xx, a.yy-b.yy, a.zz-b.zz, a.xy-b.xy, a.xz-b.xz, a.yz-b.yz)
@inline Base.:*(a::Tensor2D, b::Number) = Tensor2D(a.xx*Float32(b), a.yy*Float32(b), a.zz*Float32(b), a.xy*Float32(b), a.xz*Float32(b), a.yz*Float32(b))
@inline Base.:*(b::Number, a::Tensor2D) = a * b
@inline Base.:/(a::Tensor2D, b::Number) = Tensor2D(a.xx/Float32(b), a.yy/Float32(b), a.zz/Float32(b), a.xy/Float32(b), a.xz/Float32(b), a.yz/Float32(b))

@inline trace(v::Tensor2D) = v.xx + v.yy + v.zz
@inline J2(v::Tensor2D) = v.xx^2 + v.yy^2 + v.zz^2 + 2f0*(v.xy^2 + v.xz^2 + v.yz^2)

# ==============================================================================
# Material Properties (GPU Lookup Table)
# ==============================================================================

struct MaterialProperties{T, TI}
    rho0::T
    K::T
    G::T
    c_model::TI
    param_1::T   # DP-kphi 
    param_2::T   # DP-kpsi; model 8 uses dilatancy angle psi (rad)
    param_3::T   # DP-kcohesion; model 8 uses cohesion c
    d_p::T       # d (m)
    rho_s::T     #  (kg/m³)
    mu_s::T      #  μ_s
    mu_2::T      #  μ_2
    I_0::T       #  I_0
    mu_I_lambda::T # regularization lambda for model 8
end

Adapt.adapt_structure(to, s::MaterialProperties) = MaterialProperties(
    Adapt.adapt(to, s.rho0),
    Adapt.adapt(to, s.K),
    Adapt.adapt(to, s.G),
    Adapt.adapt(to, s.c_model),
    Adapt.adapt(to, s.param_1),
    Adapt.adapt(to, s.param_2),
    Adapt.adapt(to, s.param_3),
    Adapt.adapt(to, s.d_p),
    Adapt.adapt(to, s.rho_s),
    Adapt.adapt(to, s.mu_s),
    Adapt.adapt(to, s.mu_2),
    Adapt.adapt(to, s.I_0),
    Adapt.adapt(to, s.mu_I_lambda)
)

struct Parameters
    np::Int32
    npb::Int32
    npm::Int32
    np_delete::Int32

    domain_size_x::Float32
    domain_size_y::Float32
    domain_size_z::Float32
    domain_min_x::Float32
    domain_min_y::Float32
    domain_min_z::Float32
    
    cell_num_x::Int32
    cell_num_y::Int32
    cell_num_z::Int32
    ncell::Int32
    ncell_one_layer::Int32
    ncell_delete::Int32
    
    h::Float32
    i_h::Float32
    eta::Float32
    dr::Float32
    hdr::Float32

    m::Float32 
    cs::Float32
    cfl::Float32
    rho0::Float32 

    constitutive_model::Int32
    model_par_1::Float32
    model_par_2::Float32
    model_par_3::Float32
    model_par_4::Float32
    model_par_5::Float32

    density_shepard_filter::Int32
    stress_shepard_filter::Int32
    dynamic_time_step::Int32
    integration_method::Int32
    viscosity_type::Int32
    artificial_viscosity_alpha::Float32
    artificial_viscosity_q1::Float32
    artificial_viscosity_q2::Float32
    momentum_form::Int32

    artificial_pressure::Int32
    artificial_pressure_coefficient::Float32
    
    acc_x::Float32
    acc_y::Float32
    acc_z::Float32
    
    time_max::Float32
    time::Float32
    dt::Float32
    output_frequency::Float32
    output_number::Int32

    kernel_normalization_par::Float32
    kernel_gradient_normalization_par::Float32

    kernel_zero::Float32
    kernel_dr::Float32
    kernel_type::Int32

    output_format::Int32
    
    # Velocity threshold for particle culling
    velocity_limit::Float32

    # Adami boundary velocity mode: 0=no-slip, 1=free-slip
    boundary_velocity_type::Int32

    # Prescribed wall velocity used in ghost-velocity reconstruction
    wall_velocity_x::Float32
    wall_velocity_y::Float32
    wall_velocity_z::Float32

    # Initial damping controls
    initial_damping_duration::Float32
    initial_damping_coefficient::Float32

    # PART ID = 0 boundary output mode: 0=initial-only, 1=all-time
    boundary_part0_output_mode::Int32

    # Neighbor-grid mode: 0=legacy full sort, 1=sort static boundary once and material each rebuild
    static_boundary_grid::Int32

    # Boundary treatment mode: 0=launch Adami on all boundaries, 1=launch only material-near active boundaries
    active_boundary_adami::Int32

    # Model-8 topography extraction: physical footprint radius and maximum
    # normal search depth. Non-positive input values are replaced by defaults
    # in update_derived_params.
    depth_footprint_radius::Float32
    depth_search_max::Float32
    normal_orientation_radius::Float32

    # Section 4 of Su et al. (2024): progressive entrainment evaluated on
    # fixed topography particles. Friction angles are stored in radians.
    entrainment_enabled::Int32
    entrainment_pore_pressure::Float32
    entrainment_flow_friction_angle::Float32
    entrainment_bed_friction_angle::Float32
    erosion_depth_max::Float32
    entrainment_speed_min::Float32
    entrainment_rate_max::Float32
end

# Accept mixed numeric input types (e.g. Int64 from XML/older code paths)
# and normalize to the canonical field types used by Parameters.
function Parameters(
    np, npb, npm, np_delete,
    domain_size_x, domain_size_y, domain_size_z,
    domain_min_x, domain_min_y, domain_min_z,
    cell_num_x, cell_num_y, cell_num_z, ncell, ncell_one_layer, ncell_delete,
    h, i_h, eta, dr, hdr,
    m, cs, cfl, rho0,
    constitutive_model, model_par_1, model_par_2, model_par_3, model_par_4, model_par_5,
    density_shepard_filter, stress_shepard_filter, dynamic_time_step, integration_method,
    viscosity_type, artificial_viscosity_alpha, artificial_viscosity_q1, artificial_viscosity_q2, momentum_form,
    artificial_pressure, artificial_pressure_coefficient,
    acc_x, acc_y, acc_z,
    time_max, time, dt, output_frequency, output_number,
    kernel_normalization_par, kernel_gradient_normalization_par,
    kernel_zero, kernel_dr, kernel_type,
    output_format,
    velocity_limit,
    boundary_velocity_type,
    wall_velocity_x, wall_velocity_y, wall_velocity_z,
    initial_damping_duration, initial_damping_coefficient,
    boundary_part0_output_mode,
    static_boundary_grid = Int32(1),
    active_boundary_adami = Int32(1),
    depth_footprint_radius = 0f0,
    depth_search_max = 0f0,
    normal_orientation_radius = 0f0,
    entrainment_enabled = Int32(0),
    entrainment_pore_pressure = 0.55f0,
    entrainment_flow_friction_angle = 0.19739625f0,
    entrainment_bed_friction_angle = 0.5105088f0,
    erosion_depth_max = 0f0,
    entrainment_speed_min = 0.01f0,
    entrainment_rate_max = 0f0
)
    return Parameters(
        Int32(np), Int32(npb), Int32(npm), Int32(np_delete),
        Float32(domain_size_x), Float32(domain_size_y), Float32(domain_size_z),
        Float32(domain_min_x), Float32(domain_min_y), Float32(domain_min_z),
        Int32(cell_num_x), Int32(cell_num_y), Int32(cell_num_z), Int32(ncell), Int32(ncell_one_layer), Int32(ncell_delete),
        Float32(h), Float32(i_h), Float32(eta), Float32(dr), Float32(hdr),
        Float32(m), Float32(cs), Float32(cfl), Float32(rho0),
        Int32(constitutive_model), Float32(model_par_1), Float32(model_par_2), Float32(model_par_3), Float32(model_par_4), Float32(model_par_5),
        Int32(density_shepard_filter), Int32(stress_shepard_filter), Int32(dynamic_time_step), Int32(integration_method),
        Int32(viscosity_type), Float32(artificial_viscosity_alpha), Float32(artificial_viscosity_q1), Float32(artificial_viscosity_q2), Int32(momentum_form),
        Int32(artificial_pressure), Float32(artificial_pressure_coefficient),
        Float32(acc_x), Float32(acc_y), Float32(acc_z),
        Float32(time_max), Float32(time), Float32(dt), Float32(output_frequency), Int32(output_number),
        Float32(kernel_normalization_par), Float32(kernel_gradient_normalization_par),
        Float32(kernel_zero), Float32(kernel_dr), Int32(kernel_type),
        Int32(output_format),
        Float32(velocity_limit),
        Int32(boundary_velocity_type),
        Float32(wall_velocity_x), Float32(wall_velocity_y), Float32(wall_velocity_z),
        Float32(initial_damping_duration), Float32(initial_damping_coefficient),
        Int32(boundary_part0_output_mode),
        Int32(static_boundary_grid),
        Int32(active_boundary_adami),
        Float32(depth_footprint_radius),
        Float32(depth_search_max),
        Float32(normal_orientation_radius),
        Int32(entrainment_enabled),
        Float32(entrainment_pore_pressure),
        Float32(entrainment_flow_friction_angle),
        Float32(entrainment_bed_friction_angle),
        Float32(erosion_depth_max),
        Float32(entrainment_speed_min),
        Float32(entrainment_rate_max)
    )
end

# Backward-compatible constructor (legacy parameter list without boundary-velocity and damping fields).
function Parameters(
    np, npb, npm, np_delete,
    domain_size_x, domain_size_y, domain_size_z,
    domain_min_x, domain_min_y, domain_min_z,
    cell_num_x, cell_num_y, cell_num_z, ncell, ncell_one_layer, ncell_delete,
    h, i_h, eta, dr, hdr,
    m, cs, cfl, rho0,
    constitutive_model, model_par_1, model_par_2, model_par_3, model_par_4, model_par_5,
    density_shepard_filter, stress_shepard_filter, dynamic_time_step, integration_method,
    viscosity_type, artificial_viscosity_alpha, artificial_viscosity_q1, artificial_viscosity_q2, momentum_form,
    artificial_pressure, artificial_pressure_coefficient,
    acc_x, acc_y, acc_z,
    time_max, time, dt, output_frequency, output_number,
    kernel_normalization_par, kernel_gradient_normalization_par,
    kernel_zero, kernel_dr, kernel_type,
    output_format,
    velocity_limit
)
    return Parameters(
        np, npb, npm, np_delete,
        domain_size_x, domain_size_y, domain_size_z,
        domain_min_x, domain_min_y, domain_min_z,
        cell_num_x, cell_num_y, cell_num_z, ncell, ncell_one_layer, ncell_delete,
        h, i_h, eta, dr, hdr,
        m, cs, cfl, rho0,
        constitutive_model, model_par_1, model_par_2, model_par_3, model_par_4, model_par_5,
        density_shepard_filter, stress_shepard_filter, dynamic_time_step, integration_method,
        viscosity_type, artificial_viscosity_alpha, artificial_viscosity_q1, artificial_viscosity_q2, momentum_form,
        artificial_pressure, artificial_pressure_coefficient,
        acc_x, acc_y, acc_z,
        time_max, time, dt, output_frequency, output_number,
        kernel_normalization_par, kernel_gradient_normalization_par,
        kernel_zero, kernel_dr, kernel_type,
        output_format,
        velocity_limit,
        Int32(0), 0f0, 0f0, 0f0,
        0f0, 0f0,
        Int32(1),
        Int32(1),
        Int32(1)
    )
end

function update_derived_params(p::Parameters)
    h = p.dr * p.hdr
    i_h = 1.0f0 / h
    eta = 0.01f0 * h * h
    m = p.dr^3 * p.rho0

    cx = floor(Int32, 0.5f0 * i_h * p.domain_size_x) + 1
    cy = floor(Int32, 0.5f0 * i_h * p.domain_size_y) + 1
    cz = floor(Int32, 0.5f0 * i_h * p.domain_size_z) + 1
    nc_layer = cx * cy * cz
    ncell = 2 * nc_layer + 1
    ncell_del = ncell - 1

    # Keep old input files valid while allowing resolution-independent,
    # physical radii to be supplied explicitly in XML.
    depth_footprint_radius = p.depth_footprint_radius > 0f0 ? max(p.depth_footprint_radius, 2f0 * h) : 2f0 * h
    depth_search_max = p.depth_search_max > 0f0 ? p.depth_search_max : p.domain_size_y
    normal_orientation_radius = p.normal_orientation_radius > 0f0 ? max(p.normal_orientation_radius, 2f0 * h) : 4f0 * h

    if p.kernel_type == Int32(1)
        k_norm = 1.0f0 / (PI_F * h^3)
        k_grad_norm = 1.0f0 / (PI_F * h^5)
    else
        k_norm = 42.0f0 / (256.0f0 * (h^3) * PI_F)
        k_grad_norm = -5.0f0 * 42.0f0 / (256.0f0 * (h^5) * PI_F)
    end
    
    k_zero = 0f0
    k_dr = 0f0
    if p.artificial_pressure == 1
        q = p.dr * i_h
        if p.kernel_type == Int32(1)
            k_zero = k_norm
            if q < 1f0
                k_dr = k_norm * (1f0 - 1.5f0 * q * q + 0.75f0 * q * q * q)
            elseif q < 2f0
                t = 2f0 - q
                k_dr = k_norm * (0.25f0 * t * t * t)
            else
                k_dr = 0f0
            end
        else
            k_zero = k_norm * 16.0f0 * 0.5f0
            k_dr = k_norm * ((2.0f0 - q)^4) * (q + 0.5f0)
        end
    end

    return Parameters(
        p.np, p.npb, p.npm, p.np_delete,
        p.domain_size_x, p.domain_size_y, p.domain_size_z,
        p.domain_min_x, p.domain_min_y, p.domain_min_z,
        cx, cy, cz, ncell, nc_layer, ncell_del,
        h, i_h, eta, p.dr, p.hdr,
        m, p.cs, p.cfl, p.rho0,
        p.constitutive_model, p.model_par_1, p.model_par_2, p.model_par_3, p.model_par_4, p.model_par_5,
        p.density_shepard_filter, p.stress_shepard_filter, p.dynamic_time_step, p.integration_method,
        p.viscosity_type, p.artificial_viscosity_alpha, p.artificial_viscosity_q1, p.artificial_viscosity_q2, p.momentum_form,
        p.artificial_pressure, p.artificial_pressure_coefficient,
        p.acc_x, p.acc_y, p.acc_z,
        p.time_max, p.time, p.dt, p.output_frequency, p.output_number,
        k_norm, k_grad_norm, k_zero, k_dr, p.kernel_type,
        p.output_format,
        p.velocity_limit,
        p.boundary_velocity_type,
        p.wall_velocity_x, p.wall_velocity_y, p.wall_velocity_z,
        p.initial_damping_duration, p.initial_damping_coefficient,
        p.boundary_part0_output_mode,
        p.static_boundary_grid,
        p.active_boundary_adami,
        depth_footprint_radius,
        depth_search_max,
        normal_orientation_radius,
        p.entrainment_enabled,
        p.entrainment_pore_pressure,
        p.entrainment_flow_friction_angle,
        p.entrainment_bed_friction_angle,
        p.erosion_depth_max,
        p.entrainment_speed_min,
        p.entrainment_rate_max
    )
end

function update_time_step(p::Parameters, new_time::Real, new_dt::Real)
    return Parameters(
        p.np, p.npb, p.npm, p.np_delete,
        p.domain_size_x, p.domain_size_y, p.domain_size_z,
        p.domain_min_x, p.domain_min_y, p.domain_min_z,
        p.cell_num_x, p.cell_num_y, p.cell_num_z, p.ncell, p.ncell_one_layer, p.ncell_delete,
        p.h, p.i_h, p.eta, p.dr, p.hdr,
        p.m, p.cs, p.cfl, p.rho0,
        p.constitutive_model, p.model_par_1, p.model_par_2, p.model_par_3, p.model_par_4, p.model_par_5,
        p.density_shepard_filter, p.stress_shepard_filter, p.dynamic_time_step, p.integration_method,
        p.viscosity_type, p.artificial_viscosity_alpha, p.artificial_viscosity_q1, p.artificial_viscosity_q2, p.momentum_form,
        p.artificial_pressure, p.artificial_pressure_coefficient,
        p.acc_x, p.acc_y, p.acc_z,
        p.time_max, Float32(new_time), Float32(new_dt), p.output_frequency, p.output_number,
        p.kernel_normalization_par, p.kernel_gradient_normalization_par, p.kernel_zero, p.kernel_dr, p.kernel_type,
        p.output_format,
        p.velocity_limit,
        p.boundary_velocity_type,
        p.wall_velocity_x, p.wall_velocity_y, p.wall_velocity_z,
        p.initial_damping_duration, p.initial_damping_coefficient,
        p.boundary_part0_output_mode,
        p.static_boundary_grid,
        p.active_boundary_adami,
        p.depth_footprint_radius,
        p.depth_search_max,
        p.normal_orientation_radius,
        p.entrainment_enabled,
        p.entrainment_pore_pressure,
        p.entrainment_flow_friction_angle,
        p.entrainment_bed_friction_angle,
        p.erosion_depth_max,
        p.entrainment_speed_min,
        p.entrainment_rate_max
    )
end

function update_output_counter(p::Parameters, new_count::Integer)
    return Parameters(
        p.np, p.npb, p.npm, p.np_delete,
        p.domain_size_x, p.domain_size_y, p.domain_size_z,
        p.domain_min_x, p.domain_min_y, p.domain_min_z,
        p.cell_num_x, p.cell_num_y, p.cell_num_z, p.ncell, p.ncell_one_layer, p.ncell_delete,
        p.h, p.i_h, p.eta, p.dr, p.hdr,
        p.m, p.cs, p.cfl, p.rho0,
        p.constitutive_model, p.model_par_1, p.model_par_2, p.model_par_3, p.model_par_4, p.model_par_5,
        p.density_shepard_filter, p.stress_shepard_filter, p.dynamic_time_step, p.integration_method,
        p.viscosity_type, p.artificial_viscosity_alpha, p.artificial_viscosity_q1, p.artificial_viscosity_q2, p.momentum_form,
        p.artificial_pressure, p.artificial_pressure_coefficient,
        p.acc_x, p.acc_y, p.acc_z,
        p.time_max, p.time, p.dt, p.output_frequency, Int32(new_count),
        p.kernel_normalization_par, p.kernel_gradient_normalization_par, p.kernel_zero, p.kernel_dr, p.kernel_type,
        p.output_format,
        p.velocity_limit,
        p.boundary_velocity_type,
        p.wall_velocity_x, p.wall_velocity_y, p.wall_velocity_z,
        p.initial_damping_duration, p.initial_damping_coefficient,
        p.boundary_part0_output_mode,
        p.static_boundary_grid,
        p.active_boundary_adami,
        p.depth_footprint_radius,
        p.depth_search_max,
        p.normal_orientation_radius,
        p.entrainment_enabled,
        p.entrainment_pore_pressure,
        p.entrainment_flow_friction_angle,
        p.entrainment_bed_friction_angle,
        p.erosion_depth_max,
        p.entrainment_speed_min,
        p.entrainment_rate_max
    )
end

function Base.zero(::Type{Parameters})
    return Parameters(
        0,0,0,0,
        0f0,0f0,0f0, 0f0,0f0,0f0,
        0,0,0,0,0,0,
        0f0,0f0,0f0,0f0,0f0,
        0f0,0f0,0f0,0f0,
        0,0f0,0f0,0f0,0f0,0f0,
        0,0,0,0,0,0f0,0f0,0f0,0,
        0,0f0,
        0f0,0f0,0f0,
        0f0,0f0,0f0,0f0,0,
        0f0,0f0,0f0,0f0,0,
        0,
        0f0, # velocity_limit
        0,   # boundary_velocity_type
        0f0, 0f0, 0f0, # wall velocity
        0f0, 0f0, # initial damping params
        1,  # boundary_part0_output_mode: default all-time
        1,  # static_boundary_grid: default enabled
        1,  # active_boundary_adami: default enabled
        0f0, # depth_footprint_radius: derive as 2h
        0f0, # depth_search_max: derive from vertical domain size
        0f0, # normal_orientation_radius: derive as 4h
        0,   # entrainment disabled by default
        0.55f0,
        0.19739625f0, # 11.31 degrees
        0.5105088f0,  # 29.25 degrees
        0f0,          # no erodible layer unless explicitly configured
        0.01f0,       # regularize the Eq. (15) low-speed singularity
        0f0           # no explicit rate cap; erosion-depth-max still applies
    )
end

mutable struct SimTimer
    total::Float64
    neighbor_search::Float64
    adami_boundary::Float64
    interaction::Float64
    integration::Float64
    coupling::Float64
    coupling_extract::Float64
    coupling_callback::Float64
    coupling_apply::Float64
    coupling_upload::Float64
    saving::Float64
    misc::Float64
    
    clock_initial::UInt64
    clock_neighbor_search::UInt64
    clock_adami_boundary::UInt64
    clock_interaction::UInt64
    clock_integration::UInt64
    clock_coupling::UInt64
    clock_saving::UInt64
    clock_interval::UInt64

    SimTimer() = new(
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        0, 0, 0, 0, 0, 0, 0, 0
    )
end

end # module

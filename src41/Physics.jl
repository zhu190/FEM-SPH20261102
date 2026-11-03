module Physics

using CUDA
using StaticArrays
using LinearAlgebra
using ..Types: Parameters, Float4, Tensor2D, J2, trace, MaterialProperties
using ..Grid: LoquatGrid

export kernel_adami_boundary, kernel_adami_boundary_active, kernel_particle_interaction, kernel_stress_regularization, get_mat_idx
export kernel_kill_high_velocity_particles
export kernel_add_external_acceleration
export kernel_extract_model8_topography_quantities
export kernel_update_model8_entrainment

const PCA_EIGEN_PI = 3.14159265358979323846f0
const WENDLAND_2D_UNIT_NORM = 7.0f0 / PCA_EIGEN_PI
const BED_MATERIAL_ALIGNMENT_MIN = 0.65f0
const BASE_LAYER_MIN_GAP_FACTOR = 0.25f0
const DEPTH_COVERAGE_MIN = 0.10f0
const WET_TANGENT_RADIUS_FACTOR = 1.5f0
const BED_MATERIAL_MAX_GAP_H_FACTOR = 2.0f0
const BED_MATERIAL_MAX_GAP_DR_FACTOR = 3.0f0
const SURFACE_PCA_TRACE_RATIO_MAX = 0.29f0
const SURFACE_MEAN_OFFSET_H_MIN = 0.08f0
const DEPTH_CONNECTIVITY_BIN_COUNT = Int32(256)

@inline @fastmath function norm3df(x::Float32, y::Float32, z::Float32)
    return sqrt(muladd(x, x, muladd(y, y, z*z)))
end

@inline @fastmath function kernel_wendland_val(q::Float32, norm::Float32)
    t = 2f0 - q
    t2 = t * t
    return norm * (t2 * t2) * (q + 0.5f0)
end

@inline @fastmath function kernel_grad_val(q::Float32, grad_norm::Float32)
    t = 2f0 - q
    return grad_norm * (t * t * t)
end

@inline @fastmath function kernel_cubic_spline_val(q::Float32, norm::Float32)
    if q < 1f0
        return norm * (1f0 - 1.5f0 * q * q + 0.75f0 * q * q * q)
    elseif q < 2f0
        t = 2f0 - q
        return norm * (0.25f0 * t * t * t)
    end
    return 0f0
end

@inline @fastmath function kernel_cubic_spline_grad_val(q::Float32, grad_norm::Float32)
    if q < 1f0
        return grad_norm * (-3f0 + 2.25f0 * q)
    elseif q < 2f0
        t = 2f0 - q
        return grad_norm * (-0.75f0 * t * t / max(q, 1f-12))
    end
    return 0f0
end

# Normalized 2-D Wendland C2 kernel whose compact support is `radius`.
# Its integral over the tangent plane is one, so volume * kernel has units
# of length and directly estimates the local column depth.
@inline @fastmath function kernel_wendland_2d_support(
    tangent2::Float32,
    radius::Float32,
    inv_radius::Float32
)
    if tangent2 >= radius * radius
        return 0f0
    end
    u = sqrt(max(tangent2, 0f0)) * inv_radius
    t = 1f0 - u
    t2 = t * t
    return WENDLAND_2D_UNIT_NORM * inv_radius * inv_radius * t2 * t2 * (4f0 * u + 1f0)
end

@inline @fastmath function kernel_val(q::Float32, par::Parameters)
    if par.kernel_type == Int32(1)
        return kernel_cubic_spline_val(q, par.kernel_normalization_par)
    end
    return kernel_wendland_val(q, par.kernel_normalization_par)
end

@inline @fastmath function kernel_grad_shape_val(q::Float32, par::Parameters)
    if par.kernel_type == Int32(1)
        return kernel_cubic_spline_grad_val(q, par.kernel_gradient_normalization_par)
    end
    return kernel_grad_val(q, par.kernel_gradient_normalization_par)
end

@inline @fastmath function normalize3_or(
    x::Float32, y::Float32, z::Float32,
    fx::Float32, fy::Float32, fz::Float32
)
    n2 = muladd(x, x, muladd(y, y, z * z))
    if !isfinite(n2) || n2 < 1f-20
        return fx, fy, fz
    end
    inv_n = 1f0 / sqrt(n2)
    return x * inv_n, y * inv_n, z * inv_n
end

@inline @fastmath function smallest_eigenvector_sym3(
    a11::Float32, a12::Float32, a13::Float32,
    a22::Float32, a23::Float32, a33::Float32
)
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    if p1 < 1f-20
        if a11 <= a22 && a11 <= a33
            return 1f0, 0f0, 0f0
        elseif a22 <= a33
            return 0f0, 1f0, 0f0
        else
            return 0f0, 0f0, 1f0
        end
    end

    q = (a11 + a22 + a33) / 3f0
    b11 = a11 - q
    b22 = a22 - q
    b33 = a33 - q
    p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2f0 * p1
    p = sqrt(max(p2 / 6f0, 0f0))
    if p < 1f-20
        return 0f0, 0f0, 1f0
    end

    inv_p = 1f0 / p
    c11 = b11 * inv_p
    c12 = a12 * inv_p
    c13 = a13 * inv_p
    c22 = b22 * inv_p
    c23 = a23 * inv_p
    c33 = b33 * inv_p
    det_c = c11 * (c22 * c33 - c23 * c23) -
            c12 * (c12 * c33 - c23 * c13) +
            c13 * (c12 * c23 - c22 * c13)
    r = clamp(det_c * 0.5f0, -1f0, 1f0)
    phi = acos(r) / 3f0
    lambda_min = q + 2f0 * p * cos(phi + 2f0 * PCA_EIGEN_PI / 3f0)

    r11 = a11 - lambda_min
    r22 = a22 - lambda_min
    r33 = a33 - lambda_min

    c12x = a12 * a23 - a13 * r22
    c12y = a13 * a12 - r11 * a23
    c12z = r11 * r22 - a12 * a12
    c13x = a12 * r33 - a13 * a23
    c13y = a13 * a13 - r11 * r33
    c13z = r11 * a23 - a12 * a13
    c23x = r22 * r33 - a23 * a23
    c23y = a23 * a13 - a12 * r33
    c23z = a12 * a23 - r22 * a13

    n12 = c12x * c12x + c12y * c12y + c12z * c12z
    n13 = c13x * c13x + c13y * c13y + c13z * c13z
    n23 = c23x * c23x + c23y * c23y + c23z * c23z

    if n12 >= n13 && n12 >= n23
        return normalize3_or(c12x, c12y, c12z, 0f0, 1f0, 0f0)
    elseif n13 >= n23
        return normalize3_or(c13x, c13y, c13z, 0f0, 1f0, 0f0)
    end
    return normalize3_or(c23x, c23y, c23z, 0f0, 1f0, 0f0)
end

@inline @fastmath function lsdyna_A_operator(
    dr_x::Float32, dr_y::Float32, dr_z::Float32,
    dist::Float32, q::Float32, par::Parameters
)
    if dist <= 1f-12
        return 0f0, 0f0, 0f0, 0f0, 0f0, 0f0
    end
    inv_dist = 1f0 / dist
    nx = dr_x * inv_dist
    ny = dr_y * inv_dist
    nz = dr_z * inv_dist

    # LS-DYNA-style A operator:
    # A_ij = h^(-(d+1)) * e_ij * theta'(r_ij / h)
    a_scalar = kernel_grad_shape_val(q, par) * dist
    aij_x = a_scalar * nx
    aij_y = a_scalar * ny
    aij_z = a_scalar * nz
    return aij_x, aij_y, aij_z, -aij_x, -aij_y, -aij_z
end

@inline @fastmath function invert_3x3_or_identity(
    a11::Float32, a12::Float32, a13::Float32,
    a21::Float32, a22::Float32, a23::Float32,
    a31::Float32, a32::Float32, a33::Float32
)
    c11 = muladd(a22, a33, -a23 * a32)
    c12 = muladd(a13, a32, -a12 * a33)
    c13 = muladd(a12, a23, -a13 * a22)
    c21 = muladd(a23, a31, -a21 * a33)
    c22 = muladd(a11, a33, -a13 * a31)
    c23 = muladd(a13, a21, -a11 * a23)
    c31 = muladd(a21, a32, -a22 * a31)
    c32 = muladd(a12, a31, -a11 * a32)
    c33 = muladd(a11, a22, -a12 * a21)

    det = muladd(a11, c11, muladd(a12, c21, a13 * c31))
    if !isfinite(det) || abs(det) < 1f-6
        return 1f0, 0f0, 0f0,
               0f0, 1f0, 0f0,
               0f0, 0f0, 1f0
    end

    inv_det = 1f0 / det
    return c11 * inv_det, c12 * inv_det, c13 * inv_det,
           c21 * inv_det, c22 * inv_det, c23 * inv_det,
           c31 * inv_det, c32 * inv_det, c33 * inv_det
end

@inline @fastmath function inf_norm_3x3(
    a11::Float32, a12::Float32, a13::Float32,
    a21::Float32, a22::Float32, a23::Float32,
    a31::Float32, a32::Float32, a33::Float32
)
    r1 = abs(a11) + abs(a12) + abs(a13)
    r2 = abs(a21) + abs(a22) + abs(a23)
    r3 = abs(a31) + abs(a32) + abs(a33)
    return max(r1, max(r2, r3))
end

@inline function condinf_3x3_or_inf(
    a11::Float32, a12::Float32, a13::Float32,
    a21::Float32, a22::Float32, a23::Float32,
    a31::Float32, a32::Float32, a33::Float32
)
    c11 = muladd(a22, a33, -a23 * a32)
    c12 = muladd(a13, a32, -a12 * a33)
    c13 = muladd(a12, a23, -a13 * a22)
    c21 = muladd(a23, a31, -a21 * a33)
    c22 = muladd(a11, a33, -a13 * a31)
    c23 = muladd(a13, a21, -a11 * a23)
    c31 = muladd(a21, a32, -a22 * a31)
    c32 = muladd(a12, a31, -a11 * a32)
    c33 = muladd(a11, a22, -a12 * a21)

    det = muladd(a11, c11, muladd(a12, c21, a13 * c31))
    if !isfinite(det) || abs(det) < 1f-6
        return Inf32, false
    end

    inv_det = 1f0 / det
    norma = inf_norm_3x3(
        a11, a12, a13,
        a21, a22, a23,
        a31, a32, a33
    )
    norminv = inf_norm_3x3(
        c11 * inv_det, c12 * inv_det, c13 * inv_det,
        c21 * inv_det, c22 * inv_det, c23 * inv_det,
        c31 * inv_det, c32 * inv_det, c33 * inv_det
    )
    return norma * norminv, true
end

@inline function get_mat_idx(uid::Float32)
    id_i = Int32(uid)
    if id_i < 100
        return 1
    else
        return Int32(div(id_i, 100)) + 1
    end
end

@inline function is_part_id_100(uid::Float32)
    return Int32(round(uid)) == 100
end

@inline function is_part_id_200(uid::Float32)
    return Int32(round(uid)) == 200
end

@inline function resolve_particle_acceleration(
    uid::Float32,
    part_acc_ids,
    part_acc_vals,
    part_acc_count::Int32,
    par::Parameters
)
    uid_i = Int32(round(uid))
    @inbounds for k in Int32(1):part_acc_count
        if part_acc_ids[k] == uid_i
            a = part_acc_vals[k]
            return a[1], a[2], a[3]
        end
    end
    return par.acc_x, par.acc_y, par.acc_z
end


@inline function compute_adami_boundary_at!(
    i::Int32, pospres, velrhop, idep, str, v_vis, cell_beg, cell_end, mat_props,
    part_acc_ids, part_acc_vals, part_acc_count::Int32, par::Parameters
)
    @inbounds begin
            pos_i = pospres[i]
            if !isfinite(pos_i[1]) return nothing end
            uid_i = idep[i][1]
            acc_x_i, acc_y_i, acc_z_i = resolve_particle_acceleration(
                uid_i, part_acc_ids, part_acc_vals, part_acc_count, par
            )

          
            s_xx, s_yy, s_zz = 0f0, 0f0, 0f0
            s_xy, s_xz, s_yz = 0f0, 0f0, 0f0
            v_x, v_y, v_z = 0f0, 0f0, 0f0
            rho0_b = mat_props.rho0[1]
            sp_x, sp_y, sp_z = 0f0, 0f0, 0f0
            sumker = 0f0
            sumker_vel = 0f0

            ih = par.i_h
            cx = floor(Int32, 0.5f0 * (pos_i[1] - par.domain_min_x) * ih)
            cy = floor(Int32, 0.5f0 * (pos_i[2] - par.domain_min_y) * ih)
            cz = floor(Int32, 0.5f0 * (pos_i[3] - par.domain_min_z) * ih)

            for dcz = Int32(-1):Int32(1)
                for dcy = Int32(-1):Int32(1)
                    for dcx = Int32(-1):Int32(1)
                        ncx, ncy, ncz = cx+dcx, cy+dcy, cz+dcz
                        if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                            c_idx = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz + par.ncell_one_layer + 1
                            p_beg = cell_beg[c_idx]
                            p_end = cell_end[c_idx]

                            if p_beg != -1
                                for j = p_beg:(p_end-1)
                                    pos_j = pospres[j]
                                    dr_x = pos_i[1] - pos_j[1]
                                    dr_y = pos_i[2] - pos_j[2]
                                    dr_z = pos_i[3] - pos_j[3]
                                    dist2 = muladd(dr_x, dr_x, muladd(dr_y, dr_y, dr_z*dr_z))
                                    dist = sqrt(dist2)
                                    q = dist * ih

                                    if q < 2f0
                                        vel_j = velrhop[j]
                                        w_ij = kernel_val(q, par)
                                        kv = w_ij * (par.m / vel_j[4])

                                      
                                      
                                        sp_x += dr_x * kv
                                        sp_y += dr_y * kv
                                        sp_z += dr_z * kv

                                    
                                        sj = str[j]
                                        s_xx += sj.xx * kv
                                        s_yy += sj.yy * kv
                                        s_zz += sj.zz * kv
                                        s_xy += sj.xy * kv
                                        s_xz += sj.xz * kv
                                        s_yz += sj.yz * kv

                                      
                                        # Keep velocity extrapolation from real material particles only.
                                        if j > par.npb
                                            v_x += vel_j[1] * kv
                                            v_y += vel_j[2] * kv
                                            v_z += vel_j[3] * kv
                                            sumker_vel += kv
                                        end

                                        sumker += kv
                                    end
                                end
                            end
                        end
                    end
                end
            end

            if sumker > 1f-6
                inv = 1f0 / sumker


                # Legacy-stable component-wise gravity correction:
                # sigma_nn,w = (sum(sigma_nn*W*V) - rho0*g_n*sum(dr_n*W*V))/sum(W*V)
                s_xx_b = (s_xx - rho0_b * acc_x_i * sp_x) * inv
                s_yy_b = (s_yy - rho0_b * acc_y_i * sp_y) * inv
                s_zz_b = (s_zz - rho0_b * acc_z_i * sp_z) * inv
                if par.boundary_velocity_type == 1
                    # Free-slip: mirror shear stress to reduce interfacial tangential traction.
                    s_xy_b = -s_xy * inv
                    s_xz_b = -s_xz * inv
                    s_yz_b = -s_yz * inv
                else
                    # No-slip / rough wall: keep shear-stress sign to preserve basal friction.
                    s_xy_b = s_xy * inv
                    s_xz_b = s_xz * inv
                    s_yz_b = s_yz * inv
                end

                # Boundary pressure is derived from stress mean value (not EOS/hydrostatic fluid relation).
                p_b = -(s_xx_b + s_yy_b + s_zz_b) * (1f0 / 3f0)
                pospres[i] = Float4(pos_i[1], pos_i[2], pos_i[3], p_b)
                velrhop[i] = Float4(par.wall_velocity_x, par.wall_velocity_y, par.wall_velocity_z, velrhop[i][4])

               
                # Adami no-slip ghost velocity with volume weighting:
                # v_tilde = sum(v_j * W_ij * V_j) / sum(W_ij * V_j), V_j = m_j / rho_j
                if sumker_vel > 1f-6
                    inv_vel = 1f0 / sumker_vel
                    v_tilde_x = v_x * inv_vel
                    v_tilde_y = v_y * inv_vel
                    v_tilde_z = v_z * inv_vel
                    v_vis[i] = Float4(
                        2f0 * par.wall_velocity_x - v_tilde_x,
                        2f0 * par.wall_velocity_y - v_tilde_y,
                        2f0 * par.wall_velocity_z - v_tilde_z,
                        0f0
                    )
                else
                    v_vis[i] = Float4(par.wall_velocity_x, par.wall_velocity_y, par.wall_velocity_z, 0f0)
                end

               
                str[i] = Tensor2D(
                    s_xx_b,
                    s_yy_b,
                    s_zz_b,
                    s_xy_b,
                    s_xz_b,
                    s_yz_b
                )
            else
               
                pospres[i] = Float4(pos_i[1], pos_i[2], pos_i[3], 0f0)
                velrhop[i] = Float4(par.wall_velocity_x, par.wall_velocity_y, par.wall_velocity_z, velrhop[i][4])
                v_vis[i] = Float4(par.wall_velocity_x, par.wall_velocity_y, par.wall_velocity_z, 0f0)
                str[i] = Tensor2D(0f0)
            end
    end
    return nothing
end 

function kernel_adami_boundary(
    pospres, velrhop, idep, str, v_vis, cell_beg, cell_end, mat_props,
    part_acc_ids, part_acc_vals, part_acc_count::Int32, par::Parameters
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= par.npb
        compute_adami_boundary_at!(
            Int32(i), pospres, velrhop, idep, str, v_vis, cell_beg, cell_end, mat_props,
            part_acc_ids, part_acc_vals, part_acc_count, par
        )
    end
    return nothing
end

function kernel_adami_boundary_active(
    pospres, velrhop, idep, str, v_vis, cell_beg, cell_end, active_boundary_indices,
    active_count::Int32, mat_props, part_acc_ids, part_acc_vals, part_acc_count::Int32, par::Parameters
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= active_count
        @inbounds begin
            i = active_boundary_indices[idx]
            compute_adami_boundary_at!(
                i, pospres, velrhop, idep, str, v_vis, cell_beg, cell_end, mat_props,
                part_acc_ids, part_acc_vals, part_acc_count, par
            )
        end
    end
    return nothing
end

function kernel_particle_interaction(
    pospres, velrhop, idep, str, stn, accad, feedback_acc, spin, acestn,
    cell_beg, cell_end, mass_arr, bridge_mask, mat_props, v_vis,
    part_acc_ids, part_acc_vals, part_acc_count::Int32, par::Parameters
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npm
        i = idx + par.npb
        @inbounds begin
            pos_i = pospres[i]
            if !isfinite(pos_i[1]) return nothing end
            
            uid_i = idep[i][1]
            mat_idx_i = get_mat_idx(uid_i)
            skip_boundary_interaction = is_part_id_200(uid_i)
            skip_boundary_correction = is_part_id_100(uid_i)
            acc_x_i, acc_y_i, acc_z_i = resolve_particle_acceleration(
                uid_i, part_acc_ids, part_acc_vals, part_acc_count, par
            )
            
            mass_i = mass_arr[i]
            rho0_i = mat_props.rho0[mat_idx_i]
            K_i    = mat_props.K[mat_idx_i]
            G_i    = mat_props.G[mat_idx_i]
            # old c formula:
            # cs_i = (rho0_i > 1f-6) ? sqrt(K_i / rho0_i) : par.cs
            # old cs_vis formula:
            # denom_Ei = 3f0 * K_i + G_i
            # E_vis_i = (rho0_i > 1f-6 && G_i > 1f-6 && denom_Ei > 1f-6) ? (9f0 * K_i * G_i / denom_Ei) : 0f0
            # cs_vis_i = (E_vis_i > 1f-6 && rho0_i > 1f-6) ? sqrt(E_vis_i / rho0_i) : cs_i
            rho_i = max(velrhop[i][4], 1f-6)
            c2_i = 4f0 * G_i / 3f0 + K_i
            cs_i = (c2_i > 1f-6) ? sqrt(c2_i / rho_i) : par.cs

            vel_i = velrhop[i]
            rho2_i = vel_i[4] * vel_i[4]
            inv_rho2_i = 1f0 / rho2_i
            s_i = str[i]
            
            s_i_xx_r = s_i.xx * inv_rho2_i; s_i_yy_r = s_i.yy * inv_rho2_i; s_i_zz_r = s_i.zz * inv_rho2_i
            s_i_xy_r = s_i.xy * inv_rho2_i; s_i_xz_r = s_i.xz * inv_rho2_i; s_i_yz_r = s_i.yz * inv_rho2_i
            
            artp_i = 0f0
            if par.artificial_pressure == 1
                tr = s_i_xx_r + s_i_yy_r + s_i_zz_r
                if -tr < 0f0
                    artp_i = (-tr) * -par.artificial_pressure_coefficient
                end
            end

            ax, ay, az, drho = 0f0, 0f0, 0f0, 0f0
            ax_fb, ay_fb, az_fb = 0f0, 0f0, 0f0
            gux, guy, guz = 0f0, 0f0, 0f0; gvx, gvy, gvz = 0f0, 0f0, 0f0; gwx, gwy, gwz = 0f0, 0f0, 0f0
            is_bridge_i = bridge_mask[i] > 0.5f0
            
            ih = par.i_h
            
            cx = floor(Int32, 0.5f0 * (pos_i[1] - par.domain_min_x) * ih)
            cy = floor(Int32, 0.5f0 * (pos_i[2] - par.domain_min_y) * ih)
            cz = floor(Int32, 0.5f0 * (pos_i[3] - par.domain_min_z) * ih)
            
            k_gnorm = par.kernel_gradient_normalization_par

            li11, li12, li13 = 1f0, 0f0, 0f0
            li21, li22, li23 = 0f0, 1f0, 0f0
            li31, li32, li33 = 0f0, 0f0, 1f0
            use_corrected_gradient = (par.momentum_form == 2 || par.momentum_form == 3)
            has_boundary_neighbor = false
            material_neighbor_count = Int32(0)
            material_support_sum = 0f0

            if par.momentum_form == 2 || par.momentum_form == 3
                l11, l12, l13 = 0f0, 0f0, 0f0
                l21, l22, l23 = 0f0, 0f0, 0f0
                l31, l32, l33 = 0f0, 0f0, 0f0
                boundary_exclusion2 = 6.25f0 * par.h * par.h

                for dcz = Int32(-1):Int32(1)
                    for dcy = Int32(-1):Int32(1)
                        for dcx = Int32(-1):Int32(1)
                            ncx, ncy, ncz = cx+dcx, cy+dcy, cz+dcz
                            if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                                base = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz
                                for lk = Int32(0):Int32(1)
                                    off = (lk==0) ? 0 : par.ncell_one_layer
                                    c_idx = base + off + 1
                                    p_beg = cell_beg[c_idx]
                                    p_end = cell_end[c_idx]
                                    if p_beg != -1
                                        for j = p_beg:(p_end-1)
                                            if i == j; continue; end
                                            if skip_boundary_interaction && j <= par.npb
                                                continue
                                            end

                                            pos_j = pospres[j]
                                            dr_x = pos_i[1]-pos_j[1]
                                            dr_y = pos_i[2]-pos_j[2]
                                            dr_z = pos_i[3]-pos_j[3]
                                            dist2 = muladd(dr_x, dr_x, muladd(dr_y, dr_y, dr_z*dr_z))
                                            if skip_boundary_correction && j <= par.npb && dist2 <= boundary_exclusion2
                                                has_boundary_neighbor = true
                                            end
                                            if dist2 * ih * ih < 4f0
                                                if skip_boundary_correction && j <= par.npb
                                                    continue
                                                end

                                                dist = sqrt(dist2)
                                                q = dist * ih
                                                dw_val = kernel_grad_shape_val(q, par)
                                                g_x = dw_val * dr_x
                                                g_y = dw_val * dr_y
                                                g_z = dw_val * dr_z

                                                vel_j = velrhop[j]
                                                vol_j = mass_arr[j] / vel_j[4]
                                                rx = -dr_x
                                                ry = -dr_y
                                                rz = -dr_z
                                                wab = kernel_val(q, par)

                                                l11 += vol_j * rx * g_x
                                                l12 += vol_j * rx * g_y
                                                l13 += vol_j * rx * g_z
                                                l21 += vol_j * ry * g_x
                                                l22 += vol_j * ry * g_y
                                                l23 += vol_j * ry * g_z
                                                l31 += vol_j * rz * g_x
                                                l32 += vol_j * rz * g_y
                                                l33 += vol_j * rz * g_z
                                                if skip_boundary_correction
                                                    material_neighbor_count += 1
                                                    material_support_sum += wab * vol_j
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                (li11, li12, li13,
                 li21, li22, li23,
                 li31, li32, li33) = invert_3x3_or_identity(
                    l11, l12, l13,
                    l21, l22, l23,
                    l31, l32, l33
                )
                if skip_boundary_correction
                    condinf_l, okcond_l = condinf_3x3_or_inf(
                        l11, l12, l13,
                        l21, l22, l23,
                        l31, l32, l33
                    )
                    enough_support = material_neighbor_count >= Int32(12) && material_support_sum >= 0.4f0
                    use_corrected_gradient = !has_boundary_neighbor &&
                                             enough_support &&
                                             okcond_l &&
                                             isfinite(condinf_l) &&
                                             condinf_l <= 60f0
                end
            end
            
            for dcz = Int32(-1):Int32(1)
                for dcy = Int32(-1):Int32(1)
                    for dcx = Int32(-1):Int32(1)
                        ncx, ncy, ncz = cx+dcx, cy+dcy, cz+dcz
                        if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                            base = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz
                            for lk = Int32(0):Int32(1)
                                off = (lk==0) ? 0 : par.ncell_one_layer
                                c_idx = base + off + 1
                                p_beg = cell_beg[c_idx]
                                p_end = cell_end[c_idx]
                                
                                if p_beg != -1
                                    for j = p_beg:(p_end-1)
                                        if i == j; continue; end
                                        if skip_boundary_interaction && j <= par.npb
                                            continue
                                        end
                                        
                                        uid_j = idep[j][1]
                                        mat_idx_j = get_mat_idx(uid_j)
                                        mass_j = mass_arr[j]
                                        rho0_j = mat_props.rho0[mat_idx_j]
                                        K_j    = mat_props.K[mat_idx_j]
                                        G_j    = mat_props.G[mat_idx_j]
                                        # old c formula:
                                        # cs_j = (rho0_j > 1f-6) ? sqrt(K_j / rho0_j) : par.cs
                                        # old cs_vis formula:
                                        # denom_Ej = 3f0 * K_j + G_j
                                        # E_vis_j = (rho0_j > 1f-6 && G_j > 1f-6 && denom_Ej > 1f-6) ? (9f0 * K_j * G_j / denom_Ej) : 0f0
                                        # cs_vis_j = (E_vis_j > 1f-6 && rho0_j > 1f-6) ? sqrt(E_vis_j / rho0_j) : cs_j
                                        rho_j_vis = max(velrhop[j][4], 1f-6)
                                        c2_j = 4f0 * G_j / 3f0 + K_j
                                        cs_j = (c2_j > 1f-6) ? sqrt(c2_j / rho_j_vis) : par.cs

                                        pos_j = pospres[j]
                                        dr_x = pos_i[1]-pos_j[1]
                                        dr_y = pos_i[2]-pos_j[2]
                                        dr_z = pos_i[3]-pos_j[3]
                                        dist2 = muladd(dr_x, dr_x, muladd(dr_y, dr_y, dr_z*dr_z))
                                        
                                        if dist2 * ih * ih < 4f0
                                            dist = sqrt(dist2)
                                            q = dist * ih
                                            dw_val = kernel_grad_shape_val(q, par)
                                            dw_x_raw = dw_val * dr_x
                                            dw_y_raw = dw_val * dr_y
                                            dw_z_raw = dw_val * dr_z
                                            dw_x = dw_x_raw
                                            dw_y = dw_y_raw
                                            dw_z = dw_z_raw
                                            if use_corrected_gradient
                                                if !(skip_boundary_correction && j <= par.npb)
                                                    dw_x = muladd(li11, dw_x_raw, muladd(li12, dw_y_raw, li13 * dw_z_raw))
                                                    dw_y = muladd(li21, dw_x_raw, muladd(li22, dw_y_raw, li23 * dw_z_raw))
                                                    dw_z = muladd(li31, dw_x_raw, muladd(li32, dw_y_raw, li33 * dw_z_raw))
                                                end
                                            end
                                            
                                            vel_j = velrhop[j]
                                            
                                   
                                            # Strict Adami no-slip treatment:
                                            # use mirrored ghost velocity only in viscous term;
                                            # continuity/velocity-gradient use physical wall velocity.
                                            v_jx_real::Float32 = (j <= par.npb) ? par.wall_velocity_x : vel_j[1]
                                            v_jy_real::Float32 = (j <= par.npb) ? par.wall_velocity_y : vel_j[2]
                                            v_jz_real::Float32 = (j <= par.npb) ? par.wall_velocity_z : vel_j[3]
                                            
                                            du_real::Float32 = vel_i[1] - v_jx_real
                                            dv_real::Float32 = vel_i[2] - v_jy_real
                                            dw_real::Float32 = vel_i[3] - v_jz_real
                                            
                                        
                                            drho = drho + mass_j * (du_real * dw_x + dv_real * dw_y + dw_real * dw_z)
                                            
                                      
                                            du::Float32 = vel_i[1] - vel_j[1]
                                            dv::Float32 = vel_i[2] - vel_j[2]
                                            dw::Float32 = vel_i[3] - vel_j[3]
                                            
                                            
                                            vol_j::Float32 = mass_j / vel_j[4]
                                            
                                            gux -= du_real * dw_x * vol_j
                                            guy -= du_real * dw_y * vol_j
                                            guz -= du_real * dw_z * vol_j
                                            gvx -= dv_real * dw_x * vol_j
                                            gvy -= dv_real * dw_y * vol_j
                                            gvz -= dv_real * dw_z * vol_j
                                            gwx -= dw_real * dw_x * vol_j
                                            gwy -= dw_real * dw_y * vol_j
                                            gwz -= dw_real * dw_z * vol_j
                                            
                                    
                                            art_visc = 0f0
                                            if par.viscosity_type == 1
                                             
                                                if j <= par.npb
                                                    vj_vis = v_vis[j]
                                                    du_vis = vel_i[1] - vj_vis[1]
                                                    dv_vis = vel_i[2] - vj_vis[2]
                                                    dw_vis = vel_i[3] - vj_vis[3]
                                                else
                                                    du_vis = du
                                                    dv_vis = dv
                                                    dw_vis = dw
                                                end
                                                
                                                v_dot_r = muladd(du_vis, dr_x, muladd(dv_vis, dr_y, dw_vis*dr_z))
                                                if v_dot_r < 0f0
                                                    mu = par.hdr * par.dr * v_dot_r / (dist2 + par.eta)
                                                    cs_avg = 0.5f0 * (cs_i + cs_j)
                                                    rho_avg = 0.5f0 * (vel_i[4] + vel_j[4])
                                                    art_visc = par.artificial_viscosity_alpha * cs_avg * mu / rho_avg
                                                end
                                            elseif par.viscosity_type == 2
                                                if j <= par.npb
                                                    vj_vis = v_vis[j]
                                                    du_vis = vel_i[1] - vj_vis[1]
                                                    dv_vis = vel_i[2] - vj_vis[2]
                                                    dw_vis = vel_i[3] - vj_vis[3]
                                                else
                                                    du_vis = du
                                                    dv_vis = dv
                                                    dw_vis = dw
                                                end

                                                v_dot_r = muladd(du_vis, dr_x, muladd(dv_vis, dr_y, dw_vis * dr_z))
                                                if v_dot_r < 0f0
                                                    h_ij = par.h
                                                    phi_reg = 0.1f0 * h_ij
                                                    phi_ij = h_ij * v_dot_r / (dist2 + phi_reg * phi_reg)
                                                    # old formula:
                                                    # cs_avg = 0.5f0 * (cs_vis_i + cs_vis_j)
                                                    cs_avg = 0.5f0 * (cs_i + cs_j)
                                                    rho_avg = max(0.5f0 * (vel_i[4] + vel_j[4]), 1f-12)
                                                    q_pos = (-par.artificial_viscosity_q2 * cs_avg * phi_ij +
                                                             par.artificial_viscosity_q1 * phi_ij * phi_ij) / rho_avg
                                                    art_visc = -q_pos
                                                end
                                            end
                                            
                                         
                                            artp_term = 0f0
                                            if par.artificial_pressure == 1
                                                inv_rho2_j = 1f0/(vel_j[4]*vel_j[4])
                                                s_j = str[j]
                                                tr_j = (s_j.xx+s_j.yy+s_j.zz)*inv_rho2_j
                                                if -tr_j < 0f0
                                                    ap_j = (-tr_j)*-par.artificial_pressure_coefficient
                                                    tot = artp_i + ap_j
                                                    if tot > 0f0
                                                        w_ij = kernel_val(q, par)
                                                        base_w = w_ij / par.kernel_dr
                                                        t2 = base_w*base_w
                                                        artp_term = tot * (t2*t2)
                                                    end
                                                end
                                            end
                                            
                                            rho_j = vel_j[4]
                                            inv_rho2_j = 1f0 / (rho_j * rho_j)
                                            s_j = str[j]
                                            if par.momentum_form == 1 || par.momentum_form == 3
                                                inv_rhoij = 1f0 / (vel_i[4] * rho_j)
                                                if par.momentum_form == 3
                                                    aij_x, aij_y, aij_z = dw_x, dw_y, dw_z
                                                    aji_x, aji_y, aji_z = -dw_x, -dw_y, -dw_z
                                                else
                                                    aij_x, aij_y, aij_z, aji_x, aji_y, aji_z =
                                                        lsdyna_A_operator(dr_x, dr_y, dr_z, dist, q, par)
                                                end

                                                sigi_aij_x = muladd(s_i.xx, aij_x, muladd(s_i.xy, aij_y, s_i.xz * aij_z))
                                                sigi_aij_y = muladd(s_i.xy, aij_x, muladd(s_i.yy, aij_y, s_i.yz * aij_z))
                                                sigi_aij_z = muladd(s_i.xz, aij_x, muladd(s_i.yz, aij_y, s_i.zz * aij_z))

                                                sigj_aji_x = muladd(s_j.xx, aji_x, muladd(s_j.xy, aji_y, s_j.xz * aji_z))
                                                sigj_aji_y = muladd(s_j.xy, aji_x, muladd(s_j.yy, aji_y, s_j.yz * aji_z))
                                                sigj_aji_z = muladd(s_j.xz, aji_x, muladd(s_j.yz, aji_y, s_j.zz * aji_z))

                                                stab_x = (art_visc - artp_term) * aij_x
                                                stab_y = (art_visc - artp_term) * aij_y
                                                stab_z = (art_visc - artp_term) * aij_z

                                                dax = mass_j * inv_rhoij * (sigi_aij_x - sigj_aji_x) + mass_j * stab_x
                                                day = mass_j * inv_rhoij * (sigi_aij_y - sigj_aji_y) + mass_j * stab_y
                                                daz = mass_j * inv_rhoij * (sigi_aij_z - sigj_aji_z) + mass_j * stab_z
                                            else
                                                txx = s_i_xx_r + s_j.xx * inv_rho2_j + art_visc - artp_term
                                                tyy = s_i_yy_r + s_j.yy * inv_rho2_j + art_visc - artp_term
                                                tzz = s_i_zz_r + s_j.zz * inv_rho2_j + art_visc - artp_term
                                                txy = s_i_xy_r + s_j.xy * inv_rho2_j
                                                txz = s_i_xz_r + s_j.xz * inv_rho2_j
                                                tyz = s_i_yz_r + s_j.yz * inv_rho2_j
                                                
                                                dax = mass_j * muladd(txx, dw_x, muladd(txy, dw_y, txz*dw_z))
                                                day = mass_j * muladd(txy, dw_x, muladd(tyy, dw_y, tyz*dw_z))
                                                daz = mass_j * muladd(txz, dw_x, muladd(tyz, dw_y, tzz*dw_z))
                                            end

                                            ax += dax
                                            ay += day
                                            az += daz

                                            # FEM feedback should only carry bridge-to-outside physical-particle interaction.
                                            # Do not feed back bridge-bridge or bridge-boundary SPH support forces.
                                            if !is_bridge_i || ((j > par.npb) && !(bridge_mask[j] > 0.5f0))
                                                ax_fb += dax
                                                ay_fb += day
                                                az_fb += daz
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
            if par.initial_damping_coefficient > 0f0 && par.initial_damping_duration > 0f0 &&
               par.time < par.initial_damping_duration && rho0_i > 1f-12 && par.h > 1f-12
                denom_E = 3f0 * K_i + G_i
                E_i = (G_i > 1f-12 && denom_E > 1f-12) ? (9f0 * K_i * G_i / denom_E) : K_i
                if E_i > 0f0
                    cd = par.initial_damping_coefficient * sqrt(E_i / (rho0_i * par.h * par.h))
                    ax -= cd * vel_i[1]
                    ay -= cd * vel_i[2]
                    az -= cd * vel_i[3]
                end
            end

            accad[i] = Float4(ax + acc_x_i, ay + acc_y_i, az + acc_z_i, drho)
            feedback_acc[i] = Float4(ax_fb, ay_fb, az_fb, drho)
            acestn[i] = Tensor2D(gux, gvy, gwz, 0.5f0*(guy+gvx), 0.5f0*(guz+gwx), 0.5f0*(gvz+gwy))
            spin[i] = Float4(0.5f0*(gvx-guy), 0.5f0*(gwx-guz), 0.5f0*(gwy-gvz), 0f0)
        end
    end
    return nothing
end

function kernel_extract_model8_topography_quantities(
    topo_normal,
    topo_quantities,
    topo_basal_velocity,
    pospres,
    velrhop,
    idep,
    cell_beg,
    cell_end,
    mass_arr,
    mat_props,
    model8_centroid::Float4,
    active_boundary_indices,
    work_count::Int32,
    use_active::Int32,
    par::Parameters
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= work_count
        @inbounds begin
            i = (use_active != 0) ? active_boundary_indices[idx] : idx
            if i <= 0 || i > par.npb
                return nothing
            end

            pos_i = pospres[i]
            uid_i = Int32(round(idep[i][1]))
            if !isfinite(pos_i[1]) || idep[i][1] == -1f0 || uid_i != 0
                topo_normal[i] = Float4(0f0, 0f0, 0f0, 0f0)
                topo_quantities[i] = Float4(0f0, 0f0, 0f0, 0f0)
                topo_basal_velocity[i] = Float4(0f0, 0f0, 0f0, 0f0)
                return nothing
            end

            ih = par.i_h
            cx = floor(Int32, 0.5f0 * (pos_i[1] - par.domain_min_x) * ih)
            cy = floor(Int32, 0.5f0 * (pos_i[2] - par.domain_min_y) * ih)
            cz = floor(Int32, 0.5f0 * (pos_i[3] - par.domain_min_z) * ih)

            mean_x, mean_y, mean_z = 0f0, 0f0, 0f0
            boundary_weight = 0f0
            boundary_count = Int32(0)
            for dcz = Int32(-1):Int32(1)
                for dcy = Int32(-1):Int32(1)
                    for dcx = Int32(-1):Int32(1)
                        ncx, ncy, ncz = cx + dcx, cy + dcy, cz + dcz
                        if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                            c_idx = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz + Int32(1)
                            p_beg = cell_beg[c_idx]
                            p_end = cell_end[c_idx]
                            if p_beg != -1
                                for j = p_beg:(p_end - 1)
                                    if j <= par.npb && idep[j][1] != -1f0
                                        pos_j = pospres[j]
                                        dx = pos_i[1] - pos_j[1]
                                        dy = pos_i[2] - pos_j[2]
                                        dz = pos_i[3] - pos_j[3]
                                        scaled_dist2 = muladd(dx, dx, muladd(dy, dy, dz * dz)) * ih * ih
                                        if scaled_dist2 < 4f0
                                            # Compact distance weighting gives nearby points
                                            # more influence on curved terrain and reduces
                                            # contamination from an adjacent face at a corner.
                                            wt = 1f0 - 0.25f0 * scaled_dist2
                                            wb = wt * wt
                                            mean_x += pos_j[1] * wb
                                            mean_y += pos_j[2] * wb
                                            mean_z += pos_j[3] * wb
                                            boundary_weight += wb
                                            boundary_count += Int32(1)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            nx, ny, nz = 0f0, 1f0, 0f0
            surface_like = false
            if boundary_count >= 3 && boundary_weight > 1f-12
                inv_weight = 1f0 / boundary_weight
                mean_x *= inv_weight
                mean_y *= inv_weight
                mean_z *= inv_weight

                c11, c12, c13 = 0f0, 0f0, 0f0
                c22, c23, c33 = 0f0, 0f0, 0f0
                for dcz = Int32(-1):Int32(1)
                    for dcy = Int32(-1):Int32(1)
                        for dcx = Int32(-1):Int32(1)
                            ncx, ncy, ncz = cx + dcx, cy + dcy, cz + dcz
                            if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                                c_idx = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz + Int32(1)
                                p_beg = cell_beg[c_idx]
                                p_end = cell_end[c_idx]
                                if p_beg != -1
                                    for j = p_beg:(p_end - 1)
                                        if j <= par.npb && idep[j][1] != -1f0
                                            pos_j = pospres[j]
                                            dx0 = pos_i[1] - pos_j[1]
                                            dy0 = pos_i[2] - pos_j[2]
                                            dz0 = pos_i[3] - pos_j[3]
                                            scaled_dist2 = muladd(dx0, dx0, muladd(dy0, dy0, dz0 * dz0)) * ih * ih
                                            if scaled_dist2 < 4f0
                                                wt = 1f0 - 0.25f0 * scaled_dist2
                                                wb = wt * wt
                                                px = pos_j[1] - mean_x
                                                py = pos_j[2] - mean_y
                                                pz = pos_j[3] - mean_z
                                                c11 += wb * px * px
                                                c12 += wb * px * py
                                                c13 += wb * px * pz
                                                c22 += wb * py * py
                                                c23 += wb * py * pz
                                                c33 += wb * pz * pz
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                c11 *= inv_weight
                c12 *= inv_weight
                c13 *= inv_weight
                c22 *= inv_weight
                c23 *= inv_weight
                c33 *= inv_weight
                nx, ny, nz = smallest_eigenvector_sym3(c11, c12, c13, c22, c23, c33)

                # A volume-filled terrain contains many interior particles.
                # Their PCA neighbourhood is almost isotropic, so its
                # smallest eigenvector is arbitrary and must not be used as a
                # surface normal. Accept either a plane-like neighbourhood
                # (also covers one-particle-thick walls) or the one-sided
                # centroid offset of an exposed volumetric surface.
                lambda_min = muladd(
                    nx * nx, c11,
                    muladd(
                        ny * ny, c22,
                        nz * nz * c33 + 2f0 * (nx * ny * c12 + nx * nz * c13 + ny * nz * c23)
                    )
                )
                covariance_trace = max(c11 + c22 + c33, 1f-12)
                pca_trace_ratio = max(lambda_min, 0f0) / covariance_trace
                mean_dx = mean_x - pos_i[1]
                mean_dy = mean_y - pos_i[2]
                mean_dz = mean_z - pos_i[3]
                mean_normal_offset = abs(muladd(nx, mean_dx, muladd(ny, mean_dy, nz * mean_dz)))
                surface_like = pca_trace_ratio <= SURFACE_PCA_TRACE_RATIO_MAX ||
                               mean_normal_offset >= SURFACE_MEAN_OFFSET_H_MIN * par.h
            end

            # PCA determines an unoriented axis: n and -n are equally valid.
            # Orient it toward nearby model-8 material first. This works for
            # curved beds, vertical walls and ceilings whenever material is
            # locally present, without relying on a particular global axis.
            orient_x, orient_y, orient_z = 0f0, 0f0, 0f0
            orient_count = Int32(0)
            orient_radius = par.normal_orientation_radius
            orient_radius2 = orient_radius * orient_radius
            inv_orient_radius = 1f0 / max(orient_radius, 1f-12)
            ocx_min = clamp(
                floor(Int32, 0.5f0 * (pos_i[1] - orient_radius - par.domain_min_x) * ih),
                Int32(0), par.cell_num_x - Int32(1)
            )
            ocy_min = clamp(
                floor(Int32, 0.5f0 * (pos_i[2] - orient_radius - par.domain_min_y) * ih),
                Int32(0), par.cell_num_y - Int32(1)
            )
            ocz_min = clamp(
                floor(Int32, 0.5f0 * (pos_i[3] - orient_radius - par.domain_min_z) * ih),
                Int32(0), par.cell_num_z - Int32(1)
            )
            ocx_max = clamp(
                floor(Int32, 0.5f0 * (pos_i[1] + orient_radius - par.domain_min_x) * ih),
                Int32(0), par.cell_num_x - Int32(1)
            )
            ocy_max = clamp(
                floor(Int32, 0.5f0 * (pos_i[2] + orient_radius - par.domain_min_y) * ih),
                Int32(0), par.cell_num_y - Int32(1)
            )
            ocz_max = clamp(
                floor(Int32, 0.5f0 * (pos_i[3] + orient_radius - par.domain_min_z) * ih),
                Int32(0), par.cell_num_z - Int32(1)
            )

            for ocz = ocz_min:ocz_max
                for ocy = ocy_min:ocy_max
                    for ocx = ocx_min:ocx_max
                        c_idx = ocx * par.cell_num_y * par.cell_num_z +
                                ocy * par.cell_num_z + ocz + par.ncell_one_layer + Int32(1)
                        p_beg = cell_beg[c_idx]
                        p_end = cell_end[c_idx]
                        if p_beg != -1
                            for j = p_beg:(p_end - 1)
                                uid_j = idep[j][1]
                                if j > par.npb && uid_j != -1f0
                                    m_idx_j = get_mat_idx(uid_j)
                                    if mat_props.c_model[m_idx_j] == Int32(8)
                                        pos_j = pospres[j]
                                        rx = pos_j[1] - pos_i[1]
                                        ry = pos_j[2] - pos_i[2]
                                        rz = pos_j[3] - pos_i[3]
                                        r2 = muladd(rx, rx, muladd(ry, ry, rz * rz))
                                        if r2 < orient_radius2
                                            u = sqrt(max(r2, 0f0)) * inv_orient_radius
                                            t = 1f0 - u
                                            t2 = t * t
                                            shape_w = t2 * t2 * (4f0 * u + 1f0)
                                            rho_j = max(velrhop[j][4], 1f-6)
                                            volume_w = shape_w * (mass_arr[j] / rho_j)
                                            orient_x += volume_w * rx
                                            orient_y += volume_w * ry
                                            orient_z += volume_w * rz
                                            orient_count += Int32(1)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            acc_norm = norm3df(par.acc_x, par.acc_y, par.acc_z)
            if acc_norm > 1f-12
                inv_acc = 1f0 / acc_norm
                up_x = -par.acc_x * inv_acc
                up_y = -par.acc_y * inv_acc
                up_z = -par.acc_z * inv_acc
            else
                up_x, up_y, up_z = 0f0, 1f0, 0f0
            end

            normal_oriented = false
            orient_len2 = muladd(orient_x, orient_x, muladd(orient_y, orient_y, orient_z * orient_z))
            if orient_count > 0 && orient_len2 > 1f-24
                orient_score = muladd(nx, orient_x, muladd(ny, orient_y, nz * orient_z))
                if abs(orient_score) > 1f-5 * sqrt(orient_len2)
                    if orient_score < 0f0
                        nx, ny, nz = -nx, -ny, -nz
                    end
                    normal_oriented = true
                end
            end

            # For dry, ordinary terrain (including strongly curved slopes),
            # orient against gravity. Near-vertical dry walls are deliberately
            # left for the material-centroid fallback below.
            up_score = muladd(nx, up_x, muladd(ny, up_y, nz * up_z))
            if !normal_oriented && abs(up_score) >= 0.2f0
                if up_score < 0f0
                    nx, ny, nz = -nx, -ny, -nz
                    up_score = -up_score
                end
                normal_oriented = true
            end

            # A dry vertical container wall has no local material cue. The
            # initial model-8 centroid supplies a stable interior-side seed.
            if !normal_oriented && model8_centroid[4] > 0f0
                gx = model8_centroid[1] - pos_i[1]
                gy = model8_centroid[2] - pos_i[2]
                gz = model8_centroid[3] - pos_i[3]
                global_len2 = muladd(gx, gx, muladd(gy, gy, gz * gz))
                global_score = muladd(nx, gx, muladd(ny, gy, nz * gz))
                if global_len2 > 1f-20 && abs(global_score) > 1f-6 * sqrt(global_len2)
                    if global_score < 0f0
                        nx, ny, nz = -nx, -ny, -nz
                    end
                    normal_oriented = true
                end
            end

            # Fully ambiguous points are rare (e.g. an exactly symmetric
            # isolated sheet). Pick a deterministic sign so neighboring runs
            # cannot alternate because of floating-point roundoff.
            if !normal_oriented
                if (abs(nx) >= abs(ny) && abs(nx) >= abs(nz) && nx < 0f0) ||
                   (abs(ny) > abs(nx) && abs(ny) >= abs(nz) && ny < 0f0) ||
                   (abs(nz) > abs(nx) && abs(nz) > abs(ny) && nz < 0f0)
                    nx, ny, nz = -nx, -ny, -nz
                end
            end

            up_score = muladd(nx, up_x, muladd(ny, up_y, nz * up_z))
            depth_radius = par.depth_footprint_radius
            inv_depth_radius = 1f0 / max(depth_radius, 1f-12)
            depth_search_max = par.depth_search_max
            support2 = depth_radius * depth_radius
            bed_gap_max = max(
                BED_MATERIAL_MAX_GAP_H_FACTOR * par.h,
                BED_MATERIAL_MAX_GAP_DR_FACTOR * par.dr
            )
            has_local_material_direction = orient_count > 0 && orient_len2 > 1f-24
            material_alignment = 1f0
            if has_local_material_direction
                material_alignment = abs(
                    muladd(nx, orient_x, muladd(ny, orient_y, nz * orient_z))
                ) / sqrt(orient_len2)
            end
            # At a sharp floor/wall intersection PCA can select the floor
            # axis for a particle that actually belongs to the wall. A valid
            # wet bed normal must also agree with the local material-side
            # direction. Dry terrain has no such cue and retains the
            # anti-gravity classification.
            bed_like = surface_like && up_score >= 0.5f0 &&
                       (!has_local_material_direction || material_alignment >= BED_MATERIAL_ALIGNMENT_MIN)

            # Find the closest material layer above the locally fitted bed and
            # the closest tangential material projection. The former defines
            # a single projected particle sheet for kernel-coverage
            # renormalization; the latter is the wet/dry criterion.
            base_h = typemax(Float32)
            nearest_tangent2 = typemax(Float32)
            if bed_like
                base_scan_radius = max(
                    orient_radius,
                    sqrt(depth_radius * depth_radius + bed_gap_max * bed_gap_max)
                )
                bcx_min = clamp(
                    floor(Int32, 0.5f0 * (pos_i[1] - base_scan_radius - par.domain_min_x) * ih),
                    Int32(0), par.cell_num_x - Int32(1)
                )
                bcy_min = clamp(
                    floor(Int32, 0.5f0 * (pos_i[2] - base_scan_radius - par.domain_min_y) * ih),
                    Int32(0), par.cell_num_y - Int32(1)
                )
                bcz_min = clamp(
                    floor(Int32, 0.5f0 * (pos_i[3] - base_scan_radius - par.domain_min_z) * ih),
                    Int32(0), par.cell_num_z - Int32(1)
                )
                bcx_max = clamp(
                    floor(Int32, 0.5f0 * (pos_i[1] + base_scan_radius - par.domain_min_x) * ih),
                    Int32(0), par.cell_num_x - Int32(1)
                )
                bcy_max = clamp(
                    floor(Int32, 0.5f0 * (pos_i[2] + base_scan_radius - par.domain_min_y) * ih),
                    Int32(0), par.cell_num_y - Int32(1)
                )
                bcz_max = clamp(
                    floor(Int32, 0.5f0 * (pos_i[3] + base_scan_radius - par.domain_min_z) * ih),
                    Int32(0), par.cell_num_z - Int32(1)
                )

                for bcz = bcz_min:bcz_max
                    for bcy = bcy_min:bcy_max
                        for bcx = bcx_min:bcx_max
                            c_idx = bcx * par.cell_num_y * par.cell_num_z +
                                    bcy * par.cell_num_z + bcz + par.ncell_one_layer + Int32(1)
                            p_beg = cell_beg[c_idx]
                            p_end = cell_end[c_idx]
                            if p_beg != -1
                                for j = p_beg:(p_end - 1)
                                    uid_j = idep[j][1]
                                    if j > par.npb && uid_j != -1f0
                                        m_idx_j = get_mat_idx(uid_j)
                                        if mat_props.c_model[m_idx_j] == Int32(8)
                                            pos_j = pospres[j]
                                            rx = pos_j[1] - pos_i[1]
                                            ry = pos_j[2] - pos_i[2]
                                            rz = pos_j[3] - pos_i[3]
                                            dist2 = muladd(rx, rx, muladd(ry, ry, rz * rz))
                                            h_proj = muladd(rx, nx, muladd(ry, ny, rz * nz))
                                            tangent2 = max(dist2 - h_proj * h_proj, 0f0)
                                            if h_proj > 0f0 && h_proj <= bed_gap_max &&
                                               tangent2 < support2 && dist2 <= base_scan_radius * base_scan_radius
                                                nearest_tangent2 = min(nearest_tangent2, tangent2)
                                                # Ignore particles that have almost crossed the
                                                # fitted bed plane. A single near-zero positive
                                                # projection must not define the basal sheet.
                                                if h_proj >= BASE_LAYER_MIN_GAP_FACTOR * par.dr
                                                    base_h = min(base_h, h_proj)
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            # Flow depth is estimated as column volume / bed area:
            #   H = sum_j (m_j / rho_j) W_2D(||s_j||)
            # where s_j is the particle offset projected onto the PCA tangent
            # plane and W_2D integrates to one. This avoids using the highest
            # particle centre as a resolution-dependent free-surface proxy.
            depth = 0f0
            basal_w = 0f0
            basal_vx, basal_vy, basal_vz = 0f0, 0f0, 0f0
            avg_vx, avg_vy, avg_vz = 0f0, 0f0, 0f0
            model8_count = Int32(0)
            depth_coverage = 0f0
            wet_column = false
            top_h = 0f0
            occupancy_0 = UInt64(0)
            occupancy_1 = UInt64(0)
            occupancy_2 = UInt64(0)
            occupancy_3 = UInt64(0)
            axial_bin_width = max(
                par.dr,
                depth_search_max / Float32(DEPTH_CONNECTIVITY_BIN_COUNT)
            )
            column_lower_h = base_h < typemax(Float32) ?
                             max(base_h - 0.75f0 * par.dr, 0f0) : 0f0

            if bed_like && base_h < typemax(Float32)
                # The old +/-1-cell loop silently capped measurable depth at
                # O(h). Bound the complete normal cylinder by a grid-aligned
                # box instead. Include the 2h basal-velocity support as well.
                basal_radius = 2f0 * par.h
                radial_x = depth_radius * sqrt(max(1f0 - nx * nx, 0f0))
                radial_y = depth_radius * sqrt(max(1f0 - ny * ny, 0f0))
                radial_z = depth_radius * sqrt(max(1f0 - nz * nz, 0f0))
                axial_x = nx * depth_search_max
                axial_y = ny * depth_search_max
                axial_z = nz * depth_search_max
                min_x = min(min(0f0, axial_x) - radial_x, -basal_radius)
                min_y = min(min(0f0, axial_y) - radial_y, -basal_radius)
                min_z = min(min(0f0, axial_z) - radial_z, -basal_radius)
                max_x = max(max(0f0, axial_x) + radial_x, basal_radius)
                max_y = max(max(0f0, axial_y) + radial_y, basal_radius)
                max_z = max(max(0f0, axial_z) + radial_z, basal_radius)
                cell_scale = 0.5f0 * ih
                neg_x = min(ceil(Int32, -min_x * cell_scale) + Int32(1), cx)
                neg_y = min(ceil(Int32, -min_y * cell_scale) + Int32(1), cy)
                neg_z = min(ceil(Int32, -min_z * cell_scale) + Int32(1), cz)
                pos_x = min(ceil(Int32, max_x * cell_scale) + Int32(1), par.cell_num_x - Int32(1) - cx)
                pos_y = min(ceil(Int32, max_y * cell_scale) + Int32(1), par.cell_num_y - Int32(1) - cy)
                pos_z = min(ceil(Int32, max_z * cell_scale) + Int32(1), par.cell_num_z - Int32(1) - cz)

                for dcz = -neg_z:pos_z
                    for dcy = -neg_y:pos_y
                        for dcx = -neg_x:pos_x
                            ncx, ncy, ncz = cx + dcx, cy + dcy, cz + dcz
                            if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                                c_idx = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz + par.ncell_one_layer + Int32(1)
                                p_beg = cell_beg[c_idx]
                                p_end = cell_end[c_idx]
                                if p_beg != -1
                                    for j = p_beg:(p_end - 1)
                                        uid_j = idep[j][1]
                                        if j > par.npb && uid_j != -1f0
                                            m_idx_j = get_mat_idx(uid_j)
                                            if mat_props.c_model[m_idx_j] == Int32(8)
                                                pos_j = pospres[j]
                                                rx = pos_j[1] - pos_i[1]
                                                ry = pos_j[2] - pos_i[2]
                                                rz = pos_j[3] - pos_i[3]
                                                dist2 = muladd(rx, rx, muladd(ry, ry, rz * rz))
                                                h_proj = muladd(rx, nx, muladd(ry, ny, rz * nz))
                                                tangent2 = max(dist2 - h_proj * h_proj, 0f0)

                                                vj = velrhop[j]
                                                if h_proj >= column_lower_h && h_proj <= depth_search_max &&
                                                   tangent2 < support2
                                                    rho_j = max(vj[4], 1f-6)
                                                    volume_j = mass_arr[j] / rho_j
                                                    w2d = kernel_wendland_2d_support(
                                                        tangent2, depth_radius, inv_depth_radius
                                                    )
                                                    depth += volume_j * w2d
                                                    top_h = max(top_h, h_proj)
                                                    if h_proj >= base_h - 0.75f0 * par.dr
                                                        axial_bin = clamp(
                                                            floor(Int32, max(h_proj - base_h, 0f0) / axial_bin_width),
                                                            Int32(0), DEPTH_CONNECTIVITY_BIN_COUNT - Int32(1)
                                                        )
                                                        axial_word = axial_bin >> 6
                                                        axial_bit = UInt64(1) << (axial_bin & Int32(63))
                                                        if axial_word == Int32(0)
                                                            occupancy_0 |= axial_bit
                                                        elseif axial_word == Int32(1)
                                                            occupancy_1 |= axial_bit
                                                        elseif axial_word == Int32(2)
                                                            occupancy_2 |= axial_bit
                                                        else
                                                            occupancy_3 |= axial_bit
                                                        end
                                                    end
                                                    # One particle layer represents the occupied
                                                    # tangent-plane area. Dividing by this discrete
                                                    # coverage removes kernel truncation at walls and
                                                    # wet-region edges while preserving volume depth.
                                                    if h_proj <= base_h + 0.75f0 * par.dr
                                                        spacing_j = cbrt(max(volume_j, 1f-30))
                                                        depth_coverage += spacing_j * spacing_j * w2d
                                                    end
                                                    avg_vx += vj[1]
                                                    avg_vy += vj[2]
                                                    avg_vz += vj[3]
                                                    model8_count += Int32(1)
                                                end

                                                q = sqrt(dist2) * ih
                                                if q < 2f0
                                                    rho_j = max(vj[4], 1f-6)
                                                    wv = kernel_val(q, par) * (mass_arr[j] / rho_j)
                                                    basal_vx += vj[1] * wv
                                                    basal_vy += vj[2] * wv
                                                    basal_vz += vj[3] * wv
                                                    basal_w += wv
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            wet_radius = WET_TANGENT_RADIUS_FACTOR * par.dr
            wet_radius2 = wet_radius * wet_radius
            wet_column = bed_like && nearest_tangent2 <= wet_radius2 &&
                         depth_coverage >= DEPTH_COVERAGE_MIN && base_h <= bed_gap_max
            if wet_column
                corrected_depth = depth / depth_coverage
                if isfinite(corrected_depth)
                    # Only the material body connected to the basal sheet may
                    # bound the local depth. This prevents a normal ray from
                    # jumping across an empty gap to a remote fold/body.
                    last_connected_bin = Int32(-1)
                    empty_run = Int32(0)
                    max_axial_bin = min(
                        ceil(Int32, depth_search_max / axial_bin_width),
                        DEPTH_CONNECTIVITY_BIN_COUNT
                    )
                    for axial_bin = Int32(0):(max_axial_bin - Int32(1))
                        axial_word = axial_bin >> 6
                        axial_bit = UInt64(1) << (axial_bin & Int32(63))
                        occupied = if axial_word == Int32(0)
                            (occupancy_0 & axial_bit) != UInt64(0)
                        elseif axial_word == Int32(1)
                            (occupancy_1 & axial_bit) != UInt64(0)
                        elseif axial_word == Int32(2)
                            (occupancy_2 & axial_bit) != UInt64(0)
                        else
                            (occupancy_3 & axial_bit) != UInt64(0)
                        end
                        if occupied
                            last_connected_bin = axial_bin
                            empty_run = Int32(0)
                        else
                            empty_run += Int32(1)
                            if empty_run >= Int32(2)
                                break
                            end
                        end
                    end

                    if last_connected_bin >= Int32(0)
                        particle_span_depth = max(top_h - base_h + par.dr, par.dr)
                        connected_span_depth =
                            Float32(last_connected_bin + Int32(1)) * axial_bin_width
                        geometric_depth_max = min(
                            particle_span_depth,
                            connected_span_depth + par.dr
                        )
                        depth = clamp(corrected_depth, 0f0, min(depth_search_max, geometric_depth_max))
                    else
                        wet_column = false
                    end
                else
                    wet_column = false
                end
            end
            if !wet_column
                depth = 0f0
                basal_w = 0f0
                basal_vx, basal_vy, basal_vz = 0f0, 0f0, 0f0
                avg_vx, avg_vy, avg_vz = 0f0, 0f0, 0f0
                model8_count = Int32(0)
            end

            if basal_w > 1f-12
                inv_w = 1f0 / basal_w
                basal_vx *= inv_w
                basal_vy *= inv_w
                basal_vz *= inv_w
            end

            dot_bn = muladd(basal_vx, nx, muladd(basal_vy, ny, basal_vz * nz))
            tan_vx = basal_vx - dot_bn * nx
            tan_vy = basal_vy - dot_bn * ny
            tan_vz = basal_vz - dot_bn * nz
            basal_speed = norm3df(tan_vx, tan_vy, tan_vz)

            avg_speed = 0f0
            if model8_count > 0
                inv_m8 = 1f0 / Float32(model8_count)
                avg_vx *= inv_m8
                avg_vy *= inv_m8
                avg_vz *= inv_m8
                dot_an = muladd(avg_vx, nx, muladd(avg_vy, ny, avg_vz * nz))
                avg_tx = avg_vx - dot_an * nx
                avg_ty = avg_vy - dot_an * ny
                avg_tz = avg_vz - dot_an * nz
                avg_speed = norm3df(avg_tx, avg_ty, avg_tz)
            end

            slope = 0f0
            if surface_like
                ny_abs = abs(ny)
                vxz = sqrt(tan_vx * tan_vx + tan_vz * tan_vz)
                if ny_abs > 1f-6 && vxz > 1f-12
                    dydx = -nx / ny
                    dydz = -nz / ny
                    slope = atan(abs(dydx * (tan_vx / vxz) + dydz * (tan_vz / vxz)))
                elseif ny_abs > 1f-6
                    slope = atan(sqrt(nx * nx + nz * nz) / ny_abs)
                end
            end

            extraction_mask = (wet_column && model8_count > 0) ? 1f0 : 0f0
            topo_normal[i] = surface_like ?
                Float4(nx, ny, nz, Float32(model8_count)) :
                Float4(0f0, 0f0, 0f0, 0f0)
            topo_quantities[i] = Float4(depth, basal_speed, avg_speed, slope)
            topo_basal_velocity[i] = Float4(tan_vx, tan_vy, tan_vz, extraction_mask)
        end
    end
    return nothing
end

@inline function entrainment_random01(idx::Int32, time_step_key::Int32)
    # A small, deterministic integer hash gives the paper's random B_d
    # perturbation without making results depend on GPU launch ordering.
    x = UInt32(idx) + UInt32(time_step_key + Int32(1)) * UInt32(0x9e3779b9)
    x = xor(x, x >> 16) * UInt32(0x7feb352d)
    x = xor(x, x >> 15) * UInt32(0x846ca68b)
    x = xor(x, x >> 16)
    return Float32(x & UInt32(0x00ffffff)) / 16777215f0
end

function kernel_update_model8_entrainment(
    topo_entrainment,
    topo_previous_depth,
    topo_quantities,
    topo_basal_velocity,
    idep,
    initialize::Int32,
    par::Parameters
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npb
        @inbounds begin
            q = topo_quantities[idx]
            basal = topo_basal_velocity[idx]
            depth = isfinite(q[1]) ? max(q[1], 0f0) : 0f0
            previous_depth = topo_previous_depth[idx]
            old = topo_entrainment[idx]
            erosion_depth = isfinite(old[2]) ? max(old[2], 0f0) : 0f0
            base_bd = clamp(par.entrainment_pore_pressure, 0f0, 1f0)

            if initialize != 0
                topo_previous_depth[idx] = depth
                topo_entrainment[idx] = Float4(0f0, erosion_depth, base_bd, 0f0)
                return nothing
            end

            # Use the robust volume/area depth change rather than the paper's
            # maximum-particle elevation prediction. This keeps load/unload
            # classification insensitive to isolated splashing particles.
            depth_tolerance = max(0.01f0 * par.dr, 1f-6)
            depth_change = depth - previous_depth
            wet = basal[4] > 0.5f0 && depth > 0f0
            # Eq. (18) classifies dh/dt == 0 as loading. The tolerance only
            # prevents small depth-estimator noise from causing unloading.
            load_status = wet ? (depth_change < -depth_tolerance ? -1f0 : 1f0) : 0f0

            bd_effective = old[3]
            if !isfinite(bd_effective) || bd_effective < 0f0 || bd_effective > 1f0
                bd_effective = base_bd
            end
            if load_status != 0f0
                time_step_key = floor(Int32, max(par.time, 0f0) / max(par.dt, 1f-12))
                particle_key = Int32(round(idep[idx][3]))
                random01 = entrainment_random01(particle_key, time_step_key)
                if load_status > 0f0
                    bd_low = base_bd
                    bd_high = min(1f0, 1.2f0 * base_bd)
                else
                    bd_low = max(0f0, 0.8f0 * base_bd)
                    bd_high = base_bd
                end
                bd_effective = muladd(random01, bd_high - bd_low, bd_low)
            end

            erosion_rate = 0f0
            can_erode = par.entrainment_enabled != 0 &&
                        par.erosion_depth_max > 0f0 &&
                        erosion_depth < par.erosion_depth_max && wet
            basal_speed = isfinite(q[2]) ? max(q[2], 0f0) : 0f0
            if can_erode && basal_speed >= par.entrainment_speed_min
                gravity = sqrt(par.acc_x * par.acc_x + par.acc_y * par.acc_y + par.acc_z * par.acc_z)
                slope = isfinite(q[4]) ? clamp(q[4], 0f0, 0.5f0 * PCA_EIGEN_PI) : 0f0
                driving_margin = tan(par.entrainment_flow_friction_angle) -
                                 (1f0 - bd_effective) * tan(par.entrainment_bed_friction_angle)

                # Eq. (15), with G represented by the acceleration magnitude.
                # A non-positive shear margin means that the bed resists
                # entrainment; this implementation does not model deposition.
                if gravity > 0f0 && driving_margin > 0f0
                    erosion_rate = gravity * depth * cos(slope) * driving_margin / basal_speed
                    if par.entrainment_rate_max > 0f0
                        erosion_rate = min(erosion_rate, par.entrainment_rate_max)
                    end
                    if par.dt > 0f0
                        erosion_rate = min(erosion_rate, (par.erosion_depth_max - erosion_depth) / par.dt)
                    else
                        erosion_rate = 0f0
                    end
                    if !isfinite(erosion_rate) || erosion_rate < 0f0
                        erosion_rate = 0f0
                    end
                end
            end

            erosion_depth = min(par.erosion_depth_max, erosion_depth + erosion_rate * max(par.dt, 0f0))
            topo_previous_depth[idx] = depth
            topo_entrainment[idx] = Float4(erosion_rate, erosion_depth, bd_effective, load_status)
        end
    end
    return nothing
end

function kernel_add_external_acceleration(accad, cpl_acc, npb, npm)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= npm
        i = idx + npb
        @inbounds begin
            a = accad[i]
            c = cpl_acc[i]
            accad[i] = Float4(a[1] + c[1], a[2] + c[2], a[3] + c[3], a[4])
        end
    end
    return nothing
end


function kernel_stress_regularization(pospres, velrhop, str, str_cache, cell_beg, cell_end, par::Parameters)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npm
        i = idx + par.npb
        @inbounds begin
            pos_i = pospres[i]
            if !isfinite(pos_i[1]) return nothing end
            sxx, syy, szz = 0f0, 0f0, 0f0; sxy, sxz, syz = 0f0, 0f0, 0f0
            sumker = 0f0
            ih = par.i_h
            
            cx = floor(Int32, 0.5f0 * (pos_i[1] - par.domain_min_x) * ih)
            cy = floor(Int32, 0.5f0 * (pos_i[2] - par.domain_min_y) * ih)
            cz = floor(Int32, 0.5f0 * (pos_i[3] - par.domain_min_z) * ih)
            
            k_norm = par.kernel_normalization_par
            mass = par.m
            
            for dcz = Int32(-1):Int32(1)
                for dcy = Int32(-1):Int32(1)
                    for dcx = Int32(-1):Int32(1)
                        ncx, ncy, ncz = cx+dcx, cy+dcy, cz+dcz
                        if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                            c_idx = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz + par.ncell_one_layer + 1
                            p_beg = cell_beg[c_idx]; p_end = cell_end[c_idx]
                            if p_beg != -1
                                for j = p_beg:(p_end-1)
                                    pos_j = pospres[j]
                                    dr_x = pos_i[1]-pos_j[1]; dr_y = pos_i[2]-pos_j[2]; dr_z = pos_i[3]-pos_j[3]
                                    dist2 = muladd(dr_x, dr_x, muladd(dr_y, dr_y, dr_z*dr_z))
                                    if dist2 * ih * ih < 4f0
                                        dist = sqrt(dist2); q = dist * ih
                                        kv = kernel_val(q, par) * (mass / velrhop[j][4])
                                        sj = str[j]
                                        sxx += sj.xx * kv; syy += sj.yy * kv; szz += sj.zz * kv
                                        sxy += sj.xy * kv; sxz += sj.xz * kv; syz += sj.yz * kv
                                        sumker += kv
                                    end
                                end
                            end
                        end
                    end
                end
            end
            if sumker > 1f-6
                inv = 1f0/sumker
                str_cache[i] = Tensor2D(sxx*inv, syy*inv, szz*inv, sxy*inv, sxz*inv, syz*inv)
            else
                str_cache[i] = str[i]
            end
        end
    end
    return nothing
end


function kernel_kill_high_velocity_particles(velrhop, idep, limit_sq, npb, npm)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= npm
        i = idx + npb
        @inbounds begin
            v = velrhop[i]
            v2 = v[1]*v[1] + v[2]*v[2] + v[3]*v[3]
            
            if v2 > limit_sq
                mat_id = idep[i][1]
                if mat_id != -1f0
                    idep[i] = Float4(-1f0, idep[i][2], idep[i][3], idep[i][4])
                    velrhop[i] = Float4(0f0, 0f0, 0f0, v[4])
                end
            end
        end
    end
    return nothing
end

end # module

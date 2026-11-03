module TimeIntegration

using CUDA
using StaticArrays
using ..Types: Parameters, Float4, Tensor2D, J2, trace, MaterialProperties
using ..Physics: get_mat_idx

export kernel_predictor, kernel_corrector
export kernel_leapfrog_kickstart, kernel_leapfrog_kinematics, kernel_leapfrog_stress

const SQRT_2_DIV_3 = sqrt(2f0 / 3f0)

@inline function is_solid_model(c_model::Int32)
    return c_model == 1 || c_model == 2 || c_model == 3 ||
           c_model == 4 || c_model == 5 || c_model == 6 ||
           c_model == 7 || c_model == 8
end


@inline @fastmath function stabilize_solid_density(rho_trial::Float32, rho0::Float32)
    # Geomaterial runs are typically quasi-incompressible in SPH.
    # Keep density variation bounded to avoid non-physical large oscillations.
    rho_min = 0.1f0 * rho0
    rho_max = 3f0 * rho0
    return clamp(rho_trial, rho_min, rho_max)
end

@inline @fastmath function mat_mod_dp!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref, K_val, G_val, kphi, kpsi, kcohesion)
    p = p_ref[]
    ep = ep_ref[]
    epd = epd_ref[]
    sigma = sigma_ref[]

    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

    ds_xx = muladd(-2f0, muladd(w_xy, s_xy, w_xz * s_xz), 0f0)
    ds_yy = muladd(2f0, muladd(w_xy, s_xy, -(w_yz * s_yz)), 0f0)
    ds_zz = muladd(2f0, muladd(w_xz, s_xz, w_yz * s_yz), 0f0)
    ds_xy = muladd(w_xy, (s_xx - s_yy), muladd(-w_yz, s_xz, -(w_xz * s_yz)))
    ds_xz = muladd(w_xz, (s_xx - s_zz), muladd(w_yz, s_xy, -(w_xy * s_yz)))
    ds_yz = muladd(w_yz, (s_yy - s_zz), muladd(w_xy, s_xz, w_xz * s_xy))
    
    sigma = sigma + Tensor2D(ds_xx, ds_yy, ds_zz, ds_xy, ds_xz, ds_yz) * dt
    trace_d = trace(d)
    
    sigma = sigma + d * (2f0 * G_val * dt)
    val_ace = (K_val - 2f0 / 3f0 * G_val) * trace_d * dt
    sigma = Tensor2D(sigma.xx + val_ace, sigma.yy + val_ace, sigma.zz + val_ace, sigma.xy, sigma.xz, sigma.yz)
    
    sm = trace(sigma) / 3f0
    sigma_dev = Tensor2D(sigma.xx - sm, sigma.yy - sm, sigma.zz - sm, sigma.xy, sigma.xz, sigma.yz)
    
    tenf = (kphi == 0f0) ? 0f0 : (kcohesion / kphi)
    
    dpTi = sm - tenf
    if dpTi >= 0f0
        dlambda = dpTi / K_val 
        sm = tenf
        ep += dlambda * (1f0 / 3f0) * sqrt(2f0)
    end
    
    tau = sqrt(0.5f0 * J2(sigma_dev))
    dpFi = tau + kphi * sm - kcohesion
    
    if dpFi > 0f0
        denom = G_val + K_val * kphi * kpsi
        dlambda = dpFi / denom
        
        sm -= K_val * kpsi * dlambda
        ratio = (tau < 1f-10) ? 1f0 : (kcohesion - kphi * sm) / tau
        sigma_dev = sigma_dev * ratio
        ep += dlambda * sqrt(1f0/3f0 + 2f0/9f0 * kpsi * kpsi)
        epd += dlambda * sqrt(1f0/3f0)
    end

    sigma = Tensor2D(sigma_dev.xx + sm, sigma_dev.yy + sm, sigma_dev.zz + sm, sigma_dev.xy, sigma_dev.xz, sigma_dev.yz)
    p = -sm
    p_ref[] = p
    ep_ref[] = ep
    epd_ref[] = epd
    sigma_ref[] = sigma
    return nothing
end

@inline @fastmath function mat_mod_hypo!(dt, d, omega, sigma_ref, K_val, G_val)
    sigma = sigma_ref[]
    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

    ds_xx = muladd(-2f0, muladd(w_xy, s_xy, w_xz * s_xz), 0f0)
    ds_yy = muladd(2f0, muladd(w_xy, s_xy, -(w_yz * s_yz)), 0f0)
    ds_zz = muladd(2f0, muladd(w_xz, s_xz, w_yz * s_yz), 0f0)
    ds_xy = muladd(w_xy, (s_xx - s_yy), muladd(-w_yz, s_xz, -(w_xz * s_yz)))
    ds_xz = muladd(w_xz, (s_xx - s_zz), muladd(w_yz, s_xy, -(w_xy * s_yz)))
    ds_yz = muladd(w_yz, (s_yy - s_zz), muladd(w_xy, s_xz, w_xz * s_xy))
    
    sigma = sigma + Tensor2D(ds_xx, ds_yy, ds_zz, ds_xy, ds_xz, ds_yz) * dt

    lambda = K_val - 0.6666667f0 * G_val 
    two_G_dt = 2f0 * G_val * dt              
    term_diag = (lambda * trace(d)) * dt 

    dsigma_xx = muladd(two_G_dt, d.xx, term_diag)
    dsigma_yy = muladd(two_G_dt, d.yy, term_diag)
    dsigma_zz = muladd(two_G_dt, d.zz, term_diag)
    dsigma_xy = two_G_dt * d.xy
    dsigma_xz = two_G_dt * d.xz
    dsigma_yz = two_G_dt * d.yz
    
    sigma_new = sigma + Tensor2D(dsigma_xx, dsigma_yy, dsigma_zz, dsigma_xy, dsigma_xz, dsigma_yz)
    sigma_ref[] = sigma_new
    return nothing
end

@inline @fastmath function mat_mod_elastic!(dt, d, omega, sigma_ref, p_ref, K_val, G_val)
    sigma = sigma_ref[]
    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

    ds_rot_xx = -2f0 * (s_xy * w_xy + s_xz * w_xz)
    ds_rot_yy =  2f0 * (s_xy * w_xy - s_yz * w_yz)
    ds_rot_zz =  2f0 * (s_xz * w_xz + s_yz * w_yz)
    ds_rot_xy = (s_yy - s_xx) * w_xy + s_xz * w_yz - s_yz * w_xz
    ds_rot_xz = (s_zz - s_xx) * w_xz - s_xy * w_yz + s_yz * w_xy
    ds_rot_yz = (s_zz - s_yy) * w_yz + s_xy * w_xz - s_xz * w_xy
    
    sigma = sigma + Tensor2D(ds_rot_xx, ds_rot_yy, ds_rot_zz, ds_rot_xy, ds_rot_xz, ds_rot_yz) * dt

    lambda = K_val - 0.6666667f0 * G_val 
    two_G_dt = 2f0 * G_val * dt              
    term_diag = (lambda * trace(d)) * dt 

    dsigma_xx = muladd(two_G_dt, d.xx, term_diag)
    dsigma_yy = muladd(two_G_dt, d.yy, term_diag)
    dsigma_zz = muladd(two_G_dt, d.zz, term_diag)
    dsigma_xy = two_G_dt * d.xy
    dsigma_xz = two_G_dt * d.xz
    dsigma_yz = two_G_dt * d.yz
  
    sigma_new = sigma + Tensor2D(dsigma_xx, dsigma_yy, dsigma_zz, dsigma_xy, dsigma_xz, dsigma_yz)
    sigma_ref[] = sigma_new
    p_ref[] = -(sigma_new.xx + sigma_new.yy + sigma_new.zz) * 0.3333333f0
    return nothing
end

@inline @fastmath function mat_mod_dp_fluid_solid!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                                   K_val, G_val, kcohesion, kpsi,
                                                   d_p, rho_s, mu_s, mu_2, I_0)
    ep = ep_ref[]
    epd = epd_ref[]
    sigma = sigma_ref[]

   
    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

   
    ds_xx = muladd(-2f0, muladd(w_xy, s_xy, w_xz * s_xz), 0f0)
    ds_yy = muladd(2f0, muladd(w_xy, s_xy, -(w_yz * s_yz)), 0f0)
    ds_zz = muladd(2f0, muladd(w_xz, s_xz, w_yz * s_yz), 0f0)
    ds_xy = muladd(w_xy, (s_xx - s_yy), muladd(-w_yz, s_xz, -(w_xz * s_yz)))
    ds_xz = muladd(w_xz, (s_xx - s_zz), muladd(w_yz, s_xy, -(w_xy * s_yz)))
    ds_yz = muladd(w_yz, (s_yy - s_zz), muladd(w_xy, s_xz, w_xz * s_xy))
    
    sigma = sigma + Tensor2D(ds_xx, ds_yy, ds_zz, ds_xy, ds_xz, ds_yz) * dt
    trace_d = trace(d)
    
   
    sigma = sigma + d * (2f0 * G_val * dt)
    val_ace = (K_val - 2f0/3f0 * G_val) * trace_d * dt
    sigma = Tensor2D(sigma.xx + val_ace, sigma.yy + val_ace, sigma.zz + val_ace,
                     sigma.xy, sigma.xz, sigma.yz)
    
  
    # Trial stress from Jaumann + elastic predictor.
    sm_trial = trace(sigma) / 3f0
    p_trial = -sm_trial 
    
    s_dev_trial = Tensor2D(sigma.xx - sm_trial, sigma.yy - sm_trial, sigma.zz - sm_trial,
                           sigma.xy, sigma.xz, sigma.yz)
    tau_trial = sqrt(max(0.5f0 * J2(s_dev_trial), 0f0))

    p_t = (mu_s <= 1f-12) ? 0f0 : (kcohesion / mu_s)
    
    # Stage 1: tensile cut-off projected to tensile apex (avoid stress chatter).
    if p_trial < -p_t
        # For cohesive material, tensile cut-off contributes volumetric plastic strain.
        if p_t > 0f0 && K_val > 1f-12
            ep += (-p_t - p_trial) / K_val * (sqrt(2f0) / 3f0)
        end
        sm_apex = p_t
        p_ref[] = -sm_apex
        ep_ref[] = ep
        epd_ref[] = epd
        sigma_ref[] = Tensor2D(sm_apex, sm_apex, sm_apex, 0f0, 0f0, 0f0)
        return nothing
    end

 
    d_dev_xx = d.xx - trace_d / 3f0
    d_dev_yy = d.yy - trace_d / 3f0
    d_dev_zz = d.zz - trace_d / 3f0
    
    j2_d = 0.5f0 * (d_dev_xx^2 + d_dev_yy^2 + d_dev_zz^2) + d.xy^2 + d.xz^2 + d.yz^2
    gamma_dot = 2f0 * sqrt(max(j2_d, 0f0))
    
 
    p_val = max(p_trial, 1f-6)
    inertial_denom = sqrt(p_val / rho_s)
    I_num = (inertial_denom > 1f-12) ? (gamma_dot * d_p / inertial_denom) : 0f0
    
 
    mu_denom = I_0 + I_num
    kphi_dyn = (mu_denom > 1f-12) ? (mu_s + (mu_2 - mu_s) * I_num / mu_denom) : mu_s

    
    f_s = tau_trial - (mu_s * p_val + kcohesion)     
    f_d = tau_trial - (kphi_dyn * p_val + kcohesion)  

  
    # Stage 2: elastic state.
    if f_s <= 0f0
        sm_final = sm_trial
        s_dev_final = s_dev_trial

    # Stage 3: micro-viscoplastic state.
    elseif f_d <= 0f0
        tau_visco = kphi_dyn * p_val + kcohesion
        
        if gamma_dot > 1f-12
            factor = 2f0 * tau_visco / gamma_dot
            s_dev_final = Tensor2D(factor * d_dev_xx, factor * d_dev_yy, factor * d_dev_zz,
                                   factor * d.xy, factor * d.xz, factor * d.yz)
        else
            s_dev_final = Tensor2D(0f0)
        end
        sm_final = sm_trial

    # Stage 4: complete viscoplastic state.
    else
        denom = G_val + K_val * kphi_dyn * kpsi
        dlambda = (denom > 1f-12) ? (f_d / denom) : 0f0
        
        sm_final = sm_trial - K_val * kpsi * dlambda
        
        tau_new = tau_trial - G_val * dlambda
        if tau_new <= 1f-12 || tau_trial <= 1f-12
            s_dev_final = Tensor2D(0f0)
        else
            ratio = tau_new / tau_trial
            s_dev_final = s_dev_trial * ratio
        end
        
    
        ep += dlambda * sqrt(1f0/3f0 + 2f0/9f0 * kpsi^2)
        epd += dlambda * sqrt(1f0/3f0)
    end

   
    sigma_final = Tensor2D(s_dev_final.xx + sm_final, s_dev_final.yy + sm_final, s_dev_final.zz + sm_final,
                           s_dev_final.xy, s_dev_final.xz, s_dev_final.yz)
    
    p_ref[] = -sm_final
    ep_ref[] = ep
    epd_ref[] = epd
    sigma_ref[] = sigma_final
    
    return nothing
end

@inline @fastmath function regularized_mu_i(eps_norm::Float32, p_val::Float32,
                                            d_p::Float32, rho_s::Float32,
                                            mu_s::Float32, mu_d::Float32,
                                            I_0::Float32, lambda_reg::Float32)
    p_safe = max(p_val, 0f0)
    p_rate = (rho_s > 1f-12 && d_p > 1f-12) ? sqrt(p_safe / (rho_s * d_p * d_p)) : 0f0
    denom = I_0 * p_rate + sqrt(eps_norm * eps_norm + lambda_reg * lambda_reg)
    return (denom > 1f-12) ? (mu_s + (mu_d - mu_s) * eps_norm / denom) : mu_s
end

@inline @fastmath function dp_k_from_mu(mu::Float32)
    denom = sqrt(9f0 + 12f0 * mu * mu)
    return (denom > 1f-12) ? (3f0 * mu / denom) : 0f0
end

@inline @fastmath function dp_kc_from_mu(cohesion::Float32, mu::Float32)
    denom = sqrt(9f0 + 12f0 * mu * mu)
    return (denom > 1f-12) ? (3f0 * cohesion / denom) : 0f0
end

@inline @fastmath function mat_mod_regularized_mu_i_dp!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                                        K_val, G_val, cohesion, psi,
                                                        d_p, rho_s, mu_s, mu_d, I_0, lambda_reg)
    ep = ep_ref[]
    epd = epd_ref[]
    sigma = sigma_ref[]

    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

    ds_xx = muladd(-2f0, muladd(w_xy, s_xy, w_xz * s_xz), 0f0)
    ds_yy = muladd(2f0, muladd(w_xy, s_xy, -(w_yz * s_yz)), 0f0)
    ds_zz = muladd(2f0, muladd(w_xz, s_xz, w_yz * s_yz), 0f0)
    ds_xy = muladd(w_xy, (s_xx - s_yy), muladd(-w_yz, s_xz, -(w_xz * s_yz)))
    ds_xz = muladd(w_xz, (s_xx - s_zz), muladd(w_yz, s_xy, -(w_xy * s_yz)))
    ds_yz = muladd(w_yz, (s_yy - s_zz), muladd(w_xy, s_xz, w_xz * s_xy))

    trace_d = trace(d)
    one_third_trace_d = trace_d / 3f0
    d_dev = Tensor2D(d.xx - one_third_trace_d, d.yy - one_third_trace_d, d.zz - one_third_trace_d,
                     d.xy, d.xz, d.yz)

    sigma = sigma + Tensor2D(ds_xx, ds_yy, ds_zz, ds_xy, ds_xz, ds_yz) * dt
    sigma = sigma + d_dev * (2f0 * G_val * dt)
    val_ace = K_val * trace_d * dt
    sigma = Tensor2D(sigma.xx + val_ace, sigma.yy + val_ace, sigma.zz + val_ace,
                     sigma.xy, sigma.xz, sigma.yz)

    sm_trial = trace(sigma) / 3f0
    p_trial = -sm_trial
    s_dev_trial = Tensor2D(sigma.xx - sm_trial, sigma.yy - sm_trial, sigma.zz - sm_trial,
                           sigma.xy, sigma.xz, sigma.yz)
    sqrt_j2_trial = sqrt(max(0.5f0 * J2(s_dev_trial), 0f0))

    eps_norm = sqrt(max(2f0 * J2(d), 0f0))
    mu_val = regularized_mu_i(eps_norm, max(p_trial, 0f0), d_p, rho_s, mu_s, mu_d, I_0, lambda_reg)
    k_mu = dp_k_from_mu(mu_val)
    k_c = dp_kc_from_mu(cohesion, mu_val)
    k_psi = 9f0 * sin(psi)

    p_t = (k_mu > 1f-12) ? (k_c / k_mu) : 0f0
    if p_trial < -p_t
        if p_t > 0f0 && K_val > 1f-12
            ep += (-p_t - p_trial) / K_val * (sqrt(2f0) / 3f0)
        end
        sm_apex = p_t
        p_ref[] = -sm_apex
        ep_ref[] = ep
        epd_ref[] = epd
        sigma_ref[] = Tensor2D(sm_apex, sm_apex, sm_apex, 0f0, 0f0, 0f0)
        return nothing
    end

    f_trial = sqrt_j2_trial - (k_mu * max(p_trial, 0f0) + k_c)
    sm_final = sm_trial
    s_dev_final = s_dev_trial

    if f_trial > 0f0
        denom = G_val + K_val * k_mu * k_psi
        dlambda = (denom > 1f-12) ? max(f_trial / denom, 0f0) : 0f0
        p_final = p_trial + K_val * k_psi * dlambda
        sm_final = -p_final

        sqrt_j2_final = sqrt_j2_trial - G_val * dlambda
        if sqrt_j2_final <= 1f-12 || sqrt_j2_trial <= 1f-12
            s_dev_final = Tensor2D(0f0)
        else
            s_dev_final = s_dev_trial * (sqrt_j2_final / sqrt_j2_trial)
        end

        tau_yield = k_mu * max(p_final, 0f0) + k_c
        tau_curr = sqrt(max(0.5f0 * J2(s_dev_final), 0f0))
        if tau_curr > tau_yield && tau_curr > 1f-12
            s_dev_final = s_dev_final * (tau_yield / tau_curr)
        end

        ep += dlambda * sqrt(1f0 / 3f0 + 2f0 / 9f0 * k_psi * k_psi)
        epd += dlambda * sqrt(1f0 / 3f0)
    end

    sigma_final = Tensor2D(s_dev_final.xx + sm_final, s_dev_final.yy + sm_final, s_dev_final.zz + sm_final,
                           s_dev_final.xy, s_dev_final.xz, s_dev_final.yz)
    p_ref[] = -sm_final
    ep_ref[] = ep
    epd_ref[] = epd
    sigma_ref[] = sigma_final
    return nothing
end

@inline @fastmath function mat_mod_dp_bui2008!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                               K_val, G_val, kphi, kpsi, kcohesion)
    ep = ep_ref[]
    epd = epd_ref[]
    sigma = sigma_ref[]

    # 1) Jaumann rotation + elastic predictor
    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

    ds_xx = muladd(-2f0, muladd(w_xy, s_xy, w_xz * s_xz), 0f0)
    ds_yy = muladd(2f0, muladd(w_xy, s_xy, -(w_yz * s_yz)), 0f0)
    ds_zz = muladd(2f0, muladd(w_xz, s_xz, w_yz * s_yz), 0f0)
    ds_xy = muladd(w_xy, (s_xx - s_yy), muladd(-w_yz, s_xz, -(w_xz * s_yz)))
    ds_xz = muladd(w_xz, (s_xx - s_zz), muladd(w_yz, s_xy, -(w_xy * s_yz)))
    ds_yz = muladd(w_yz, (s_yy - s_zz), muladd(w_xy, s_xz, w_xz * s_xy))

    sigma = sigma + Tensor2D(ds_xx, ds_yy, ds_zz, ds_xy, ds_xz, ds_yz) * dt
    trace_d = trace(d)
    sigma = sigma + d * (2f0 * G_val * dt)
    val_ace = (K_val - 2f0 / 3f0 * G_val) * trace_d * dt
    sigma = Tensor2D(sigma.xx + val_ace, sigma.yy + val_ace, sigma.zz + val_ace,
                     sigma.xy, sigma.xz, sigma.yz)

    sm_trial = trace(sigma) / 3f0
    p_trial = -sm_trial
    s_dev_trial = Tensor2D(sigma.xx - sm_trial, sigma.yy - sm_trial, sigma.zz - sm_trial,
                           sigma.xy, sigma.xz, sigma.yz)
    tau_trial = sqrt(max(0.5f0 * J2(s_dev_trial), 0f0))

    # 2) Tensile cut-off with fixed friction coefficient kphi
    p_t = (kphi <= 1f-12) ? 0f0 : (kcohesion / kphi)
    if p_trial < -p_t
        if p_t > 0f0 && K_val > 1f-12
            ep += (-p_t - p_trial) / K_val * (sqrt(2f0) / 3f0)
        end
        sm_apex = p_t
        p_ref[] = -sm_apex
        ep_ref[] = ep
        epd_ref[] = epd
        sigma_ref[] = Tensor2D(sm_apex, sm_apex, sm_apex, 0f0, 0f0, 0f0)
        return nothing
    end

    # 3) Bui (2008) strain-rate driven plastic multiplier (rate-independent DP)
    f_trial = tau_trial - (kphi * max(p_trial, 0f0) + kcohesion)
    sm_final = sm_trial
    s_dev_final = s_dev_trial
    if f_trial > 0f0
        d_dev_xx = d.xx - trace_d / 3f0
        d_dev_yy = d.yy - trace_d / 3f0
        d_dev_zz = d.zz - trace_d / 3f0
        s_dot_d = s_dev_trial.xx * d_dev_xx + s_dev_trial.yy * d_dev_yy + s_dev_trial.zz * d_dev_zz +
                  2f0 * (s_dev_trial.xy * d.xy + s_dev_trial.xz * d.xz + s_dev_trial.yz * d.yz)

        tau_safe = max(tau_trial, 1f-12)
        numerator = 3f0 * kphi * K_val * trace_d + G_val * (s_dot_d / tau_safe)
        denom = G_val + 9f0 * K_val * kphi * kpsi
        lambda_dot = (denom > 1f-12) ? max(numerator / denom, 0f0) : 0f0
        dlambda = lambda_dot * dt

        sm_final = sm_trial - K_val * kpsi * dlambda
        tau_new = tau_trial - G_val * dlambda
        if tau_new <= 1f-12 || tau_trial <= 1f-12
            s_dev_final = Tensor2D(0f0)
        else
            ratio = tau_new / tau_trial
            s_dev_final = s_dev_trial * ratio
        end

        # keep stress on/inside current yield surface
        p_new = max(-sm_final, 0f0)
        tau_yield = kphi * p_new + kcohesion
        tau_curr = sqrt(max(0.5f0 * J2(s_dev_final), 0f0))
        if tau_curr > tau_yield && tau_curr > 1f-12
            s_dev_final = s_dev_final * (tau_yield / tau_curr)
        end

        ep += dlambda * sqrt(1f0 / 3f0 + 2f0 / 9f0 * kpsi^2)
        epd += dlambda * sqrt(1f0 / 3f0)
    end

    sigma_final = Tensor2D(s_dev_final.xx + sm_final, s_dev_final.yy + sm_final, s_dev_final.zz + sm_final,
                           s_dev_final.xy, s_dev_final.xz, s_dev_final.yz)
    p_ref[] = -sm_final
    ep_ref[] = ep
    epd_ref[] = epd
    sigma_ref[] = sigma_final
    return nothing
end

@inline @fastmath function mat_mod_dp_fluid_solid_bui2008!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                                            K_val, G_val, kphi, kpsi, kcohesion,
                                                            d_p, rho_s, mu_s, mu_2, I_0)
    ep = ep_ref[]
    epd = epd_ref[]
    sigma = sigma_ref[]

    # 1) Jaumann rotation + elastic predictor
    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

    ds_xx = muladd(-2f0, muladd(w_xy, s_xy, w_xz * s_xz), 0f0)
    ds_yy = muladd(2f0, muladd(w_xy, s_xy, -(w_yz * s_yz)), 0f0)
    ds_zz = muladd(2f0, muladd(w_xz, s_xz, w_yz * s_yz), 0f0)
    ds_xy = muladd(w_xy, (s_xx - s_yy), muladd(-w_yz, s_xz, -(w_xz * s_yz)))
    ds_xz = muladd(w_xz, (s_xx - s_zz), muladd(w_yz, s_xy, -(w_xy * s_yz)))
    ds_yz = muladd(w_yz, (s_yy - s_zz), muladd(w_xy, s_xz, w_xz * s_xy))

    sigma = sigma + Tensor2D(ds_xx, ds_yy, ds_zz, ds_xy, ds_xz, ds_yz) * dt
    trace_d = trace(d)
    sigma = sigma + d * (2f0 * G_val * dt)
    val_ace = (K_val - 2f0 / 3f0 * G_val) * trace_d * dt
    sigma = Tensor2D(sigma.xx + val_ace, sigma.yy + val_ace, sigma.zz + val_ace,
                     sigma.xy, sigma.xz, sigma.yz)

    sm_trial = trace(sigma) / 3f0
    p_trial = -sm_trial
    s_dev_trial = Tensor2D(sigma.xx - sm_trial, sigma.yy - sm_trial, sigma.zz - sm_trial,
                           sigma.xy, sigma.xz, sigma.yz)
    tau_trial = sqrt(max(0.5f0 * J2(s_dev_trial), 0f0))

    # 2) Tensile cut-off uses DP friction coefficient (kphi from XML).
    p_t = (kphi <= 1f-12) ? 0f0 : (kcohesion / kphi)
    if p_trial < -p_t
        if p_t > 0f0 && K_val > 1f-12
            ep += (-p_t - p_trial) / K_val * (sqrt(2f0) / 3f0)
        end
        sm_apex = p_t
        p_ref[] = -sm_apex
        ep_ref[] = ep
        epd_ref[] = epd
        sigma_ref[] = Tensor2D(sm_apex, sm_apex, sm_apex, 0f0, 0f0, 0f0)
        return nothing
    end

    d_dev_xx = d.xx - trace_d / 3f0
    d_dev_yy = d.yy - trace_d / 3f0
    d_dev_zz = d.zz - trace_d / 3f0
    j2_d = 0.5f0 * (d_dev_xx^2 + d_dev_yy^2 + d_dev_zz^2) + d.xy^2 + d.xz^2 + d.yz^2
    gamma_dot = 2f0 * sqrt(max(j2_d, 0f0))

    p_val = max(p_trial, 1f-6)
    inertial_denom = sqrt(p_val / rho_s)
    I_num = (inertial_denom > 1f-12) ? (gamma_dot * d_p / inertial_denom) : 0f0
    mu_denom = I_0 + I_num
    kphi_dyn = (mu_denom > 1f-12) ? (mu_s + (mu_2 - mu_s) * I_num / mu_denom) : mu_s

    # 3) Solid/liquid transition criteria:
    #    f_s uses fixed DP friction (Bui), f_d uses mu(I) dynamic friction.
    f_s = tau_trial - (kphi * p_val + kcohesion)
    f_d = tau_trial - (kphi_dyn * p_val + kcohesion)

    if f_s <= 0f0
        # Elastic stage.
        sm_final = sm_trial
        s_dev_final = s_dev_trial
    elseif f_d <= 0f0
        # Micro-viscoplastic stage from fluid-solid model.
        tau_visco = kphi_dyn * p_val + kcohesion
        if gamma_dot > 1f-12
            factor = 2f0 * tau_visco / gamma_dot
            s_dev_final = Tensor2D(factor * d_dev_xx, factor * d_dev_yy, factor * d_dev_zz,
                                   factor * d.xy, factor * d.xz, factor * d.yz)
        else
            s_dev_final = Tensor2D(0f0)
        end
        sm_final = sm_trial
    else
        # 4) Complete viscoplastic stage must stay consistent with dynamic yield surface.
        denom = G_val + K_val * kphi_dyn * kpsi
        dlambda = (denom > 1f-12) ? max(f_d / denom, 0f0) : 0f0

        sm_final = sm_trial - K_val * kpsi * dlambda
        tau_new = tau_trial - G_val * dlambda
        if tau_new <= 1f-12 || tau_trial <= 1f-12
            s_dev_final = Tensor2D(0f0)
        else
            ratio = tau_new / tau_trial
            s_dev_final = s_dev_trial * ratio
        end

        # Keep stress on/inside dynamic yield surface.
        p_new = max(-sm_final, 0f0)
        tau_yield = kphi_dyn * p_new + kcohesion
        tau_curr = sqrt(max(0.5f0 * J2(s_dev_final), 0f0))
        if tau_curr > tau_yield && tau_curr > 1f-12
            s_dev_final = s_dev_final * (tau_yield / tau_curr)
        end

        ep += dlambda * sqrt(1f0 / 3f0 + 2f0 / 9f0 * kpsi^2)
        epd += dlambda * sqrt(1f0 / 3f0)
    end

    sigma_final = Tensor2D(s_dev_final.xx + sm_final, s_dev_final.yy + sm_final, s_dev_final.zz + sm_final,
                           s_dev_final.xy, s_dev_final.xz, s_dev_final.yz)
    p_ref[] = -sm_final
    ep_ref[] = ep
    epd_ref[] = epd
    sigma_ref[] = sigma_final
    return nothing
end

@inline @fastmath function normalize_angle_rad(theta::Float32)
    # Constitutive model 7 reads DP-kphi / DP-kpsi as angles in radians.
    return theta
end

@inline @fastmath function mat_mod_dp_sphinx!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                              K_val, G_val, friction_angle, dilatancy_angle, cohesion)
    ep = ep_ref[]
    epd = epd_ref[]
    sigma = sigma_ref[]

    s_xx, s_yy, s_zz = sigma.xx, sigma.yy, sigma.zz
    s_xy, s_xz, s_yz = sigma.xy, sigma.xz, sigma.yz
    w_xy, w_xz, w_yz = omega[1], omega[2], omega[3]

    ds_xx = muladd(-2f0, muladd(w_xy, s_xy, w_xz * s_xz), 0f0)
    ds_yy = muladd(2f0, muladd(w_xy, s_xy, -(w_yz * s_yz)), 0f0)
    ds_zz = muladd(2f0, muladd(w_xz, s_xz, w_yz * s_yz), 0f0)
    ds_xy = muladd(w_xy, (s_xx - s_yy), muladd(-w_yz, s_xz, -(w_xz * s_yz)))
    ds_xz = muladd(w_xz, (s_xx - s_zz), muladd(w_yz, s_xy, -(w_xy * s_yz)))
    ds_yz = muladd(w_yz, (s_yy - s_zz), muladd(w_xy, s_xz, w_xz * s_xy))

    trace_d = trace(d)
    one_third_trace_d = trace_d / 3f0
    d_dev = Tensor2D(d.xx - one_third_trace_d, d.yy - one_third_trace_d, d.zz - one_third_trace_d,
                     d.xy, d.xz, d.yz)

    stress_rate_elastic = d_dev * (2f0 * G_val) +
                          Tensor2D(trace_d * K_val, trace_d * K_val, trace_d * K_val, 0f0, 0f0, 0f0)
    stress_rate_elastic = stress_rate_elastic + Tensor2D(ds_xx, ds_yy, ds_zz, ds_xy, ds_xz, ds_yz)

    phi = normalize_angle_rad(Float32(friction_angle))
    psi = normalize_angle_rad(Float32(dilatancy_angle))
    tan_phi = tan(phi)
    phi_denom = sqrt(9f0 + 12f0 * tan_phi * tan_phi)
    alpha_phi = (phi_denom > 1f-12) ? (tan_phi / phi_denom) : 0f0
    k_c = (phi_denom > 1f-12) ? (3f0 * cohesion / phi_denom) : 0f0

    i1_curr = trace(sigma)
    sm_curr = i1_curr / 3f0
    s_dev_curr = Tensor2D(sigma.xx - sm_curr, sigma.yy - sm_curr, sigma.zz - sm_curr,
                          sigma.xy, sigma.xz, sigma.yz)
    j2_curr = 0.5f0 * J2(s_dev_curr)
    sqrt_j2_curr = sqrt(max(j2_curr, 0f0))
    f_val = sqrt_j2_curr + alpha_phi * i1_curr - k_c

    stress_rate = stress_rate_elastic
    lambda_dot = 0f0
    a_psi = 0f0

    if f_val >= 1f-8 && sqrt_j2_curr > 1f-10
        tan_psi = tan(psi)
        psi_denom = sqrt(9f0 + 12f0 * tan_psi * tan_psi)
        a_psi = (psi_denom > 1f-12) ? (tan_psi / psi_denom) : 0f0

        s_dot_d = s_dev_curr.xx * d.xx + s_dev_curr.yy * d.yy + s_dev_curr.zz * d.zz +
                  2f0 * (s_dev_curr.xy * d.xy + s_dev_curr.xz * d.xz + s_dev_curr.yz * d.yz)

        denom = 9f0 * alpha_phi * K_val * a_psi + G_val
        if abs(denom) > 1f-12
            lambda_dot = (3f0 * alpha_phi * K_val * trace_d + (G_val / sqrt_j2_curr) * s_dot_d) / denom
            g_iso = 3f0 * K_val * a_psi * lambda_dot
            g_dev_scale = G_val * lambda_dot / sqrt_j2_curr
            g = Tensor2D(g_iso + g_dev_scale * s_dev_curr.xx,
                         g_iso + g_dev_scale * s_dev_curr.yy,
                         g_iso + g_dev_scale * s_dev_curr.zz,
                         g_dev_scale * s_dev_curr.xy,
                         g_dev_scale * s_dev_curr.xz,
                         g_dev_scale * s_dev_curr.yz)
            stress_rate = stress_rate_elastic - g
        end
    end

    sigma_new = sigma + stress_rate * dt

    i1 = trace(sigma_new)
    if (-alpha_phi * i1 + k_c < 0f0) && abs(alpha_phi) > 1f-12
        correction = (i1 - k_c / alpha_phi) / 3f0
        sigma_new = Tensor2D(sigma_new.xx - correction, sigma_new.yy - correction, sigma_new.zz - correction,
                             sigma_new.xy, sigma_new.xz, sigma_new.yz)
    end

    i1 = trace(sigma_new)
    sm_new = i1 / 3f0
    s_dev_new = Tensor2D(sigma_new.xx - sm_new, sigma_new.yy - sm_new, sigma_new.zz - sm_new,
                         sigma_new.xy, sigma_new.xz, sigma_new.yz)
    j2_new = 0.5f0 * J2(s_dev_new)
    sqrt_j2_new = sqrt(max(j2_new, 0f0))

    if (-alpha_phi * i1 + k_c < sqrt_j2_new) && sqrt_j2_new > 1f-10
        tau_allow = max(-alpha_phi * i1 + k_c, 0f0)
        r = tau_allow / sqrt_j2_new
        sigma_new = Tensor2D(r * s_dev_new.xx + sm_new, r * s_dev_new.yy + sm_new, r * s_dev_new.zz + sm_new,
                             r * s_dev_new.xy, r * s_dev_new.xz, r * s_dev_new.yz)
    end

    dlambda = max(lambda_dot, 0f0) * dt
    if dlambda > 0f0
        ep += dlambda * sqrt(1f0 / 3f0 + 2f0 / 9f0 * a_psi * a_psi)
        epd += dlambda * sqrt(1f0 / 3f0)
    end

    p_ref[] = -trace(sigma_new) / 3f0
    ep_ref[] = ep
    epd_ref[] = epd
    sigma_ref[] = sigma_new
    return nothing
end


function kernel_predictor(pospres, velrhop, idep, str, stn, pospres_ini, velrhop_ini, idep_ini, str_ini, stn_ini, accad, spin, acestn, mat_props, par::Parameters)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npm
        i = idx + par.npb
        @inbounds begin
            pospres_ini[i] = pospres[i]
            velrhop_ini[i] = velrhop[i]
            idep_ini[i]    = idep[i]
            str_ini[i]     = str[i]
            stn_ini[i]     = stn[i]
            
            accad_i = accad[i]
            dt_half = 0.5f0 * par.dt
            
            v_x = muladd(accad_i[1], dt_half, velrhop_ini[i][1])
            v_y = muladd(accad_i[2], dt_half, velrhop_ini[i][2])
            v_z = muladd(accad_i[3], dt_half, velrhop_ini[i][3])
            rho_trial = muladd(accad_i[4], dt_half, velrhop_ini[i][4])
            
            p_x = muladd(v_x, dt_half, pospres_ini[i][1])
            p_y = muladd(v_y, dt_half, pospres_ini[i][2])
            p_z = muladd(v_z, dt_half, pospres_ini[i][3])
            
            d_stn = acestn[i] * dt_half
            stn_new = stn_ini[i] + d_stn
            stn[i] = stn_new
            
            omega = spin[i]
            d = acestn[i]
            p_val = pospres_ini[i][4]
            ep_val = idep_ini[i][2]
            epd_val = idep_ini[i][4]
            sigma_val = str_ini[i]
            
            p_ref = Ref(p_val)
            ep_ref = Ref(ep_val)
            epd_ref = Ref(epd_val)
            sigma_ref = Ref(sigma_val)
            
            uid = idep_ini[i][1]
            m_idx = get_mat_idx(uid)
            K_val = mat_props.K[m_idx]
            G_val = mat_props.G[m_idx]
            c_model = mat_props.c_model[m_idx]
            rho = is_solid_model(c_model) ? stabilize_solid_density(rho_trial, mat_props.rho0[m_idx]) : rho_trial
            
            if c_model == 1
                kphi = mat_props.param_1[m_idx]
                kpsi = mat_props.param_2[m_idx]
                kcoh = mat_props.param_3[m_idx]
                mat_mod_dp!(dt_half, omega, d, p_ref, ep_ref, epd_ref, sigma_ref, K_val, G_val, kphi, kpsi, kcoh)
            elseif c_model == 2
                kphi = mat_props.param_1[m_idx]
                kpsi = mat_props.param_2[m_idx]
                kcoh = mat_props.param_3[m_idx]
                mat_mod_dp_bui2008!(dt_half, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                    K_val, G_val, kphi, kpsi, kcoh)
            elseif c_model == 3
                mat_mod_elastic!(dt_half, d, omega, sigma_ref, p_ref, K_val, G_val)
            elseif c_model == 4
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_2  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                kpsi  = mat_props.param_2[m_idx]
                kcoh  = mat_props.param_3[m_idx]
                mat_mod_dp_fluid_solid!(dt_half, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                        K_val, G_val, kcoh, kpsi,
                                        d_p, rho_s, mu_s, mu_2, I_0)
            elseif c_model == 5 || c_model == 6
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_2  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                kphi  = mat_props.param_1[m_idx]
                kpsi  = mat_props.param_2[m_idx]
                kcoh  = mat_props.param_3[m_idx]
                mat_mod_dp_fluid_solid_bui2008!(dt_half, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                                K_val, G_val, kphi, kpsi, kcoh,
                                                d_p, rho_s, mu_s, mu_2, I_0)
            elseif c_model == 7
                friction_angle = mat_props.param_1[m_idx]
                dilatancy_angle = mat_props.param_2[m_idx]
                cohesion = mat_props.param_3[m_idx]
                mat_mod_dp_sphinx!(dt_half, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                   K_val, G_val, friction_angle, dilatancy_angle, cohesion)
            elseif c_model == 8
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_d  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                lambda_reg = mat_props.mu_I_lambda[m_idx]
                psi = mat_props.param_2[m_idx]
                cohesion = mat_props.param_3[m_idx]
                mat_mod_regularized_mu_i_dp!(dt_half, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                             K_val, G_val, cohesion, psi,
                                             d_p, rho_s, mu_s, mu_d, I_0, lambda_reg)
            else
                mat_mod_hypo!(dt_half, d, omega, sigma_ref, K_val, G_val)
                ep_ref[] = sqrt(SQRT_2_DIV_3 * J2(stn_new))
            end
            
            pospres[i] = Float4(p_x, p_y, p_z, p_ref[])
            velrhop[i] = Float4(v_x, v_y, v_z, rho)
            idep[i]    = Float4(idep[i][1], ep_ref[], idep[i][3], epd_ref[])
            str[i]     = sigma_ref[]
        end
    end
    return nothing
end

function kernel_corrector(pospres, velrhop, idep, str, stn, pospres_ini, velrhop_ini, idep_ini, str_ini, stn_ini, accad, spin, acestn, mat_props, par::Parameters)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npm
        i = idx + par.npb
        @inbounds begin
            pos_ini = pospres_ini[i]
            vel_ini = velrhop_ini[i]
            
            accad_i = accad[i]
            dt = par.dt
            
            v_x = muladd(accad_i[1], dt, vel_ini[1])
            v_y = muladd(accad_i[2], dt, vel_ini[2])
            v_z = muladd(accad_i[3], dt, vel_ini[3])
            rho_trial = muladd(accad_i[4], dt, vel_ini[4])
            
            p_x = muladd(0.5f0 * dt, (v_x + vel_ini[1]), pos_ini[1])
            p_y = muladd(0.5f0 * dt, (v_y + vel_ini[2]), pos_ini[2])
            p_z = muladd(0.5f0 * dt, (v_z + vel_ini[3]), pos_ini[3])
            
            d_stn = acestn[i] * dt
            stn_new = stn_ini[i] + d_stn
            stn[i] = stn_new
            
            omega = spin[i]
            d = acestn[i]
            p_val = pospres[i][4]
            ep_val = idep_ini[i][2]
            epd_val = idep_ini[i][4]
            sigma_val = str_ini[i]
            
            p_ref = Ref(p_val)
            ep_ref = Ref(ep_val)
            epd_ref = Ref(epd_val)
            sigma_ref = Ref(sigma_val)

            uid = idep_ini[i][1]
            m_idx = get_mat_idx(uid)
            K_val = mat_props.K[m_idx]
            G_val = mat_props.G[m_idx]
            c_model = mat_props.c_model[m_idx]
            rho = is_solid_model(c_model) ? stabilize_solid_density(rho_trial, mat_props.rho0[m_idx]) : rho_trial
            
            if c_model == 1
                kphi = mat_props.param_1[m_idx]
                kpsi = mat_props.param_2[m_idx]
                kcoh = mat_props.param_3[m_idx]
                mat_mod_dp!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref, K_val, G_val, kphi, kpsi, kcoh)
            elseif c_model == 2
                kphi = mat_props.param_1[m_idx]
                kpsi = mat_props.param_2[m_idx]
                kcoh = mat_props.param_3[m_idx]
                mat_mod_dp_bui2008!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                    K_val, G_val, kphi, kpsi, kcoh)
            elseif c_model == 3
                mat_mod_elastic!(dt, d, omega, sigma_ref, p_ref, K_val, G_val)
            elseif c_model == 4
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_2  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                kpsi  = mat_props.param_2[m_idx]
                kcoh  = mat_props.param_3[m_idx]
                mat_mod_dp_fluid_solid!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                        K_val, G_val, kcoh, kpsi,
                                        d_p, rho_s, mu_s, mu_2, I_0)
            elseif c_model == 5 || c_model == 6
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_2  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                kphi  = mat_props.param_1[m_idx]
                kpsi  = mat_props.param_2[m_idx]
                kcoh  = mat_props.param_3[m_idx]
                mat_mod_dp_fluid_solid_bui2008!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                                K_val, G_val, kphi, kpsi, kcoh,
                                                d_p, rho_s, mu_s, mu_2, I_0)
            elseif c_model == 7
                friction_angle = mat_props.param_1[m_idx]
                dilatancy_angle = mat_props.param_2[m_idx]
                cohesion = mat_props.param_3[m_idx]
                mat_mod_dp_sphinx!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                   K_val, G_val, friction_angle, dilatancy_angle, cohesion)
            elseif c_model == 8
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_d  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                lambda_reg = mat_props.mu_I_lambda[m_idx]
                psi = mat_props.param_2[m_idx]
                cohesion = mat_props.param_3[m_idx]
                mat_mod_regularized_mu_i_dp!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                             K_val, G_val, cohesion, psi,
                                             d_p, rho_s, mu_s, mu_d, I_0, lambda_reg)
            else
                mat_mod_hypo!(dt, d, omega, sigma_ref, K_val, G_val)
                ep_ref[] = sqrt(SQRT_2_DIV_3 * J2(stn_new))
            end
            
            pospres[i] = Float4(p_x, p_y, p_z, p_ref[])
            velrhop[i] = Float4(v_x, v_y, v_z, rho)
            idep[i]    = Float4(idep[i][1], ep_ref[], idep[i][3], epd_ref[])
            str[i]     = sigma_ref[]
        end
    end
    return nothing
end

function kernel_leapfrog_kickstart(velrhop, accad, par::Parameters)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npm
        i = idx + par.npb
        @inbounds begin
            vel_i = velrhop[i]
            acc_i = accad[i]
            dt_half = 0.5f0 * par.dt
            v_x = muladd(-acc_i[1], dt_half, vel_i[1])
            v_y = muladd(-acc_i[2], dt_half, vel_i[2])
            v_z = muladd(-acc_i[3], dt_half, vel_i[3])
            velrhop[i] = Float4(v_x, v_y, v_z, vel_i[4])
        end
    end
    return nothing
end

function kernel_leapfrog_kinematics(pospres, velrhop, accad, bridge_mask, par::Parameters)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npm
        i = idx + par.npb
        @inbounds begin
            if bridge_mask[i] > 0.5f0
                return nothing
            end
            pos_i = pospres[i]
            vel_i = velrhop[i]
            acc_i = accad[i]
            dt = par.dt

            # 1) Velocity update to n+1/2.
            v_x = muladd(acc_i[1], dt, vel_i[1])
            v_y = muladd(acc_i[2], dt, vel_i[2])
            v_z = muladd(acc_i[3], dt, vel_i[3])

            # 2) Position update to n+1.
            p_x = muladd(v_x, dt, pos_i[1])
            p_y = muladd(v_y, dt, pos_i[2])
            p_z = muladd(v_z, dt, pos_i[3])

            # Keep pressure/density untouched in this kernel.
            pospres[i] = Float4(p_x, p_y, p_z, pos_i[4])
            velrhop[i] = Float4(v_x, v_y, v_z, vel_i[4])
        end
    end
    return nothing
end

function kernel_leapfrog_stress(pospres, velrhop, idep, str, stn, accad, spin, acestn, mat_props, par::Parameters)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= par.npm
        i = idx + par.npb
        @inbounds begin
            pos_i = pospres[i]
            vel_i = velrhop[i]
            acc_i = accad[i]
            dt = par.dt

            # 3) Density update to n+1 using latest drho/dt.
            rho_trial = muladd(acc_i[4], dt, vel_i[4])

            # 4) Strain/stress update to n+1 using latest acestn.
            d_stn = acestn[i] * dt
            stn_new = stn[i] + d_stn
            stn[i] = stn_new

            omega = spin[i]
            d = acestn[i]
            p_val = pos_i[4]
            ep_val = idep[i][2]
            epd_val = idep[i][4]
            sigma_val = str[i]

            p_ref = Ref(p_val)
            ep_ref = Ref(ep_val)
            epd_ref = Ref(epd_val)
            sigma_ref = Ref(sigma_val)

            uid = idep[i][1]
            m_idx = get_mat_idx(uid)
            K_val = mat_props.K[m_idx]
            G_val = mat_props.G[m_idx]
            c_model = mat_props.c_model[m_idx]
            rho = is_solid_model(c_model) ? stabilize_solid_density(rho_trial, mat_props.rho0[m_idx]) : rho_trial

            # Constitutive update.
            if c_model == 1
                kphi = mat_props.param_1[m_idx]
                kpsi = mat_props.param_2[m_idx]
                kcoh = mat_props.param_3[m_idx]
                mat_mod_dp!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref, K_val, G_val, kphi, kpsi, kcoh)
            elseif c_model == 2
                kphi = mat_props.param_1[m_idx]
                kpsi = mat_props.param_2[m_idx]
                kcoh = mat_props.param_3[m_idx]
                mat_mod_dp_bui2008!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                    K_val, G_val, kphi, kpsi, kcoh)
            elseif c_model == 3
                mat_mod_elastic!(dt, d, omega, sigma_ref, p_ref, K_val, G_val)
            elseif c_model == 4
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_2  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                kpsi  = mat_props.param_2[m_idx]
                kcoh  = mat_props.param_3[m_idx]
                mat_mod_dp_fluid_solid!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                        K_val, G_val, kcoh, kpsi,
                                        d_p, rho_s, mu_s, mu_2, I_0)
            elseif c_model == 5 || c_model == 6
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_2  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                kphi  = mat_props.param_1[m_idx]
                kpsi  = mat_props.param_2[m_idx]
                kcoh  = mat_props.param_3[m_idx]
                mat_mod_dp_fluid_solid_bui2008!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                                K_val, G_val, kphi, kpsi, kcoh,
                                                d_p, rho_s, mu_s, mu_2, I_0)
            elseif c_model == 7
                friction_angle = mat_props.param_1[m_idx]
                dilatancy_angle = mat_props.param_2[m_idx]
                cohesion = mat_props.param_3[m_idx]
                mat_mod_dp_sphinx!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                   K_val, G_val, friction_angle, dilatancy_angle, cohesion)
            elseif c_model == 8
                d_p   = mat_props.d_p[m_idx]
                rho_s = mat_props.rho_s[m_idx]
                mu_s  = mat_props.mu_s[m_idx]
                mu_d  = mat_props.mu_2[m_idx]
                I_0   = mat_props.I_0[m_idx]
                lambda_reg = mat_props.mu_I_lambda[m_idx]
                psi = mat_props.param_2[m_idx]
                cohesion = mat_props.param_3[m_idx]
                mat_mod_regularized_mu_i_dp!(dt, omega, d, p_ref, ep_ref, epd_ref, sigma_ref,
                                             K_val, G_val, cohesion, psi,
                                             d_p, rho_s, mu_s, mu_d, I_0, lambda_reg)
            else
                mat_mod_hypo!(dt, d, omega, sigma_ref, K_val, G_val)
                ep_ref[] = sqrt(SQRT_2_DIV_3 * J2(stn_new))
            end

            # Write back pressure/density and constitutive state.
            pospres[i] = Float4(pos_i[1], pos_i[2], pos_i[3], p_ref[])
            velrhop[i] = Float4(vel_i[1], vel_i[2], vel_i[3], rho)
            idep[i]    = Float4(idep[i][1], ep_ref[], idep[i][3], epd_ref[])
            str[i]     = sigma_ref[]
        end
    end
    return nothing
end

end # module

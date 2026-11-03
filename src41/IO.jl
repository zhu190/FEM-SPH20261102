module IO

using LightXML
using Printf
using StaticArrays
using Base64
using ..Types: Parameters, Tensor2D, Float4, PI_F

export load_parameters, load_particles, save_xml, save_particles

mutable struct SaveScratch
    head::Vector{Int32}
    next_idx::Vector{Int32}
    valid::BitVector
    cxs::Vector{Int32}
    cys::Vector{Int32}
    czs::Vector{Int32}
    neighbor_count::Vector{Int32}
    part_index_map::Dict{Int32, Vector{Int}}
    ref_pos_by_oid::Dict{Int32, Float4}
    ref_pos_cache_key::Tuple{UInt, UInt}
    float32_buf::Vector{Float32}
    int32_buf::Vector{Int32}
    uint8_buf::Vector{UInt8}
end

const SAVE_SCRATCHES = Dict{String, SaveScratch}()

function _get_save_scratch(project_name::String, np::Int, ncell::Int)
    scratch = get!(SAVE_SCRATCHES, project_name) do
        SaveScratch(
            zeros(Int32, ncell),
            zeros(Int32, np),
            falses(np),
            zeros(Int32, np),
            zeros(Int32, np),
            zeros(Int32, np),
            zeros(Int32, np),
            Dict{Int32, Vector{Int}}(),
            Dict{Int32, Float4}(),
            (UInt(0), UInt(0)),
            Float32[],
            Int32[],
            UInt8[]
        )
    end

    length(scratch.head) == ncell || resize!(scratch.head, ncell)
    length(scratch.next_idx) == np || resize!(scratch.next_idx, np)
    length(scratch.valid) == np || resize!(scratch.valid, np)
    length(scratch.cxs) == np || resize!(scratch.cxs, np)
    length(scratch.cys) == np || resize!(scratch.cys, np)
    length(scratch.czs) == np || resize!(scratch.czs, np)
    length(scratch.neighbor_count) == np || resize!(scratch.neighbor_count, np)
    return scratch
end

function get_xml_value(params_dict, key, type_convert)
    if !haskey(params_dict, key)
        error("Error! Parameter '$key' does not exist!")
    end
    try
        return type_convert(params_dict[key])
    catch
        error("Error! Type mismatch for key: $key")
    end
end

to_int(x) = parse(Int32, x)
to_float(x) = parse(Float32, x)

function parse_kernel_type(x::AbstractString)
    s = lowercase(strip(x))
    if s in ("wendland", "wendland_c2", "0")
        return Int32(0)
    elseif s in ("cubic_spline", "cubicspline", "cubic", "1")
        return Int32(1)
    end
    error("Unsupported kernel-function '$x'. Use 'wendland' or 'cubic_spline'.")
end

function parse_boundary0_output_mode(x::AbstractString)
    s = lowercase(strip(x))
    if s in ("0", "initial", "init", "initial-only")
        return Int32(0)
    elseif s in ("1", "all", "all-time", "full-time")
        return Int32(1)
    end
    error("Unsupported boundary0-output-mode '$x'. Use 0/1 or initial/all-time.")
end

function load_parameters(project_name::String)
    xml_filename = project_name * ".xml"
    if !isfile(xml_filename)
        error("Error! Cannot load $xml_filename")
    end
    println("\nLoading parameters from $xml_filename ...")
    xdoc = parse_file(xml_filename)
    xroot = root(xdoc)
    params_dict = Dict{String, String}()
    for c in child_nodes(xroot)
        if is_elementnode(c)
            e = XMLElement(c)
            n = lowercase(strip(name(e)))
            if n in ("part-acceleration", "part-local-acceleration", "local-acceleration", "part-acc")
                continue
            end
            params_dict[name(e)] = content(e)
        end
    end

    function parse_part_local_accelerations(xroot, default_ax::Float32, default_ay::Float32, default_az::Float32)
        part_acc_map = Dict{Int32, Float4}()

        function parse_part_id_str(s::AbstractString)
            sid = strip(s)
            isempty(sid) && error("part-acceleration has empty part-id.")
            return parse(Int32, sid)
        end

        function parse_part_acc_node!(elem::XMLElement)
            pid_str = attribute(elem, "part-id")
            pid_str === nothing && (pid_str = attribute(elem, "part"))

            ax = default_ax
            ay = default_ay
            az = default_az

            for p in child_nodes(elem)
                if !is_elementnode(p)
                    continue
                end
                pe = XMLElement(p)
                pn = lowercase(strip(name(pe)))
                pv = strip(content(pe))
                if pn in ("part-id", "part", "partid")
                    pid_str = pv
                elseif pn in ("acceleration-x", "acc-x", "ax")
                    ax = parse(Float32, pv)
                elseif pn in ("acceleration-y", "acc-y", "ay")
                    ay = parse(Float32, pv)
                elseif pn in ("acceleration-z", "acc-z", "az")
                    az = parse(Float32, pv)
                end
            end

            pid_str === nothing && error("part-acceleration node requires 'part-id' attribute or child element.")
            pid = parse_part_id_str(pid_str)
            part_acc_map[pid] = Float4(ax, ay, az, 0f0)
            return nothing
        end

        for c in child_nodes(xroot)
            if !is_elementnode(c)
                continue
            end
            e = XMLElement(c)
            n = lowercase(strip(name(e)))
            if n in ("part-acceleration", "part-local-acceleration", "local-acceleration", "part-acc")
                parse_part_acc_node!(e)
            end
        end

        ids = sort!(collect(keys(part_acc_map)))
        vals = [part_acc_map[id] for id in ids]
        return ids, vals
    end
    
    global_rho0 = haskey(params_dict, "reference-density") ? parse(Float32, params_dict["reference-density"]) : 1000.0f0
    global_K    = haskey(params_dict, "elastic-K") ? parse(Float32, params_dict["elastic-K"]) : 0.0f0
    global_G    = haskey(params_dict, "elastic-G") ? parse(Float32, params_dict["elastic-G"]) : 0.0f0
    global_model = haskey(params_dict, "constitutive-model") ? parse(Int32, params_dict["constitutive-model"]) : Int32(3)
    global_p3 = haskey(params_dict, "DP-kphi") ? parse(Float32, params_dict["DP-kphi"]) : 0f0
    global_p4 = haskey(params_dict, "DP-kpsi") ? parse(Float32, params_dict["DP-kpsi"]) :
                (haskey(params_dict, "dilatancy-angle") ? parse(Float32, params_dict["dilatancy-angle"]) : 0f0)
    global_p5 = haskey(params_dict, "DP-kcohesion") ? parse(Float32, params_dict["DP-kcohesion"]) :
                (haskey(params_dict, "cohesion") ? parse(Float32, params_dict["cohesion"]) : 0f0)
    
    
    global_d_p   = haskey(params_dict, "particle-diameter") ? parse(Float32, params_dict["particle-diameter"]) : 0f0
    global_rho_s = haskey(params_dict, "solid-density") ? parse(Float32, params_dict["solid-density"]) : 0f0
    global_mu_s  = haskey(params_dict, "mu-s") ? parse(Float32, params_dict["mu-s"]) : 0f0
    global_mu_2  = haskey(params_dict, "mu-2") ? parse(Float32, params_dict["mu-2"]) :
                   (haskey(params_dict, "mu-d") ? parse(Float32, params_dict["mu-d"]) : 0f0)
    global_I_0   = haskey(params_dict, "I-0") ? parse(Float32, params_dict["I-0"]) : 0f0
    global_mu_I_lambda = haskey(params_dict, "regularization-lambda") ? parse(Float32, params_dict["regularization-lambda"]) : 1f-3
    
    max_mats = 20 
    h_rho0 = fill(global_rho0, max_mats)
    h_K    = fill(global_K, max_mats)
    h_G    = fill(global_G, max_mats)
    h_c_model = fill(global_model, max_mats)
    h_p1 = fill(global_p3, max_mats)
    h_p2 = fill(global_p4, max_mats)
    h_p3 = fill(global_p5, max_mats)
    
    
    h_d_p   = fill(global_d_p, max_mats)
    h_rho_s = fill(global_rho_s, max_mats)
    h_mu_s  = fill(global_mu_s, max_mats)
    h_mu_2  = fill(global_mu_2, max_mats)
    h_I_0   = fill(global_I_0, max_mats)
    h_mu_I_lambda = fill(global_mu_I_lambda, max_mats)

    function parse_material_node!(idx, element)
        for p in child_nodes(element)
            if is_elementnode(p)
                pn = name(XMLElement(p))
                val_str = content(XMLElement(p))
                if pn == "rho0"      h_rho0[idx] = parse(Float32, val_str) end
                if pn == "elastic-K" h_K[idx]    = parse(Float32, val_str) end
                if pn == "elastic-G" h_G[idx]    = parse(Float32, val_str) end
                if pn == "constitutive-model" h_c_model[idx] = parse(Int32, val_str) end
                if pn == "DP-kphi"      h_p1[idx] = parse(Float32, val_str) end
                if pn == "DP-kpsi"      h_p2[idx] = parse(Float32, val_str) end
                if pn == "DP-kcohesion" h_p3[idx] = parse(Float32, val_str) end
                if pn == "dilatancy-angle" h_p2[idx] = parse(Float32, val_str) end
                if pn == "cohesion" h_p3[idx] = parse(Float32, val_str) end
                
                if pn == "particle-diameter" h_d_p[idx] = parse(Float32, val_str) end
                if pn == "solid-density"     h_rho_s[idx] = parse(Float32, val_str) end
                if pn == "mu-s"              h_mu_s[idx] = parse(Float32, val_str) end
                if pn == "mu-2"              h_mu_2[idx] = parse(Float32, val_str) end
                if pn == "mu-d"              h_mu_2[idx] = parse(Float32, val_str) end
                if pn == "I-0"               h_I_0[idx] = parse(Float32, val_str) end
                if pn == "regularization-lambda" h_mu_I_lambda[idx] = parse(Float32, val_str) end
            end
        end
    end

    for c in child_nodes(xroot)
        if is_elementnode(c) && name(XMLElement(c)) == "boundary-material"
            be = XMLElement(c)
            parse_material_node!(1, be)
        end
    end

    for c in child_nodes(xroot)
        if is_elementnode(c) && name(XMLElement(c)) == "material"
            me = XMLElement(c)
            idx_str = attribute(me, "index")
            if idx_str !== nothing
                idx = parse(Int, idx_str)
                array_idx = idx + 1 
                if array_idx <= max_mats
                    parse_material_node!(array_idx, me)
                end
            end
        end
    end

    np  = get_xml_value(params_dict, "number-of-particles", to_int)
    npb = get_xml_value(params_dict, "number-of-boundary-particles", to_int)
    npm = get_xml_value(params_dict, "number-of-material-particles", to_int)
    c_model = global_model
    p1 = global_K; p2 = global_G; p3 = global_p3; p4 = global_p4; p5 = global_p5

    vel_limit = haskey(params_dict, "velocity-limit") ? parse(Float32, params_dict["velocity-limit"]) : 10000.0f0
    momentum_form = haskey(params_dict, "momentum-form") ? parse(Int32, params_dict["momentum-form"]) : Int32(0)
    artificial_viscosity_q1 = haskey(params_dict, "artificial-viscosity-q1") ? parse(Float32, params_dict["artificial-viscosity-q1"]) : 0f0
    artificial_viscosity_q2 = haskey(params_dict, "artificial-viscosity-q2") ? parse(Float32, params_dict["artificial-viscosity-q2"]) : 0f0
    boundary_velocity_type = haskey(params_dict, "boundary-velocity-type") ? parse(Int32, params_dict["boundary-velocity-type"]) : Int32(0)
    wall_velocity_x = haskey(params_dict, "wall-velocity-x") ? parse(Float32, params_dict["wall-velocity-x"]) : 0f0
    wall_velocity_y = haskey(params_dict, "wall-velocity-y") ? parse(Float32, params_dict["wall-velocity-y"]) : 0f0
    wall_velocity_z = haskey(params_dict, "wall-velocity-z") ? parse(Float32, params_dict["wall-velocity-z"]) : 0f0
    initial_damping_duration = haskey(params_dict, "initial-damping-duration") ? parse(Float32, params_dict["initial-damping-duration"]) : 0f0
    initial_damping_coefficient = haskey(params_dict, "initial-damping-coefficient") ? parse(Float32, params_dict["initial-damping-coefficient"]) :
                                  (haskey(params_dict, "initial-damping-coeff") ? parse(Float32, params_dict["initial-damping-coeff"]) : 0f0)
    boundary_part0_output_mode = haskey(params_dict, "boundary0-output-mode") ?
                                 parse_boundary0_output_mode(params_dict["boundary0-output-mode"]) :
                                 (haskey(params_dict, "boundary-part0-output-mode") ?
                                  parse_boundary0_output_mode(params_dict["boundary-part0-output-mode"]) :
                                 Int32(1)
                                 )
    static_boundary_grid = haskey(params_dict, "static-boundary-grid") ? parse(Int32, params_dict["static-boundary-grid"]) : Int32(1)
    active_boundary_adami = haskey(params_dict, "active-boundary-adami") ? parse(Int32, params_dict["active-boundary-adami"]) : Int32(1)
    depth_footprint_radius = haskey(params_dict, "depth-footprint-radius") ? parse(Float32, params_dict["depth-footprint-radius"]) : 0f0
    depth_search_max = haskey(params_dict, "depth-search-max") ? parse(Float32, params_dict["depth-search-max"]) : 0f0
    normal_orientation_radius = haskey(params_dict, "normal-orientation-radius") ? parse(Float32, params_dict["normal-orientation-radius"]) : 0f0
    entrainment_enabled = haskey(params_dict, "entrainment-enabled") ? parse(Int32, params_dict["entrainment-enabled"]) : Int32(0)
    entrainment_pore_pressure = haskey(params_dict, "entrainment-pore-pressure-coefficient") ? parse(Float32, params_dict["entrainment-pore-pressure-coefficient"]) : 0.55f0
    entrainment_flow_friction_deg = haskey(params_dict, "entrainment-flow-friction-angle") ? parse(Float32, params_dict["entrainment-flow-friction-angle"]) : 11.31f0
    entrainment_bed_friction_deg = haskey(params_dict, "entrainment-bed-friction-angle") ? parse(Float32, params_dict["entrainment-bed-friction-angle"]) : 29.25f0
    erosion_depth_max = haskey(params_dict, "erosion-depth-max") ? parse(Float32, params_dict["erosion-depth-max"]) : 0f0
    entrainment_speed_min = haskey(params_dict, "entrainment-speed-min") ? parse(Float32, params_dict["entrainment-speed-min"]) : 0.01f0
    entrainment_rate_max = haskey(params_dict, "entrainment-rate-max") ? parse(Float32, params_dict["entrainment-rate-max"]) : 0f0
    depth_footprint_radius < 0f0 && error("depth-footprint-radius must be non-negative (0 selects the default 2h radius).")
    depth_search_max < 0f0 && error("depth-search-max must be non-negative (0 selects the vertical domain size).")
    normal_orientation_radius < 0f0 && error("normal-orientation-radius must be non-negative (0 selects the default 4h radius).")
    entrainment_enabled in (Int32(0), Int32(1)) || error("entrainment-enabled must be 0 or 1.")
    0f0 <= entrainment_pore_pressure <= 1f0 || error("entrainment-pore-pressure-coefficient must be in [0, 1].")
    0f0 <= entrainment_flow_friction_deg < 90f0 || error("entrainment-flow-friction-angle must be in [0, 90) degrees.")
    0f0 <= entrainment_bed_friction_deg < 90f0 || error("entrainment-bed-friction-angle must be in [0, 90) degrees.")
    erosion_depth_max >= 0f0 || error("erosion-depth-max must be non-negative.")
    entrainment_speed_min > 0f0 || error("entrainment-speed-min must be positive.")
    entrainment_rate_max >= 0f0 || error("entrainment-rate-max must be non-negative (0 disables the rate cap).")
    entrainment_enabled != 0 && erosion_depth_max <= 0f0 && error("erosion-depth-max must be positive when entrainment-enabled=1.")
    entrainment_flow_friction_angle = entrainment_flow_friction_deg * (PI_F / 180f0)
    entrainment_bed_friction_angle = entrainment_bed_friction_deg * (PI_F / 180f0)
    kernel_type = haskey(params_dict, "kernel-function") ? parse_kernel_type(params_dict["kernel-function"]) : Int32(0)
    global_acc_x = get_xml_value(params_dict, "acceleration-x", to_float)
    global_acc_y = get_xml_value(params_dict, "acceleration-y", to_float)
    global_acc_z = get_xml_value(params_dict, "acceleration-z", to_float)
    part_acc_ids, part_acc_vals = parse_part_local_accelerations(xroot, global_acc_x, global_acc_y, global_acc_z)

    args64 = (
        np, npb, npm, Int32(0),
        get_xml_value(params_dict, "domain-size-x", to_float), get_xml_value(params_dict, "domain-size-y", to_float), get_xml_value(params_dict, "domain-size-z", to_float),
        get_xml_value(params_dict, "domain-min-x", to_float), get_xml_value(params_dict, "domain-min-y", to_float), get_xml_value(params_dict, "domain-min-z", to_float),
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), 0f0, 0f0, 0f0,
        get_xml_value(params_dict, "particle-size", to_float), get_xml_value(params_dict, "h-over-dr", to_float), 0f0,
        get_xml_value(params_dict, "speed-of-sound", to_float), get_xml_value(params_dict, "cfl-number", to_float), get_xml_value(params_dict, "reference-density", to_float),
        c_model, p1, p2, p3, p4, p5,
        get_xml_value(params_dict, "density-shepard-filter", to_int), get_xml_value(params_dict, "stress-shepard-filter", to_int),
        get_xml_value(params_dict, "dynamic-time-step", to_int), get_xml_value(params_dict, "integration-method", to_int), get_xml_value(params_dict, "viscosity-type", to_int),
        haskey(params_dict, "artificial-viscosity-alpha") ? parse(Float32, params_dict["artificial-viscosity-alpha"]) : 0f0,
        artificial_viscosity_q1,
        artificial_viscosity_q2,
        momentum_form,
        get_xml_value(params_dict, "artificial-pressure", to_int),
        haskey(params_dict, "artificial-pressure-coefficient") ? parse(Float32, params_dict["artificial-pressure-coefficient"]) : 0f0,
        global_acc_x, global_acc_y, global_acc_z,
        get_xml_value(params_dict, "time-max", to_float), get_xml_value(params_dict, "time", to_float), get_xml_value(params_dict, "time-step", to_float),
        get_xml_value(params_dict, "output-frequency", to_float), get_xml_value(params_dict, "output-number", to_int),
        0f0, 0f0, 0f0, 0f0, kernel_type,
        get_xml_value(params_dict, "output-format", to_int),
        vel_limit,
        boundary_velocity_type,
        wall_velocity_x, wall_velocity_y, wall_velocity_z,
        initial_damping_duration, initial_damping_coefficient,
        boundary_part0_output_mode,
        static_boundary_grid,
        active_boundary_adami,
        depth_footprint_radius,
        depth_search_max,
        normal_orientation_radius,
        entrainment_enabled,
        entrainment_pore_pressure,
        entrainment_flow_friction_angle,
        entrainment_bed_friction_angle,
        erosion_depth_max,
        entrainment_speed_min,
        entrainment_rate_max
    )
    p = try
        Parameters(args64...)
    catch e
        if e isa MethodError
            # Compatibility fallback for sessions/code paths that still expose the legacy 57-arg constructor.
            @warn "Falling back to legacy Parameters constructor (57 args) due constructor mismatch." exception = (e, catch_backtrace())
            Parameters(args64[1:57]...)
        else
            rethrow(e)
        end
    end
    free(xdoc)
    
    return p, h_rho0, h_K, h_G, h_c_model, h_p1, h_p2, h_p3,
           h_d_p, h_rho_s, h_mu_s, h_mu_2, h_I_0, h_mu_I_lambda,
           part_acc_ids, part_acc_vals
end

function load_particles(project_name::String, par::Parameters)
    geom_filename = project_name * "_geom.dat"
    state_filename = project_name * "_state.dat"
    
    if !isfile(geom_filename) || !isfile(state_filename)
        error("Error! Cannot open $geom_filename or $state_filename")
    end
    
    println("Loading particle data from split files: $geom_filename and $state_filename ...")
    
    pospres = Vector{Float4}(undef, par.np)
    velrhop = Vector{Float4}(undef, par.np)
    idep    = Vector{Float4}(undef, par.np)
    str     = Vector{Tensor2D}(undef, par.np)
    stn     = Vector{Tensor2D}(undef, par.np)

    dom_max_x = par.domain_min_x + par.domain_size_x
    dom_max_y = par.domain_min_y + par.domain_size_y
    dom_max_z = par.domain_min_z + par.domain_size_z
    out_total = 0
    out_boundary = 0
    out_material = 0
    out_examples = String[]

    function read_fields!(io::Base.IO, expected_cols::Int, file_label::String, particle_idx::Int)
        while !eof(io)
            line = readline(io)
            s = strip(line)
            isempty(s) && continue
            cols = split(s)
            if length(cols) < expected_cols
                error("Malformed $file_label at particle index $particle_idx: expected at least $expected_cols columns, got $(length(cols)). Line='$s'")
            end
            return parse.(Float32, cols)
        end
        error("Unexpected EOF in $file_label while reading particle index $particle_idx.")
    end
    
    open(geom_filename, "r") do io_geom
        open(state_filename, "r") do io_state
            for i in 1:par.np
                fg = read_fields!(io_geom, 5, geom_filename, i)
                
                fs = read_fields!(io_state, 18, state_filename, i)
                
                partID = fg[1]
                original_id = fg[2] 

                x = fg[3]
                y = fg[4]
                z = fg[5]
                outside = (x < par.domain_min_x) || (x > dom_max_x) ||
                          (y < par.domain_min_y) || (y > dom_max_y) ||
                          (z < par.domain_min_z) || (z > dom_max_z)
                if outside
                    out_total += 1
                    if partID < 100f0
                        out_boundary += 1
                    else
                        out_material += 1
                    end
                    if length(out_examples) < 8
                        push!(out_examples, "idx=$i partID=$(Int(round(partID))) xyz=($x,$y,$z)")
                    end
                end
                
                pospres[i] = Float4(fg[3], fg[4], fg[5], fs[2])         
                velrhop[i] = Float4(fs[3], fs[4], fs[5], fs[1])         
                idep[i]    = Float4(partID, fs[6], original_id, 0f0)    
                
                str[i]     = Tensor2D(fs[7], fs[8], fs[9], fs[10], fs[11], fs[12])
                stn[i]     = Tensor2D(fs[13], fs[14], fs[15], fs[16], fs[17], fs[18])
            end
        end
    end

    if out_total > 0
        example_msg = isempty(out_examples) ? "" : ("\n  samples:\n    " * join(out_examples, "\n    "))
        error("Found out-of-domain particles in input files. total=$out_total, boundary=$out_boundary, material=$out_material. Domain x=[$(par.domain_min_x),$dom_max_x], y=[$(par.domain_min_y),$dom_max_y], z=[$(par.domain_min_z),$dom_max_z].$example_msg")
    end

    return pospres, velrhop, idep, str, stn
end

function write_vtk_binary(io, data)
    raw_bytes = reinterpret(UInt8, data)
    header = [UInt32(length(raw_bytes))]
    encoded = base64encode(vcat(reinterpret(UInt8, header), raw_bytes))
    write(io, encoded)
end

function save_to_dat(filename, pos, vel, idep, str, stn, n, start)
    open(filename, "w") do io
        for i in 0:n-1
            idx = start + i
           
            @printf(io, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                pos[idx][1], pos[idx][2], pos[idx][3], pos[idx][4], 
                vel[idx][1], vel[idx][2], vel[idx][3], vel[idx][4],
                idep[idx][1], idep[idx][2], 
                str[idx].xx, str[idx].yy, str[idx].zz, str[idx].xy, str[idx].xz, str[idx].yz,
                stn[idx].xx, stn[idx].yy, stn[idx].zz, stn[idx].xy, stn[idx].xz, stn[idx].yz,
                idep[idx][4], idep[idx][3])
        end
    end
end

function save_to_dat(filename, pos, vel, idep, str, stn, idxs::AbstractVector{Int})
    open(filename, "w") do io
        @inbounds for idx in idxs
            @printf(io, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                pos[idx][1], pos[idx][2], pos[idx][3], pos[idx][4],
                vel[idx][1], vel[idx][2], vel[idx][3], vel[idx][4],
                idep[idx][1], idep[idx][2],
                str[idx].xx, str[idx].yy, str[idx].zz, str[idx].xy, str[idx].xz, str[idx].yz,
                stn[idx].xx, stn[idx].yy, stn[idx].zz, stn[idx].xy, stn[idx].xz, stn[idx].yz,
                idep[idx][4], idep[idx][3])
        end
    end
end

function compute_neighbor_count_host(pos::Vector{Float4}, idep::Vector{Float4}, par::Parameters, scratch::SaveScratch)
    np = length(pos)
    counts = scratch.neighbor_count
    np == 0 && return counts

    ih = Float32(par.i_h)
    q2_limit = 4f0
    head = scratch.head
    next_idx = scratch.next_idx
    valid = scratch.valid
    cxs = scratch.cxs
    cys = scratch.cys
    czs = scratch.czs

    fill!(counts, 0)
    fill!(head, 0)
    fill!(next_idx, 0)
    fill!(valid, false)

    dom_max_x = par.domain_min_x + par.domain_size_x
    dom_max_y = par.domain_min_y + par.domain_size_y
    dom_max_z = par.domain_min_z + par.domain_size_z

    @inbounds for i in 1:np
        p = pos[i]
        if !(isfinite(p[1]) && isfinite(p[2]) && isfinite(p[3]) &&
             idep[i][1] != -1f0 &&
             p[1] >= par.domain_min_x && p[1] <= dom_max_x &&
             p[2] >= par.domain_min_y && p[2] <= dom_max_y &&
             p[3] >= par.domain_min_z && p[3] <= dom_max_z)
            continue
        end

        cx = floor(Int32, 0.5f0 * (p[1] - par.domain_min_x) * ih)
        cy = floor(Int32, 0.5f0 * (p[2] - par.domain_min_y) * ih)
        cz = floor(Int32, 0.5f0 * (p[3] - par.domain_min_z) * ih)
        base = cx * par.cell_num_y * par.cell_num_z + cy * par.cell_num_z + cz
        key = Int(base + (idep[i][1] < 100f0 ? 0 : par.ncell_one_layer)) + 1

        valid[i] = true
        cxs[i] = cx
        cys[i] = cy
        czs[i] = cz
        next_idx[i] = head[key]
        head[key] = Int32(i)
    end

    q2_scale = ih * ih
    @inbounds for i in 1:np
        valid[i] || continue
        p = pos[i]
        cx = cxs[i]
        cy = cys[i]
        cz = czs[i]
        count_i = Int32(0)
        for dcz = Int32(-1):Int32(1)
            for dcy = Int32(-1):Int32(1)
                for dcx = Int32(-1):Int32(1)
                    ncx, ncy, ncz = cx + dcx, cy + dcy, cz + dcz
                    if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                        base = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz
                        for off in (Int32(0), par.ncell_one_layer)
                            j = head[Int(base + off) + 1]
                            while j != 0
                                if i != j
                                    pj = pos[j]
                                    dx = p[1] - pj[1]
                                    dy = p[2] - pj[2]
                                    dz = p[3] - pj[3]
                                    dist2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
                                    if dist2 * q2_scale < q2_limit
                                        count_i += 1
                                    end
                                end
                                j = next_idx[j]
                            end
                        end
                    end
                end
            end
        end
        counts[i] = count_i
    end

    return counts
end

function _gather_float4_component(data::Vector{Float4}, idxs::AbstractVector{Int}, comp::Int)
    out = Vector{Float32}(undef, length(idxs))
    @inbounds for k in eachindex(idxs)
        out[k] = Float32(data[idxs[k]][comp])
    end
    return out
end

function _gather_float32_values(data::Vector{Float32}, idxs::AbstractVector{Int})
    out = Vector{Float32}(undef, length(idxs))
    @inbounds for k in eachindex(idxs)
        out[k] = data[idxs[k]]
    end
    return out
end

function _gather_int32_values(data::Vector{Int32}, idxs::AbstractVector{Int})
    out = Vector{Int32}(undef, length(idxs))
    @inbounds for k in eachindex(idxs)
        out[k] = data[idxs[k]]
    end
    return out
end

function _gather_float4_xyz(data::Vector{Float4}, idxs::AbstractVector{Int})
    out = Vector{Float32}(undef, 3 * length(idxs))
    @inbounds for k in eachindex(idxs)
        idx = idxs[k]
        base = 3 * (k - 1)
        out[base + 1] = Float32(data[idx][1])
        out[base + 2] = Float32(data[idx][2])
        out[base + 3] = Float32(data[idx][3])
    end
    return out
end

function _gather_displacement_xyz(pos::Vector{Float4}, idep::Vector{Float4}, idxs::AbstractVector{Int}, ref_pos_by_oid)
    out = Vector{Float32}(undef, 3 * length(idxs))
    @inbounds for k in eachindex(idxs)
        idx = idxs[k]
        oid = Int32(round(idep[idx][3]))
        pref = get(ref_pos_by_oid, oid, pos[idx])
        base = 3 * (k - 1)
        out[base + 1] = Float32(pos[idx][1] - pref[1])
        out[base + 2] = Float32(pos[idx][2] - pref[2])
        out[base + 3] = Float32(pos[idx][3] - pref[3])
    end
    return out
end

function _gather_tensor6(data::Vector{Tensor2D}, idxs::AbstractVector{Int})
    out = Vector{Float32}(undef, 6 * length(idxs))
    @inbounds for k in eachindex(idxs)
        t = data[idxs[k]]
        base = 6 * (k - 1)
        out[base + 1] = t.xx
        out[base + 2] = t.yy
        out[base + 3] = t.zz
        out[base + 4] = t.xy
        out[base + 5] = t.xz
        out[base + 6] = t.yz
    end
    return out
end

function _fill_float4_component!(buf::Vector{Float32}, data::Vector{Float4}, idxs::AbstractVector{Int}, comp::Int)
    resize!(buf, length(idxs))
    @inbounds for k in eachindex(idxs)
        buf[k] = Float32(data[idxs[k]][comp])
    end
    return buf
end

function _fill_float32_values!(buf::Vector{Float32}, data::Vector{Float32}, idxs::AbstractVector{Int})
    resize!(buf, length(idxs))
    @inbounds for k in eachindex(idxs)
        buf[k] = data[idxs[k]]
    end
    return buf
end

function _fill_int32_values!(buf::Vector{Int32}, data::Vector{Int32}, idxs::AbstractVector{Int})
    resize!(buf, length(idxs))
    @inbounds for k in eachindex(idxs)
        buf[k] = data[idxs[k]]
    end
    return buf
end

function _fill_float4_xyz!(buf::Vector{Float32}, data::Vector{Float4}, idxs::AbstractVector{Int})
    resize!(buf, 3 * length(idxs))
    @inbounds for k in eachindex(idxs)
        idx = idxs[k]
        base = 3 * (k - 1)
        buf[base + 1] = Float32(data[idx][1])
        buf[base + 2] = Float32(data[idx][2])
        buf[base + 3] = Float32(data[idx][3])
    end
    return buf
end

function _fill_scaled_float4_xyz!(buf::Vector{Float32}, data::Vector{Float4}, scale_data::Vector{Float4}, idxs::AbstractVector{Int}, scale_comp::Int)
    resize!(buf, 3 * length(idxs))
    @inbounds for k in eachindex(idxs)
        idx = idxs[k]
        scale = Float32(scale_data[idx][scale_comp])
        base = 3 * (k - 1)
        buf[base + 1] = Float32(data[idx][1]) * scale
        buf[base + 2] = Float32(data[idx][2]) * scale
        buf[base + 3] = Float32(data[idx][3]) * scale
    end
    return buf
end

function _fill_normal_inclination!(buf::Vector{Float32}, normal::Vector{Float4}, idxs::AbstractVector{Int})
    resize!(buf, length(idxs))
    @inbounds for k in eachindex(idxs)
        ny = clamp(Float32(normal[idxs[k]][2]), -1f0, 1f0)
        buf[k] = acos(ny)
    end
    return buf
end

function _fill_displacement_xyz!(buf::Vector{Float32}, pos::Vector{Float4}, idep::Vector{Float4}, idxs::AbstractVector{Int}, ref_pos_by_oid)
    resize!(buf, 3 * length(idxs))
    @inbounds for k in eachindex(idxs)
        idx = idxs[k]
        oid = Int32(round(idep[idx][3]))
        pref = get(ref_pos_by_oid, oid, pos[idx])
        base = 3 * (k - 1)
        buf[base + 1] = Float32(pos[idx][1] - pref[1])
        buf[base + 2] = Float32(pos[idx][2] - pref[2])
        buf[base + 3] = Float32(pos[idx][3] - pref[3])
    end
    return buf
end

function _fill_tensor6!(buf::Vector{Float32}, data::Vector{Tensor2D}, idxs::AbstractVector{Int})
    resize!(buf, 6 * length(idxs))
    @inbounds for k in eachindex(idxs)
        t = data[idxs[k]]
        base = 6 * (k - 1)
        buf[base + 1] = t.xx
        buf[base + 2] = t.yy
        buf[base + 3] = t.zz
        buf[base + 4] = t.xy
        buf[base + 5] = t.xz
        buf[base + 6] = t.yz
    end
    return buf
end

function _fill_connectivity!(buf::Vector{Int32}, n::Int)
    resize!(buf, n)
    @inbounds for i in 1:n
        buf[i] = Int32(i - 1)
    end
    return buf
end

function _fill_offsets!(buf::Vector{Int32}, n::Int)
    resize!(buf, n)
    @inbounds for i in 1:n
        buf[i] = Int32(i)
    end
    return buf
end

function _fill_types!(buf::Vector{UInt8}, n::Int)
    resize!(buf, n)
    fill!(buf, UInt8(1))
    return buf
end

function save_to_vtu_binary(
    filename,
    pos,
    vel,
    idep,
    str,
    stn,
    mass,
    neighbor_count,
    idxs::AbstractVector{Int},
    ref_pos_by_oid,
    acc,
    topo_normal,
    topo_quantities,
    topo_basal_velocity,
    topo_entrainment,
    scratch::SaveScratch
)
    n = length(idxs)
    open(filename, "w") do io
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n")
        write(io, "  <UnstructuredGrid>\n")
        write(io, "    <Piece NumberOfPoints=\"$n\" NumberOfCells=\"$n\">\n")
        if topo_normal !== nothing && topo_quantities !== nothing
            write(io, "      <PointData Vectors=\"FlowDepthVector\">\n")
        else
            write(io, "      <PointData>\n")
        end
        
        write(io, "        <DataArray type=\"Float32\" Name=\"Pressure\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, pos, idxs, 4))
        write(io, "\n        </DataArray>\n")
        
        write(io, "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_xyz!(scratch.float32_buf, vel, idxs))
        write(io, "\n        </DataArray>\n")

        if ref_pos_by_oid !== nothing
            write(io, "        <DataArray type=\"Float32\" Name=\"Displacement\" NumberOfComponents=\"3\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_displacement_xyz!(scratch.float32_buf, pos, idep, idxs, ref_pos_by_oid))
            write(io, "\n        </DataArray>\n")
        end

        if acc !== nothing
            write(io, "        <DataArray type=\"Float32\" Name=\"Acceleration\" NumberOfComponents=\"3\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_xyz!(scratch.float32_buf, acc, idxs))
            write(io, "\n        </DataArray>\n")
        end

        if topo_normal !== nothing
            write(io, "        <DataArray type=\"Float32\" Name=\"TopographyNormal\" NumberOfComponents=\"3\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_xyz!(scratch.float32_buf, topo_normal, idxs))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"TopographySupportCount\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_normal, idxs, 4))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"TopographyNormalY\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_normal, idxs, 2))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"TopographyNormalInclination\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_normal_inclination!(scratch.float32_buf, topo_normal, idxs))
            write(io, "\n        </DataArray>\n")
        end

        if topo_quantities !== nothing
            write(io, "        <DataArray type=\"Float32\" Name=\"FlowDepth\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_quantities, idxs, 1))
            write(io, "\n        </DataArray>\n")

            if topo_normal !== nothing
                write(io, "        <DataArray type=\"Float32\" Name=\"FlowDepthVector\" NumberOfComponents=\"3\" format=\"binary\">\n")
                write_vtk_binary(io, _fill_scaled_float4_xyz!(scratch.float32_buf, topo_normal, topo_quantities, idxs, 1))
                write(io, "\n        </DataArray>\n")
            end

            write(io, "        <DataArray type=\"Float32\" Name=\"BasalSpeed\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_quantities, idxs, 2))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"AverageTangentialSpeed\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_quantities, idxs, 3))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"TopographySlopeAngle\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_quantities, idxs, 4))
            write(io, "\n        </DataArray>\n")
        end

        if topo_basal_velocity !== nothing
            write(io, "        <DataArray type=\"Float32\" Name=\"BasalVelocity\" NumberOfComponents=\"3\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_xyz!(scratch.float32_buf, topo_basal_velocity, idxs))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"TopographyExtractionMask\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_basal_velocity, idxs, 4))
            write(io, "\n        </DataArray>\n")
        end

        if topo_entrainment !== nothing
            write(io, "        <DataArray type=\"Float32\" Name=\"EntrainmentRate\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_entrainment, idxs, 1))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"ErosionDepth\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_entrainment, idxs, 2))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"PorePressureCoefficient\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_entrainment, idxs, 3))
            write(io, "\n        </DataArray>\n")

            write(io, "        <DataArray type=\"Float32\" Name=\"EntrainmentLoadStatus\" format=\"binary\">\n")
            write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, topo_entrainment, idxs, 4))
            write(io, "\n        </DataArray>\n")
        end
        
        write(io, "        <DataArray type=\"Float32\" Name=\"Density\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, vel, idxs, 4))
        write(io, "\n        </DataArray>\n")
        
        write(io, "        <DataArray type=\"Float32\" Name=\"Mass\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float32_values!(scratch.float32_buf, mass, idxs))
        write(io, "\n        </DataArray>\n")

        write(io, "        <DataArray type=\"Int32\" Name=\"NeighborCount\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_int32_values!(scratch.int32_buf, neighbor_count, idxs))
        write(io, "\n        </DataArray>\n")
        
        write(io, "        <DataArray type=\"Float32\" Name=\"PlasticStrain\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, idep, idxs, 2))
        write(io, "\n        </DataArray>\n")

        write(io, "        <DataArray type=\"Float32\" Name=\"DeviatoricPlasticStrain\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, idep, idxs, 4))
        write(io, "\n        </DataArray>\n")
        
        write(io, "        <DataArray type=\"Float32\" Name=\"PartID\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, idep, idxs, 1))
        write(io, "\n        </DataArray>\n")

        write(io, "        <DataArray type=\"Float32\" Name=\"OriginalID\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_component!(scratch.float32_buf, idep, idxs, 3))
        write(io, "\n        </DataArray>\n")

        write(io, "        <DataArray type=\"Float32\" Name=\"Stress\" NumberOfComponents=\"6\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_tensor6!(scratch.float32_buf, str, idxs))
        write(io, "\n        </DataArray>\n")

        write(io, "        <DataArray type=\"Float32\" Name=\"Strain\" NumberOfComponents=\"6\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_tensor6!(scratch.float32_buf, stn, idxs))
        write(io, "\n        </DataArray>\n")
        
        write(io, "      </PointData>\n")
        write(io, "      <Points>\n")
        write(io, "         <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_float4_xyz!(scratch.float32_buf, pos, idxs))
        write(io, "\n        </DataArray>\n      </Points>\n")
        write(io, "      <Cells>\n")
        write(io, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_connectivity!(scratch.int32_buf, n))
        write(io, "\n        </DataArray>\n")
        write(io, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_offsets!(scratch.int32_buf, n))
        write(io, "\n        </DataArray>\n")
        write(io, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">\n")
        write_vtk_binary(io, _fill_types!(scratch.uint8_buf, n))
        write(io, "\n        </DataArray>\n      </Cells>\n")
        write(io, "    </Piece>\n")
        write(io, "  </UnstructuredGrid>\n")
        write(io, "</VTKFile>\n")
    end
end

function save_particles(
    project_name::String,
    par::Parameters,
    pos::Vector{Float4},
    vel::Vector{Float4},
    idep::Vector{Float4},
    str::Vector{Tensor2D},
    stn::Vector{Tensor2D},
    mass::Vector{Float32};
    pos_ref::Union{Nothing, Vector{Float4}} = nothing,
    idep_ref::Union{Nothing, Vector{Float4}} = nothing,
    acc::Union{Nothing, Vector{Float4}} = nothing,
    topo_normal::Union{Nothing, Vector{Float4}} = nothing,
    topo_quantities::Union{Nothing, Vector{Float4}} = nothing,
    topo_basal_velocity::Union{Nothing, Vector{Float4}} = nothing,
    topo_entrainment::Union{Nothing, Vector{Float4}} = nothing
)
    t_start = time()
    scratch = _get_save_scratch(project_name, Int(par.np), Int(par.ncell))

    neighbor_count = compute_neighbor_count_host(pos, idep, par, scratch)
    ref_pos_by_oid = nothing
    if pos_ref !== nothing
        if idep_ref === nothing
            error("save_particles requires idep_ref when pos_ref is provided so displacement can be aligned by OriginalID.")
        end
        cache_key = (UInt(objectid(pos_ref)), UInt(objectid(idep_ref)))
        if scratch.ref_pos_cache_key != cache_key
            empty!(scratch.ref_pos_by_oid)
            sizehint!(scratch.ref_pos_by_oid, length(pos_ref))
            @inbounds for i in eachindex(pos_ref)
                oid = Int32(round(idep_ref[i][3]))
                if oid > 0
                    scratch.ref_pos_by_oid[oid] = pos_ref[i]
                end
            end
            scratch.ref_pos_cache_key = cache_key
        end
        ref_pos_by_oid = scratch.ref_pos_by_oid
    end

    for idxs in values(scratch.part_index_map)
        empty!(idxs)
    end
    part_index_map = scratch.part_index_map
    @inbounds for i in 1:par.np
        pid = Int32(round(idep[i][1]))
        push!(get!(part_index_map, pid, Int[]), i)
    end
    unique_ids = sort!(collect(keys(part_index_map)); rev = true)

    t_sorted = time()

    if !isdir(project_name)
        mkpath(project_name)
    end
    
    for k in 1:length(unique_ids)
        uid = unique_ids[k]
        is_boundary = (uid < 100)
        is_material = (uid >= 100)
        idxs = part_index_map[Int32(uid)]

        # PART ID = 0 boundary output mode:
        # 0 => initial-only (skip after output_0), 1 => all-time.
        if uid == 0 && par.boundary_part0_output_mode == 0 && par.output_number > 0
            continue
        end

        if !is_boundary && !is_material
            continue 
        end

        if par.output_format == 1
            fname = joinpath(project_name, "$(uid)_$(par.output_number).dat")
            save_to_dat(fname, pos, vel, idep, str, stn, idxs)
        elseif par.output_format == 2
            fname = joinpath(project_name, "$(uid)_$(par.output_number).vtu")
            save_to_vtu_binary(
                fname,
                pos,
                vel,
                idep,
                str,
                stn,
                mass,
                neighbor_count,
                idxs,
                ref_pos_by_oid,
                acc,
                topo_normal,
                topo_quantities,
                topo_basal_velocity,
                topo_entrainment,
                scratch
            )
        end
    end
    
    t_end = time()
    
    t_sort = t_sorted - t_start
    t_write = t_end - t_sorted
    @printf("Save time: sorting = %.4f, writing = %.4f\n", t_sort, t_write)
    
    ext = par.output_format == 1 ? ".dat" : (par.output_format == 2 ? ".vtu" : ".vtk")
    file_log = joinpath(abspath(project_name), "*_$(par.output_number)$(ext) + $(par.output_number).xml")
    println("Save file: ", file_log)
end

function save_xml(
    project_name::String,
    par::Parameters;
    part_acc_ids::AbstractVector{<:Integer}=Int[],
    part_acc_vals::AbstractVector{Float4}=Float4[]
)
    filename = joinpath(project_name, "output_$(par.output_number).xml")
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "parameters")
    
    function add_p(name, val, type_str)
        c = new_child(xroot, name)
        set_content(c, string(val))
        set_attribute(c, "type", type_str)
    end
    add_p("number-of-particles", par.np, "int")
    add_p("time", par.time, "float")
    add_p("output-number", par.output_number, "int")
    add_p("output-format", par.output_format, "int")
    add_p("viscosity-type", par.viscosity_type, "int")
    add_p("artificial-viscosity-alpha", par.artificial_viscosity_alpha, "float")
    add_p("artificial-viscosity-q1", par.artificial_viscosity_q1, "float")
    add_p("artificial-viscosity-q2", par.artificial_viscosity_q2, "float")
    add_p("momentum-form", par.momentum_form, "int")
    add_p("boundary-velocity-type", par.boundary_velocity_type, "int")
    add_p("wall-velocity-x", par.wall_velocity_x, "float")
    add_p("wall-velocity-y", par.wall_velocity_y, "float")
    add_p("wall-velocity-z", par.wall_velocity_z, "float")
    add_p("initial-damping-duration", par.initial_damping_duration, "float")
    add_p("initial-damping-coefficient", par.initial_damping_coefficient, "float")
    add_p("boundary0-output-mode", par.boundary_part0_output_mode, "int")
    add_p("static-boundary-grid", par.static_boundary_grid, "int")
    add_p("active-boundary-adami", par.active_boundary_adami, "int")
    add_p("depth-footprint-radius", par.depth_footprint_radius, "float")
    add_p("depth-search-max", par.depth_search_max, "float")
    add_p("normal-orientation-radius", par.normal_orientation_radius, "float")
    add_p("entrainment-enabled", par.entrainment_enabled, "int")
    add_p("entrainment-pore-pressure-coefficient", par.entrainment_pore_pressure, "float")
    add_p("entrainment-flow-friction-angle", par.entrainment_flow_friction_angle * (180f0 / PI_F), "float")
    add_p("entrainment-bed-friction-angle", par.entrainment_bed_friction_angle * (180f0 / PI_F), "float")
    add_p("erosion-depth-max", par.erosion_depth_max, "float")
    add_p("entrainment-speed-min", par.entrainment_speed_min, "float")
    add_p("entrainment-rate-max", par.entrainment_rate_max, "float")
    @assert length(part_acc_ids) == length(part_acc_vals)
    for i in eachindex(part_acc_ids)
        c = new_child(xroot, "part-acceleration")
        set_attribute(c, "part-id", string(Int32(part_acc_ids[i])))
        add_x = new_child(c, "acceleration-x"); set_content(add_x, string(Float32(part_acc_vals[i][1]))); set_attribute(add_x, "type", "float")
        add_y = new_child(c, "acceleration-y"); set_content(add_y, string(Float32(part_acc_vals[i][2]))); set_attribute(add_y, "type", "float")
        add_z = new_child(c, "acceleration-z"); set_content(add_z, string(Float32(part_acc_vals[i][3]))); set_attribute(add_z, "type", "float")
    end
    save_file(xdoc, filename)
    free(xdoc)
end

end # module

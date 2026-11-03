module Grid

using CUDA
using ..Types: Parameters, Float4, Tensor2D

export LoquatGrid, neighbor_search!, sort_data!, sort_boundary_data!, sort_material_data!, update_active_boundary!

mutable struct LoquatGrid
    sort_buffer::CuVector{Int64}
    boundary_sort_buffer::CuVector{Int64}
    material_sort_buffer::CuVector{Int64}
    cell_beg::CuVector{Int32}
    cell_end::CuVector{Int32}
    active_boundary_flags::CuVector{Int32}
    active_boundary_indices::CuVector{Int32}
    active_boundary_count::CuVector{Int32}
    boundary_grid_initialized::Bool
    boundary_arrays_sorted::Bool
    active_boundary_count_host::Int32
    
    function LoquatGrid(par::Parameters)
        sort_buffer = CUDA.zeros(Int64, par.np)
        boundary_sort_buffer = CUDA.zeros(Int64, par.npb)
        material_sort_buffer = CUDA.zeros(Int64, par.npm)
        cell_beg = CUDA.fill(Int32(-1), par.ncell)
        cell_end = CUDA.fill(Int32(-1), par.ncell)
        active_boundary_flags = CUDA.zeros(Int32, par.npb)
        active_boundary_indices = CUDA.zeros(Int32, par.npb)
        active_boundary_count = CUDA.zeros(Int32, 1)
        new(
            sort_buffer, boundary_sort_buffer, material_sort_buffer, cell_beg, cell_end,
            active_boundary_flags, active_boundary_indices, active_boundary_count,
            false, false, Int32(0)
        )
    end
end

# 1. Pack: CellID (High 32) | ParticleID (Low 32)
function kernel_pack_particles(sort_buffer, pospres, idep, par)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= par.np
        @inbounds begin
            pos_i = pospres[i]
            invalid = (pos_i[1] < par.domain_min_x) | (pos_i[1] > (par.domain_min_x + par.domain_size_x)) |
                      (pos_i[2] < par.domain_min_y) | (pos_i[2] > (par.domain_min_y + par.domain_size_y)) |
                      (pos_i[3] < par.domain_min_z) | (pos_i[3] > (par.domain_min_z + par.domain_size_z)) |
                      (idep[i][1] == -1f0)

            cell_idx = Int32(0)
            if invalid
                cell_idx = par.ncell_delete
                idep[i] = Float4(-1f0, idep[i][2], idep[i][3], idep[i][4])
            else
                ih = par.i_h
                
                cx = floor(Int32, 0.5f0 * (pos_i[1] - par.domain_min_x) * ih)
                cy = floor(Int32, 0.5f0 * (pos_i[2] - par.domain_min_y) * ih)
                cz = floor(Int32, 0.5f0 * (pos_i[3] - par.domain_min_z) * ih)
                
                base = cx * par.cell_num_y * par.cell_num_z + cy * par.cell_num_z + cz
                cell_idx = ifelse(idep[i][1] < 100f0, base, base + par.ncell_one_layer)
            end
            
            key = UInt64(cell_idx) << 32
            val = UInt64(i)
            sort_buffer[i] = reinterpret(Int64, key | val)
        end
    end
    return nothing
end

function kernel_pack_segment_particles(sort_buffer, pospres, idep, start_idx::Int32, count::Int32, par)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= count
        @inbounds begin
            i = start_idx + idx - Int32(1)
            pos_i = pospres[i]
            invalid = (pos_i[1] < par.domain_min_x) | (pos_i[1] > (par.domain_min_x + par.domain_size_x)) |
                      (pos_i[2] < par.domain_min_y) | (pos_i[2] > (par.domain_min_y + par.domain_size_y)) |
                      (pos_i[3] < par.domain_min_z) | (pos_i[3] > (par.domain_min_z + par.domain_size_z)) |
                      (idep[i][1] == -1f0)

            cell_idx = Int32(0)
            if invalid
                cell_idx = par.ncell_delete
                idep[i] = Float4(-1f0, idep[i][2], idep[i][3], idep[i][4])
            else
                ih = par.i_h

                cx = floor(Int32, 0.5f0 * (pos_i[1] - par.domain_min_x) * ih)
                cy = floor(Int32, 0.5f0 * (pos_i[2] - par.domain_min_y) * ih)
                cz = floor(Int32, 0.5f0 * (pos_i[3] - par.domain_min_z) * ih)

                base = cx * par.cell_num_y * par.cell_num_z + cy * par.cell_num_z + cz
                cell_idx = ifelse(idep[i][1] < 100f0, base, base + par.ncell_one_layer)
            end

            key = UInt64(cell_idx) << 32
            val = UInt64(i)
            sort_buffer[idx] = reinterpret(Int64, key | val)
        end
    end
    return nothing
end

# 2. Find Beg/End
function kernel_find_beg_end_packed(cell_beg, cell_end, sort_buffer, np)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= np
        @inbounds begin
            packed_curr = reinterpret(UInt64, sort_buffer[i])
            cell_curr = Int32(packed_curr >> 32)
            idx_curr = cell_curr + 1
            
            if i == 1
                cell_beg[idx_curr] = i
            else
                packed_prev = reinterpret(UInt64, sort_buffer[i-1])
                cell_prev = Int32(packed_prev >> 32)
                
                if cell_curr != cell_prev
                    cell_beg[idx_curr] = i
                    cell_end[cell_prev + 1] = i
                end
            end
            
            if i == np
                cell_end[idx_curr] = i + 1
            end
        end
    end
    return nothing
end

function kernel_find_beg_end_packed_segment(cell_beg, cell_end, sort_buffer, np, sorted_offset::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= np
        @inbounds begin
            packed_curr = reinterpret(UInt64, sort_buffer[i])
            cell_curr = Int32(packed_curr >> 32)
            idx_curr = cell_curr + 1
            sorted_i = Int32(i) + sorted_offset

            if i == 1
                cell_beg[idx_curr] = sorted_i
            else
                packed_prev = reinterpret(UInt64, sort_buffer[i-1])
                cell_prev = Int32(packed_prev >> 32)

                if cell_curr != cell_prev
                    cell_beg[idx_curr] = sorted_i
                    cell_end[cell_prev + 1] = sorted_i
                end
            end

            if i == np
                cell_end[idx_curr] = sorted_i + Int32(1)
            end
        end
    end
    return nothing
end

# 3. Gather
function kernel_gather_packed(data, cache, sort_buffer, np)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= np
        @inbounds begin
            packed = reinterpret(UInt64, sort_buffer[i])
            src_idx = Int32(packed & 0xFFFFFFFF)
            data[i] = cache[src_idx]
        end
    end
    return nothing
end

function kernel_gather_packed_segment(data, cache, sort_buffer, sorted_offset::Int32, np)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= np
        @inbounds begin
            packed = reinterpret(UInt64, sort_buffer[i])
            src_idx = Int32(packed & 0xFFFFFFFF)
            dst_idx = Int32(i) + sorted_offset
            data[dst_idx] = cache[src_idx]
        end
    end
    return nothing
end

function kernel_copy_segment(cache, data, start_idx::Int32, np)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= np
        @inbounds begin
            src_idx = start_idx + Int32(i) - Int32(1)
            cache[src_idx] = data[src_idx]
        end
    end
    return nothing
end

function kernel_mark_active_boundary_cells(active_boundary_flags, cell_beg, cell_end, par)
    base_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if base_idx <= par.ncell_one_layer
        @inbounds begin
            material_cell = base_idx + par.ncell_one_layer
            mat_beg = cell_beg[material_cell]
            if mat_beg == -1
                return nothing
            end

            zero_base = Int32(base_idx - 1)
            yz = par.cell_num_y * par.cell_num_z
            cx = div(zero_base, yz)
            rem_xy = zero_base - cx * yz
            cy = div(rem_xy, par.cell_num_z)
            cz = rem_xy - cy * par.cell_num_z

            for dcz = Int32(-1):Int32(1)
                for dcy = Int32(-1):Int32(1)
                    for dcx = Int32(-1):Int32(1)
                        ncx, ncy, ncz = cx + dcx, cy + dcy, cz + dcz
                        if ncx >= 0 && ncx < par.cell_num_x && ncy >= 0 && ncy < par.cell_num_y && ncz >= 0 && ncz < par.cell_num_z
                            boundary_cell = ncx * par.cell_num_y * par.cell_num_z + ncy * par.cell_num_z + ncz + Int32(1)
                            b_beg = cell_beg[boundary_cell]
                            b_end = cell_end[boundary_cell]
                            if b_beg != -1
                                for j = b_beg:(b_end - 1)
                                    if j <= par.npb
                                        active_boundary_flags[j] = Int32(1)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function kernel_collect_active_boundary(active_boundary_indices, active_boundary_count, active_boundary_flags, npb::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= npb
        @inbounds begin
            if active_boundary_flags[i] != 0
                slot0 = CUDA.@atomic active_boundary_count[1] += Int32(1)
                active_boundary_indices[slot0 + Int32(1)] = i
            end
        end
    end
    return nothing
end

function neighbor_search!(grid::LoquatGrid, par::Parameters, pospres::CuVector{Float4}, idep::CuVector{Float4})
    if par.static_boundary_grid != 0 && par.npb > 0
        neighbor_search_split!(grid, par, pospres, idep)
        return nothing
    end

    threads = 256
    blocks = cld(par.np, threads)

    @cuda threads=threads blocks=blocks kernel_pack_particles(grid.sort_buffer, pospres, idep, par)
    
    CUDA.sort!(grid.sort_buffer)

    grid.cell_beg .= Int32(-1)
    grid.cell_end .= Int32(-1)
    @cuda threads=threads blocks=blocks kernel_find_beg_end_packed(grid.cell_beg, grid.cell_end, grid.sort_buffer, par.np)
end

function neighbor_search_split!(grid::LoquatGrid, par::Parameters, pospres::CuVector{Float4}, idep::CuVector{Float4})
    threads = 256

    if !grid.boundary_grid_initialized
        if par.npb > 0
            blocks_b = cld(par.npb, threads)
            @cuda threads=threads blocks=blocks_b kernel_pack_segment_particles(
                grid.boundary_sort_buffer, pospres, idep, Int32(1), par.npb, par
            )
            CUDA.sort!(grid.boundary_sort_buffer)
        end
        grid.boundary_grid_initialized = true
    end

    if par.npm > 0
        blocks_m = cld(par.npm, threads)
        @cuda threads=threads blocks=blocks_m kernel_pack_segment_particles(
            grid.material_sort_buffer, pospres, idep, par.npb + Int32(1), par.npm, par
        )
        CUDA.sort!(grid.material_sort_buffer)
    else
        blocks_m = 0
    end

    grid.cell_beg .= Int32(-1)
    grid.cell_end .= Int32(-1)

    if par.npb > 0
        blocks_b = cld(par.npb, threads)
        @cuda threads=threads blocks=blocks_b kernel_find_beg_end_packed_segment(
            grid.cell_beg, grid.cell_end, grid.boundary_sort_buffer, par.npb, Int32(0)
        )
    end
    if par.npm > 0
        @cuda threads=threads blocks=blocks_m kernel_find_beg_end_packed_segment(
            grid.cell_beg, grid.cell_end, grid.material_sort_buffer, par.npm, par.npb
        )
    end
    return nothing
end

function update_active_boundary!(grid::LoquatGrid, par::Parameters)
    if par.active_boundary_adami == 0 || par.static_boundary_grid == 0 || par.npb <= 0 || par.npm <= 0
        grid.active_boundary_count_host = par.npb
        return nothing
    end

    threads = 256
    grid.active_boundary_flags .= Int32(0)
    grid.active_boundary_count .= Int32(0)

    blocks_c = cld(par.ncell_one_layer, threads)
    @cuda threads=threads blocks=blocks_c kernel_mark_active_boundary_cells(
        grid.active_boundary_flags, grid.cell_beg, grid.cell_end, par
    )

    blocks_b = cld(par.npb, threads)
    @cuda threads=threads blocks=blocks_b kernel_collect_active_boundary(
        grid.active_boundary_indices, grid.active_boundary_count, grid.active_boundary_flags, par.npb
    )
    grid.active_boundary_count_host = Array(grid.active_boundary_count)[1]
    return nothing
end

function sort_data!(grid::LoquatGrid, data::CuVector{T}, cache::CuVector{T}) where T
    copyto!(cache, data)
    threads = 256
    blocks = cld(length(data), threads)
    @cuda threads=threads blocks=blocks kernel_gather_packed(data, cache, grid.sort_buffer, length(data))
end

function sort_boundary_data!(grid::LoquatGrid, data::CuVector{T}, cache::CuVector{T}, par::Parameters) where T
    par.npb <= 0 && return nothing
    threads = 256
    blocks = cld(par.npb, threads)
    @cuda threads=threads blocks=blocks kernel_copy_segment(cache, data, Int32(1), par.npb)
    @cuda threads=threads blocks=blocks kernel_gather_packed_segment(
        data, cache, grid.boundary_sort_buffer, Int32(0), par.npb
    )
    return nothing
end

function sort_material_data!(grid::LoquatGrid, data::CuVector{T}, cache::CuVector{T}, par::Parameters) where T
    par.npm <= 0 && return nothing
    threads = 256
    blocks = cld(par.npm, threads)
    @cuda threads=threads blocks=blocks kernel_copy_segment(cache, data, par.npb + Int32(1), par.npm)
    @cuda threads=threads blocks=blocks kernel_gather_packed_segment(
        data, cache, grid.material_sort_buffer, par.npb, par.npm
    )
    return nothing
end

end # module

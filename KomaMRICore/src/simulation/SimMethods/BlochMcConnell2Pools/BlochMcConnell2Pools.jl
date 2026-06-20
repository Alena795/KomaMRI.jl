"""Stores preallocated structs for BlochMcConnell CPU simulations."""
struct BlochMcConnell2PoolsPrealloc{T} <: PreallocResult{T}
    M::Mag2Pools{T}
    Bza_old::AbstractVector{T}
    Bzb_old::AbstractVector{T}
    Bza_new::AbstractVector{T}
    Bzb_new::AbstractVector{T}
    ϕa::AbstractVector{T}
    ϕb::AbstractVector{T}
    Rot::Spinor{T}
    ΔBza::AbstractVector{T}
    ΔBzb::AbstractVector{T}
    Ab
    DC
    D1
    D2
    D3
    V
end

Base.view(p::BlochMcConnell2PoolsPrealloc, i::UnitRange) = begin
    @views BlochMcConnell2PoolsPrealloc(
        p.M[i],
        p.Bza_old[i],
        p.Bzb_old[i],
        p.Bza_new[i],
        p.Bzb_new[i],
        p.ϕa[i],
        p.ϕb[i],
        p.Rot[i],
        p.ΔBza[i],
        p.ΔBzb[i],
        p.Ab[:, :, i],
        p.DC[:, :, i],
        p.D1[:, i],
        p.D2[:, i],
        p.D3[:, i],
        p.V[:, :, i]
    )
end

"""
Preallocates arrays for BlochMcConnell 2-pool CPU simulations.
"""
function prealloc(
    sim_method::BlochMcConnell2Pools,
    backend::KA.CPU,
    obj::Phantom2Pools{T},
    M::Mag2Pools{T},
    max_block_length::Integer,
    groupsize
) where {T<:Real}
    
    Ab, DC, D1, D2, D3, V = arrays_relax_exchange(obj, sim_method)
    
    return BlochMcConnell2PoolsPrealloc(
        Mag2Pools(
        Mag(similar(M.Ma.xy),similar(M.Ma.z)),
        Mag(similar(M.Mb.xy),similar(M.Mb.z))
        ),

        similar(obj.x), 
        similar(obj.x), 
        similar(obj.x), 
        similar(obj.x), 
        similar(obj.x), 
        similar(obj.x), 

        Spinor(
            similar(M.Ma.xy),
            similar(M.Ma.xy)
        ),

        obj.Δwa ./ T(2π * γ),
        obj.Δwb ./ T(2π * γ),
        Ab,
        DC,
        D1,
        D2,
        D3,
        V
    )
end



"""
    run_spin_excitation!(...)

Bloch–McConnell 2-pool excitation simulation optimized for CPU execution.
"""
function run_spin_excitation!(
    p::Phantom2Pools{T},
    seq::DiscreteSequence{T},
    sig::AbstractArray{Complex{T}},
    M::Mag2Pools{T},
    sim_method::BlochMcConnell2Pools,
    groupsize,
    backend::KA.CPU,
    prealloc::BlochMcConnell2PoolsPrealloc
) where {T<:Real}

    # println("entering Excit")
    # Shared preallocated arrays
    Bz = prealloc.Bza_old
    B  = prealloc.Bza_new
    φ  = prealloc.ϕa
    α  = prealloc.Rot.α
    β  = prealloc.Rot.β
    Maux_xy = prealloc.M.Ma.xy
    Maux_z  = prealloc.M.Ma.z
    ΔBza = prealloc.ΔBza
    ΔBzb = prealloc.ΔBzb
    
    # Precomputing static arrays required for constructing the relax-exchange operator 
    Ab = prealloc.Ab
    DC = prealloc.DC
    D1 = prealloc.D1
    D2 = prealloc.D2
    D3 = prealloc.D3
    V  = prealloc.V
    
    # Ab, DC, D1, D2, D3, V = arrays_relax_exchange(p, sim_method)

    # println("Types and size of seq.deltat before the loop in Excit : ")
    # @show typeof(seq.Δt)
    # @show size(seq.Δt)
    # @show typeof(seq.Δt[1, :])
    sample = 1
    # Simulation
    for i in eachindex(seq.Δt)

        # Motion
        x, y, z = get_spin_coords(p.motion,p.x, p.y, p.z,seq.t[i])
        
        # Pool A rotation 
        @. Bz = (seq.Gx[i] * x + seq.Gy[i] * y + seq.Gz[i] * z) + ΔBza - seq.Δf[i] / T(γ)
        @. B = sqrt(abs(seq.B1[i])^2 + abs(Bz)^2)
        @. B[B == 0] = eps(T)
        @. φ = T(-π * γ) * (B * seq.Δt[i])
        @. α = cos(φ) - Complex{T}(im) *(Bz / B) *sin(φ)
        @. β = -Complex{T}(im) *(seq.B1[i] / B) *sin(φ)
        mul!(Spinor(α, β), M.Ma, Maux_xy, Maux_z)

        # Pool B rotation
        @. Bz = (seq.Gx[i] * x + seq.Gy[i] * y + seq.Gz[i] * z) + ΔBzb - seq.Δf[i] / T(γ)
        @. B = sqrt(abs(seq.B1[i])^2 + abs(Bz)^2)
        @. B[B == 0] = eps(T)
        @. φ = T(-π * γ) * (B * seq.Δt[i])
        @. α = cos(φ) - Complex{T}(im) *(Bz / B) *sin(φ)
        @. β = -Complex{T}(im) *(seq.B1[i] / B) *sin(φ)
        mul!(Spinor(α, β),M.Mb,Maux_xy,Maux_z)

        # println("Before relax_exchange in Excit : s.dt =")
        # @show typeof(s.Δt)
        # @show s.Δt 
        # Relaxation + exchange
        relax_exchange_operator!(M, sim_method, seq.Δt[i], Ab, DC, D1, D2, D3, V)
        
        # Relaxation + exchange 
        # relax_exchange_operator!(M, sim_method, s.Δt, Ab, DC, D1, D2, D3, V)
        
        if seq.ADC[i + 1] # ADC at the end of the time step
            sig[sample] = sum(M.Ma.xy + M.Mb.xy) 
            sample += 1
        end
    end
    # @show maximum(abs.(M.Ma.xy))
    # @show maximum(abs.(M.Mb.xy))
    # @show pointer(M.Ma.xy)
    # println("exiting Excit")
    return nothing
end


function arrays_relax_exchange(p::Phantom2Pools{T}, sim_method::BlochMcConnell2Pools) where {T<:Real}
    
    Nspins = length(p)

    # Tensors containing arrays at each spin position
    Ab_all    = zeros(6, 1, Nspins)
    DC_all    = zeros(6, 6, Nspins)
    D1_all    = zeros(2, Nspins)
    D2_all    = zeros(2, Nspins)
    D3_all    = zeros(2, Nspins)
    V_all     = zeros(6, 6, Nspins)
    
    for i in 1:Nspins
    
        # Construction of matrix A and vector b
        A = zeros(6,6) 
        b = zeros(6,1)

        b[3] = p.ρa[i] / p.T1a[i] 
        b[6] = p.ρb[i] / p.T1b[i] 

        # Diagonal terms 
        A[1,1] = -1/p.T2a[i] - p.kab[i]
        A[2,2] = -1/p.T2a[i] - p.kab[i]
        A[3,3] = -1/p.T1a[i] - p.kab[i]
        A[4,4] = -1/p.T2b[i] - p.kba[i]
        A[5,5] = -1/p.T2b[i] - p.kba[i]
        A[6,6] = -1/p.T1b[i] - p.kba[i]

        # Off - diagonal
        A[4,1] = p.kab[i]
        A[5,2] = p.kab[i]  
        A[6,3] = p.kab[i]
        A[1,4] = p.kba[i]
        A[2,5] = p.kba[i]
        A[3,6] = p.kba[i]

        # Computing A^(-1)*b 
        if det(A) == 0
            println("A not invertible, no relaxation and no coupling is assumed")
            Ab = zeros(6,1)
            println("Ab = ", Ab)
        else
            Ab = A \ b
        end

        # Blocks of A in the basis where A is block-diagonal : At1, At2 = At1, At3
        A_t1 = A[[1,4],[1,4]]
        A_t3 = A[[3,6],[3,6]]

        # Eigenvector matrix Vi of block ATi (i=1,2,3), Eigenvalue matrices Di
        F1 = eigen(A_t1)
        V1 = F1.vectors
        D1 = F1.values
        V2 = V1 # since At2 = At1
        D2 = D1
        F3 = eigen(A_t3)
        V3 = F3.vectors
        D3 = F3.values

        # Change of basis matrix V definition (between canonical and eigenbasis)
        V = zeros(6,6)
        V[1,1], V[4,1] = V1[1,1], V1[2,1]
        V[1,4], V[4,4] = V1[1,2], V1[2,2]

        V[2,2], V[5,2] = V2[1,1], V2[2,1]
        V[2,5], V[5,5] = V2[1,2], V2[2,2]

        V[3,3], V[6,3] = V3[1,1], V3[2,1]
        V[3,6], V[6,6] = V3[1,2], V3[2,2]

        detA = [
            (V1[1,1]*V1[2,2] - V1[2,1]*V1[1,2]),

            (V2[1,1]*V2[2,2] - V2[2,1]*V2[1,2]),

            (V3[1,1]*V3[2,2] - V3[2,1]*V3[1,2])]

        DC = zeros(6, 6)

        DC[1,1] = 1/detA[1]*(V1[2,2])
        DC[1,4] = 1/detA[1]*(-V1[1,2])
        DC[2,1] = 1/detA[1]*(-V1[2,1])
        DC[2,4] = 1/detA[1]*(V1[1,1])

        DC[3,2] = 1/detA[2]*(V2[2,2])
        DC[3,5] = 1/detA[2]*(-V2[1,2])
        DC[4,2] = 1/detA[2]*(-V1[2,1])
        DC[4,5] = 1/detA[2]*(V1[1,1])

        DC[5,3] = 1/detA[3]*(V3[2,2])
        DC[5,6] = 1/detA[3]*(-V3[1,2])
        DC[6,3] = 1/detA[3]*(-V3[2,1])
        DC[6,6] = 1/detA[3]*(V3[1,1])

        DC[isnan.(DC)] .= 0
        DC[isinf.(DC)] .= 0

        # for (name, x) in [
        #     ("Ab", Ab),
        #     ("DC", DC),
        #     ("D1", D1),
        #     ("D2", D2),
        #     ("D3", D3),
        #     ("V", V)
        # ]
        #     println("$name: type=$(typeof(x)), size=$(size(x))")
        # end

        Ab_all[:,:,i] .= Ab
        DC_all[:,:,i] .= DC
        D1_all[:,i] .= D1
        D2_all[:,i] .= D2
        D3_all[:,i] .= D3
        V_all[:,:,i] .= V
    end  

    return Ab_all, DC_all, D1_all, D2_all, D3_all, V_all
end


function relax_exchange_operator!(
    M::Mag2Pools{T},
    sim_method::BlochMcConnell2Pools,
    dt,
    Ab_all, DC_all, D1_all, D2_all, D3_all, V_all
) where {T<:Real}
    
    @inbounds for i in eachindex(M.Ma.z)

        Ab = @view Ab_all[:,1,i]
        DC = @view DC_all[:,:,i]
        V  = @view V_all[:,:,i]

        e1 = exp(D1_all[1,i] * dt)
        e2 = exp(D1_all[2,i] * dt)
        e3 = exp(D2_all[1,i] * dt)
        e4 = exp(D2_all[2,i] * dt)
        e5 = exp(D3_all[1,i] * dt)
        e6 = exp(D3_all[2,i] * dt)

        x1 = real(M.Ma.xy[i]) + Ab[1]
        x2 = imag(M.Ma.xy[i]) + Ab[2]
        x3 = M.Ma.z[i]        + Ab[3]

        x4 = real(M.Mb.xy[i]) + Ab[4]
        x5 = imag(M.Mb.xy[i]) + Ab[5]
        x6 = M.Mb.z[i]        + Ab[6]

        c1 = DC[1,1]*x1 + DC[1,4]*x4
        c2 = DC[2,1]*x1 + DC[2,4]*x4

        c3 = DC[3,2]*x2 + DC[3,5]*x5
        c4 = DC[4,2]*x2 + DC[4,5]*x5

        c5 = DC[5,3]*x3 + DC[5,6]*x6
        c6 = DC[6,3]*x3 + DC[6,6]*x6

        c1 *= e1
        c2 *= e2
        c3 *= e3
        c4 *= e4
        c5 *= e5
        c6 *= e6

        y1 = V[1,1]*c1 + V[1,4]*c2
        y2 = V[2,2]*c3 + V[2,5]*c4
        y3 = V[3,3]*c5 + V[3,6]*c6

        y4 = V[4,1]*c1 + V[4,4]*c2
        y5 = V[5,2]*c3 + V[5,5]*c4
        y6 = V[6,3]*c5 + V[6,6]*c6

        M.Ma.xy[i] = complex(y1 - Ab[1], y2 - Ab[2])
        M.Ma.z[i]  = y3 - Ab[3]

        M.Mb.xy[i] = complex(y4 - Ab[4], y5 - Ab[5])
        M.Mb.z[i]  = y6 - Ab[6]
    end

    return nothing
end


function run_spin_precession!(
    p::Phantom2Pools{T},
    seq::DiscreteSequence{T},
    sig::AbstractArray{Complex{T}},
    M::Mag2Pools{T},
    sim_method::BlochMcConnell2Pools,
    groupsize,
    backend::KA.CPU,
    prealloc::BlochMcConnell2PoolsPrealloc
) where {T<:Real}
    #Simulation
    #Motion
    # println("entering Precess")
    # @show maximum(abs.(M.Ma.xy))
    # @show maximum(abs.(M.Mb.xy))
    x, y, z = get_spin_coords(p.motion, p.x, p.y, p.z, seq.t[1])
    
    #Initialize arrays
    Bza_old = prealloc.Bza_old
    Bza_new = prealloc.Bza_new
    Bzb_old = prealloc.Bzb_old
    Bzb_new = prealloc.Bzb_new
    ϕa = prealloc.ϕa
    ϕb = prealloc.ϕb
    # Mxya = prealloc.M.Ma.xy
    # Mxyb = prealloc.M.Mb.xy
    ΔBza = prealloc.ΔBza
    ΔBzb = prealloc.ΔBzb
    fill!(ϕa, zero(T))
    fill!(ϕb, zero(T))

    @. Bza_old = x[:,1] * seq.Gx[1] + y[:,1] * seq.Gy[1] + z[:,1] * seq.Gz[1] + ΔBza
    @. Bzb_old = x[:,1] * seq.Gx[1] + y[:,1] * seq.Gy[1] + z[:,1] * seq.Gz[1] + ΔBzb

    # Fill sig[1] if needed
    ADC_idx = 1
    # if (seq.ADC[1])
    #     sig[1] = sum(M.Ma.xy + M.Mb.xy)
    #     ADC_idx += 1
    # end

    # Precomputing static arrays required for constructing the relax-exchange operator 
    Ab = prealloc.Ab
    DC = prealloc.DC
    D1 = prealloc.D1
    D2 = prealloc.D2
    D3 = prealloc.D3
    V  = prealloc.V
    # Ab, DC, D1, D2, D3, V = arrays_relax_exchange(p, sim_method)

    Mtmp = prealloc.M

    t_seq = zero(T) # Time
    for i in eachindex(seq.Δt)
        x, y, z = get_spin_coords(p.motion, p.x, p.y, p.z, seq.t[i + 1])
        t_seq += seq.Δt[i]

        ### Pool A rotation angle computation        
        #Effective Field
        @. Bza_new = x * seq.Gx[i + 1] + y * seq.Gy[i + 1] + z * seq.Gz[i + 1] + ΔBza
        #Rotation
        @. ϕa += (Bza_old + Bza_new) * T(-π * γ) * seq.Δt[i]

        ### Pool B rotation angle computation 
        #Effective Field
        @. Bzb_new = x * seq.Gx[i + 1] + y * seq.Gy[i + 1] + z * seq.Gz[i + 1] + ΔBzb
        #Rotation
        @. ϕb += (Bzb_old + Bzb_new) * T(-π * γ) * seq.Δt[i]

        #Acquired Signal
        if seq.ADC[i + 1]
            # println("Entering adc condition")
            #Relaxation
            Mtmp.Ma.xy .= M.Ma.xy
            Mtmp.Ma.z  .= M.Ma.z
            Mtmp.Mb.xy .= M.Mb.xy
            Mtmp.Mb.z  .= M.Mb.z
            relax_exchange_operator!(Mtmp, sim_method, t_seq, Ab, DC, D1, D2, D3, V)
            @. Mtmp.Ma.xy = Mtmp.Ma.xy * cis(ϕa)
            @. Mtmp.Mb.xy = Mtmp.Mb.xy * cis(ϕb) # should even remove Mxya and Mxyb here, not needed

            #Reset Spin-State (Magnetization). Only for FlowPath
            outflow_spin_reset!(Mtmp.Ma.xy, seq.t[i + 1], p.motion)
            outflow_spin_reset!(Mtmp.Mb.xy, seq.t[i + 1], p.motion)
            # @show maximum(abs.(M_out.Ma.z))
            # @show maximum(abs.(M_out.Mb.z))
            @show maximum(abs.(Mtmp.Ma.xy))
            # @show maximum(abs.(Mxyb))
            sig[ADC_idx] = sum(Mtmp.Ma.xy + Mtmp.Mb.xy) 
            ADC_idx += 1
            # println("Exiting adc condition")
        end

        Bza_old, Bza_new = Bza_new, Bza_old
        Bzb_old, Bzb_new = Bzb_new, Bzb_old
    end

    #Final Spin-State
    relax_exchange_operator!(M, sim_method, t_seq, Ab, DC, D1, D2, D3, V)
    @. M.Ma.xy = M.Ma.xy * cis(ϕa)
    @. M.Mb.xy = M.Mb.xy * cis(ϕb)
 
    #Reset Spin-State (Magnetization). Only for FlowPath
    outflow_spin_reset!(M.Ma,  seq.t', p.motion; replace_by=p.ρa)
    outflow_spin_reset!(M.Mb,  seq.t', p.motion; replace_by=p.ρb)
    # println("exiting Precess")
    return nothing
end

# write_ogrid_deformed: same O-grid, but the obstacle boundary is a smooth DEFORMED circle
# r(θ)=R+Σ(Aₙ cos nθ + Bₙ sin nθ), drawn as 4 Splines (corner-free, "circular-ish"). Full
# perimeter channel: source=entire left wall, drain=entire right wall, probes top/bottom.
using Printf
function write_ogrid_deformed(path::AbstractString; L=1.8, W=1.4, xc=0.9, yc=0.15, R=0.35,
                              A=Float64[], B=Float64[], ring=0.06, xPL=0.35, xPR=1.45,
                              h=0.06, narc=8)
    W2 = W/2; nmax = max(length(A), length(B))
    Ap = vcat(A, zeros(max(0,nmax-length(A)))); Bp = vcat(B, zeros(max(0,nmax-length(B))))
    rθ(t) = R + (nmax==0 ? 0.0 : sum(Ap[n]*cos(n*t)+Bp[n]*sin(n*t) for n in 1:nmax))
    θf = range(0, 2π, length=361); rmax = maximum(rθ.(θf))
    s = rmax + ring
    X = sort(unique([0.0, xPL, xc-s, xc+s, xPR, L]))
    Y = sort(unique([-W2, yc-s, yc+s, W2]))
    nx = length(X)-1; ny = length(Y)-1
    ci = findfirst(i -> X[i]≈xc-s && X[i+1]≈xc+s, 1:nx)
    cj = findfirst(j -> Y[j]≈yc-s && Y[j+1]≈yc+s, 1:ny)
    (ci===nothing || cj===nothing) && error("box edges not on the grid — check params")
    cprobe = [i for i in 1:nx if X[i] >= xPL-1e-9 && X[i+1] <= xPR+1e-9]
    ncx = [max(2, round(Int,(X[i+1]-X[i])/h)+1) for i in 1:nx]
    ncy = [max(2, round(Int,(Y[j+1]-Y[j])/h)+1) for j in 1:ny]
    pid(i,j)=(j-1)*(nx+1)+i; HL(i,j)=1000+(j-1)*nx+i; VL(i,j)=2000+(j-1)*(nx+1)+i
    ishole(i,j)=(i==ci && j==cj)
    nbox = ncx[ci]; nrad = max(2, round(Int, ring/h)+1)
    P(t) = (xc + rθ(t)*cos(t), yc + rθ(t)*sin(t))
    open(path,"w") do io
        for j in 1:ny+1, i in 1:nx+1; @printf(io,"Point(%d)={%.6f,%.6f,0,1};\n", pid(i,j), X[i], Y[j]); end
        for j in 1:ny+1, i in 1:nx; @printf(io,"Line(%d)={%d,%d};\n", HL(i,j), pid(i,j), pid(i+1,j)); end
        for j in 1:ny, i in 1:nx+1; @printf(io,"Line(%d)={%d,%d};\n", VL(i,j), pid(i,j), pid(i,j+1)); end
        for j in 1:ny+1, i in 1:nx; @printf(io,"Transfinite Curve{%d}=%d;\n", HL(i,j), ncx[i]); end
        for j in 1:ny, i in 1:nx+1; @printf(io,"Transfinite Curve{%d}=%d;\n", VL(i,j), ncy[j]); end
        # deformed obstacle corners (θ=225,315,45,135) → 501..504
        θc = (5π/4, 7π/4, π/4, 3π/4)
        for (k,t) in enumerate(θc); x,y=P(t); @printf(io,"Point(%d)={%.6f,%.6f,0,1};\n", 500+k, x, y); end
        # 4 spline arcs through sampled deformed points
        arcs = ((501,502,5π/4,7π/4,510,601),(502,503,7π/4,9π/4,520,602),
                (503,504,π/4,3π/4,530,603),(504,501,3π/4,5π/4,540,604))
        for (c0,c1,t0,t1,base,spl) in arcs
            ids=[c0]
            for m in 1:narc; t=t0+(t1-t0)*m/(narc+1); x,y=P(t); @printf(io,"Point(%d)={%.6f,%.6f,0,1};\n",base+m,x,y); push!(ids,base+m); end
            push!(ids,c1); @printf(io,"Spline(%d)={%s};\n", spl, join(ids,","))
        end
        BL=pid(ci,cj); BR=pid(ci+1,cj); TR=pid(ci+1,cj+1); TL=pid(ci,cj+1)
        @printf(io,"Line(611)={%d,501};\n",BL); @printf(io,"Line(612)={%d,502};\n",BR)
        @printf(io,"Line(613)={%d,503};\n",TR); @printf(io,"Line(614)={%d,504};\n",TL)
        for a in (601,602,603,604); @printf(io,"Transfinite Curve{%d}=%d;\n", a, nbox); end
        for sp in (611,612,613,614); @printf(io,"Transfinite Curve{%d}=%d;\n", sp, nrad); end
        @printf(io,"Curve Loop(3900)={%d,612,-601,-611};\n", HL(ci,cj))
        @printf(io,"Curve Loop(3901)={%d,613,-602,-612};\n", VL(ci+1,cj))
        @printf(io,"Curve Loop(3902)={%d,614,-603,-613};\n", -HL(ci,cj+1))
        @printf(io,"Curve Loop(3903)={%d,611,-604,-614};\n", -VL(ci,cj))
        for (sid,cs) in ((3900,(BL,BR,502,501)),(3901,(BR,TR,503,502)),(3902,(TR,TL,504,503)),(3903,(TL,BL,501,504)))
            @printf(io,"Plane Surface(%d)={%d};\n", sid, sid); @printf(io,"Transfinite Surface{%d}={%d,%d,%d,%d};\n", sid, cs...)
        end
        sids=Int[]
        for j in 1:ny, i in 1:nx
            ishole(i,j) && continue
            sid=3000+(j-1)*nx+i; push!(sids,sid)
            @printf(io,"Curve Loop(%d)={%d,%d,%d,%d};\n", sid, HL(i,j), VL(i+1,j), -HL(i,j+1), -VL(i,j))
            @printf(io,"Plane Surface(%d)={%d};\n", sid, sid); @printf(io,"Transfinite Surface{%d};\n", sid)
        end
        alls=vcat(sids,3900,3901,3902,3903)
        println(io,"Recombine Surface{",join(alls,","),"};")
        println(io,"Physical Surface(\"domain\")={",join(alls,","),"};")
        source=[VL(1,j) for j in 1:ny]; drain=[VL(nx+1,j) for j in 1:ny]
        probeA=[HL(i,ny+1) for i in cprobe]; probeB=[HL(i,1) for i in cprobe]
        special=Set(vcat(source,drain,probeA,probeB)); walls=Int[]
        for i in 1:nx, j in (1,ny+1); id=HL(i,j); id in special || push!(walls,id); end
        pc(name,ids)=@printf(io,"Physical Curve(\"%s\")={%s};\n", name, join(ids,","))
        pc("contact_source",source); pc("contact_drain",drain)
        pc("probe_A",probeA); pc("probe_B",probeB); pc("walls",walls); pc("obstacle",[601,602,603,604])
    end
    return path, (; nx, ny, ci, cj, R, s, rmax, gap_top=W2-(yc+rmax), gap_bot=W2+(yc-rmax),
                  gap_x=min(xc-rmax, L-(xc+rmax)))
end

# ogrid_geo.jl — STRUCTURED mesh for a channel with a floating CIRCULAR obstacle, via a
# true O-GRID: the circle is wrapped by a ring of 4 curved transfinite quads connecting
# it to a square box; outside the box the channel is filled with Cartesian transfinite
# blocks (same decomposition as blockstruct_geo). Everything is transfinite quads and
# boundary-conforming ⇒ reg-GMRES converges, and the obstacle is a genuine SMOOTH circle
# (no corners), unlike the square block-hole.
#
# Contacts: source=left wall row, drain=right wall row, probe_A=top, probe_B=bottom.
# Physical curves: contact_source, contact_drain, probe_A, probe_B, walls, obstacle
# (obstacle = the 4 circle arcs).
using Printf

# Circle radius R at (xc,yc), wrapped by a square O-grid box of half-side s=R+ring, in the
# FULL channel. Source = entire left wall, drain = entire right wall (so no mid-channel
# split constrains the circle → it can be large and anywhere). Probes = top/bottom segments.
function write_ogrid_geo(path::AbstractString; L=1.6, W=0.8, xc=0.8, yc=0.10,
                         R=0.22, ring=0.05, xPL=0.35, xPR=1.25, h=0.05)
    W2 = W/2; s = R + ring
    X = sort(unique([0.0, xPL, xc-s, xc+s, xPR, L]))
    Y = sort(unique([-W2, yc-s, yc+s, W2]))
    nx = length(X)-1; ny = length(Y)-1
    ci = findfirst(i -> X[i]≈xc-s && X[i+1]≈xc+s, 1:nx)
    cj = findfirst(j -> Y[j]≈yc-s && Y[j+1]≈yc+s, 1:ny)
    (ci===nothing || cj===nothing) && error("box edges not on the grid — check params")
    cprobe = [i for i in 1:nx if X[i] >= xPL-1e-9 && X[i+1] <= xPR+1e-9]
    ncx = [max(2, round(Int,(X[i+1]-X[i])/h)+1) for i in 1:nx]
    ncy = [max(2, round(Int,(Y[j+1]-Y[j])/h)+1) for j in 1:ny]
    pid(i,j) = (j-1)*(nx+1) + i
    HL(i,j)  = 1000 + (j-1)*nx + i
    VL(i,j)  = 2000 + (j-1)*(nx+1) + i
    ishole(i,j) = (i==ci && j==cj)
    nbox = ncx[ci]                          # center block is square ⇒ ncx[ci]==ncy[cj]
    nrad = max(2, round(Int, ring/h)+1)
    r2 = R/sqrt(2)
    open(path,"w") do io
        for j in 1:ny+1, i in 1:nx+1
            @printf(io,"Point(%d)={%.6f,%.6f,0,1};\n", pid(i,j), X[i], Y[j])
        end
        for j in 1:ny+1, i in 1:nx; @printf(io,"Line(%d)={%d,%d};\n", HL(i,j), pid(i,j), pid(i+1,j)); end
        for j in 1:ny, i in 1:nx+1; @printf(io,"Line(%d)={%d,%d};\n", VL(i,j), pid(i,j), pid(i,j+1)); end
        for j in 1:ny+1, i in 1:nx; @printf(io,"Transfinite Curve{%d}=%d;\n", HL(i,j), ncx[i]); end
        for j in 1:ny, i in 1:nx+1; @printf(io,"Transfinite Curve{%d}=%d;\n", VL(i,j), ncy[j]); end
        # --- O-grid inside the center (box) block ---
        @printf(io,"Point(500)={%.6f,%.6f,0,1};\n", xc, yc)                       # circle center
        @printf(io,"Point(501)={%.6f,%.6f,0,1};\n", xc-r2, yc-r2)                 # cBL 225°
        @printf(io,"Point(502)={%.6f,%.6f,0,1};\n", xc+r2, yc-r2)                 # cBR 315°
        @printf(io,"Point(503)={%.6f,%.6f,0,1};\n", xc+r2, yc+r2)                 # cTR 45°
        @printf(io,"Point(504)={%.6f,%.6f,0,1};\n", xc-r2, yc+r2)                 # cTL 135°
        println(io,"Circle(601)={501,500,502};")                                 # bottom arc
        println(io,"Circle(602)={502,500,503};")                                 # right arc
        println(io,"Circle(603)={503,500,504};")                                 # top arc
        println(io,"Circle(604)={504,500,501};")                                 # left arc
        BL=pid(ci,cj); BR=pid(ci+1,cj); TR=pid(ci+1,cj+1); TL=pid(ci,cj+1)
        @printf(io,"Line(611)={%d,501};\n", BL)                                   # spoke BL
        @printf(io,"Line(612)={%d,502};\n", BR)                                   # spoke BR
        @printf(io,"Line(613)={%d,503};\n", TR)                                   # spoke TR
        @printf(io,"Line(614)={%d,504};\n", TL)                                   # spoke TL
        for a in (601,602,603,604); @printf(io,"Transfinite Curve{%d}=%d;\n", a, nbox); end
        for sp in (611,612,613,614); @printf(io,"Transfinite Curve{%d}=%d;\n", sp, nrad); end
        # 4 O-grid quads (box edge, spoke, arc, spoke)
        @printf(io,"Curve Loop(3900)={%d,612,-601,-611};\n", HL(ci,cj))           # bottom
        @printf(io,"Curve Loop(3901)={%d,613,-602,-612};\n", VL(ci+1,cj))         # right
        @printf(io,"Curve Loop(3902)={%d,614,-603,-613};\n", -HL(ci,cj+1))        # top
        @printf(io,"Curve Loop(3903)={%d,611,-604,-614};\n", -VL(ci,cj))          # left
        for (sid,cs) in ((3900,(BL,BR,502,501)),(3901,(BR,TR,503,502)),
                         (3902,(TR,TL,504,503)),(3903,(TL,BL,501,504)))
            @printf(io,"Plane Surface(%d)={%d};\n", sid, sid)
            @printf(io,"Transfinite Surface{%d}={%d,%d,%d,%d};\n", sid, cs...)
        end
        # --- Cartesian outer blocks (skip the center box) ---
        sids = Int[]
        for j in 1:ny, i in 1:nx
            ishole(i,j) && continue
            sid = 3000 + (j-1)*nx + i; push!(sids, sid)
            @printf(io,"Curve Loop(%d)={%d,%d,%d,%d};\n", sid, HL(i,j), VL(i+1,j), -HL(i,j+1), -VL(i,j))
            @printf(io,"Plane Surface(%d)={%d};\n", sid, sid)
            @printf(io,"Transfinite Surface{%d};\n", sid)
        end
        alls = vcat(sids, 3900,3901,3902,3903)
        println(io,"Recombine Surface{",join(alls,","),"};")
        println(io,"Physical Surface(\"domain\")={",join(alls,","),"};")
        obstacle = [601,602,603,604]                                             # the circle
        source   = [VL(1,j)    for j in 1:ny]        # entire left wall
        drain    = [VL(nx+1,j) for j in 1:ny]        # entire right wall
        probeA   = [HL(i,ny+1) for i in cprobe]; probeB = [HL(i,1) for i in cprobe]
        special  = Set(vcat(source,drain,probeA,probeB))
        walls = Int[]
        for i in 1:nx;   for j in (1,ny+1); id=HL(i,j); id in special || push!(walls,id); end; end  # top/bottom only
        pc(name,ids) = @printf(io,"Physical Curve(\"%s\")={%s};\n", name, join(ids,","))
        pc("contact_source",source); pc("contact_drain",drain)
        pc("probe_A",probeA); pc("probe_B",probeB); pc("walls",walls); pc("obstacle",obstacle)
    end
    meta = (; nx, ny, ci, cj, R, s, gap_top=W2-(yc+s), gap_bot=W2+(yc-s),
            gap_x=min(xc-s, L-(xc+s)))
    return path, meta
end

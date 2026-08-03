# blockstruct_geo.jl — shared generator for a fully STRUCTURED (transfinite, all-quad)
# channel with a SQUARE obstacle hole, via a Cartesian block decomposition. The domain
# is split into a grid of rectangular blocks by the x/y lines that bound the obstacle
# and the contacts; the obstacle block is simply omitted (a clean hole with no
# unstructured meshing). Every block is a transfinite quad ⇒ boundary-conforming
# structured mesh where matrix-free reg-GMRES converges (like the vicinity device),
# unlike the unstructured Algorithm-8 hole that scattered.
#
# Contacts: source=left wall (row of the injector), drain=right wall, probe_A=top,
# probe_B=bottom (spanning the obstacle columns). Physical curves: contact_source,
# contact_drain, probe_A, probe_B, walls, obstacle.
using Printf

# Returns (path, ncols, nrows, hole_ij, meta). Square obstacle [xc±s]×[yc±s], offset yc.
function write_blockstruct_geo(path::AbstractString; L=1.6, W=0.8, xc=0.8, yc=0.20,
                               s=0.10, wc=0.12, xPL=0.6, xPR=1.0, h=0.05)
    W2 = W/2
    # split lines (must be strictly increasing; obstacle & contact edges are grid lines)
    X = sort(unique([0.0, xPL, xc-s, xc+s, xPR, L]))
    Y = sort(unique([-W2, -wc/2, wc/2, yc-s, yc+s, W2]))
    nx = length(X)-1; ny = length(Y)-1
    # locate obstacle block and the contact rows/cols
    ci  = findfirst(i -> X[i]≈xc-s && X[i+1]≈xc+s, 1:nx)
    cj  = findfirst(j -> Y[j]≈yc-s && Y[j+1]≈yc+s, 1:ny)
    (ci===nothing || cj===nothing) && error("obstacle edges not on the grid — check params")
    jsrc = findfirst(j -> Y[j]≈-wc/2 && Y[j+1]≈wc/2, 1:ny)     # source/drain row
    cprobe = [i for i in 1:nx if X[i] >= xPL-1e-9 && X[i+1] <= xPR+1e-9]  # probe columns
    # node counts per column/row (shared edges ⇒ conforming)
    ncx = [max(2, round(Int,(X[i+1]-X[i])/h)+1) for i in 1:nx]
    ncy = [max(2, round(Int,(Y[j+1]-Y[j])/h)+1) for j in 1:ny]
    pid(i,j) = (j-1)*(nx+1) + i                    # 1..(nx+1)(ny+1)
    HL(i,j)  = 1000 + (j-1)*nx + i                 # horizontal line P(i,j)->P(i+1,j)
    VL(i,j)  = 2000 + (j-1)*(nx+1) + i             # vertical line   P(i,j)->P(i,j+1)
    ishole(i,j) = (i==ci && j==cj)
    open(path,"w") do io
        for j in 1:ny+1, i in 1:nx+1
            @printf(io,"Point(%d)={%.6f,%.6f,0,1};\n", pid(i,j), X[i], Y[j])
        end
        for j in 1:ny+1, i in 1:nx
            @printf(io,"Line(%d)={%d,%d};\n", HL(i,j), pid(i,j), pid(i+1,j))
        end
        for j in 1:ny, i in 1:nx+1
            @printf(io,"Line(%d)={%d,%d};\n", VL(i,j), pid(i,j), pid(i,j+1))
        end
        # transfinite counts
        for j in 1:ny+1, i in 1:nx; @printf(io,"Transfinite Curve{%d}=%d;\n", HL(i,j), ncx[i]); end
        for j in 1:ny, i in 1:nx+1; @printf(io,"Transfinite Curve{%d}=%d;\n", VL(i,j), ncy[j]); end
        sids = Int[]
        for j in 1:ny, i in 1:nx
            ishole(i,j) && continue
            sid = 3000 + (j-1)*nx + i; push!(sids, sid)
            @printf(io,"Curve Loop(%d)={%d,%d,%d,%d};\n", sid, HL(i,j), VL(i+1,j), -HL(i,j+1), -VL(i,j))
            @printf(io,"Plane Surface(%d)={%d};\n", sid, sid)
            @printf(io,"Transfinite Surface{%d};\n", sid)
        end
        println(io,"Recombine Surface{",join(sids,","),"};")
        println(io,"Physical Surface(\"domain\")={",join(sids,","),"};")
        # physical curves
        obstacle = [HL(ci,cj), HL(ci,cj+1), VL(ci,cj), VL(ci+1,cj)]
        source   = [VL(1,jsrc)]
        drain    = [VL(nx+1,jsrc)]
        probeA   = [HL(i,ny+1) for i in cprobe]
        probeB   = [HL(i,1)    for i in cprobe]
        special  = Set(vcat(obstacle,source,drain,probeA,probeB))
        walls = Int[]
        for j in 1:ny;   for i in (1,nx+1); id=VL(i,j); id in special || push!(walls,id); end; end  # left/right walls
        for i in 1:nx;   for j in (1,ny+1); id=HL(i,j); id in special || push!(walls,id); end; end  # bottom/top walls
        pc(name,ids) = @printf(io,"Physical Curve(\"%s\")={%s};\n", name, join(ids,","))
        pc("contact_source",source); pc("contact_drain",drain)
        pc("probe_A",probeA); pc("probe_B",probeB)
        pc("walls",walls); pc("obstacle",obstacle)
    end
    meta = (; X, Y, nx, ny, ci, cj, jsrc, cprobe,
            gap_top=W2-(yc+s), gap_bot=W2+(yc-s), gap_x=min(xc-s, L-(xc+s)),
            gap_src=(yc-s)-wc/2)
    return path, meta
end

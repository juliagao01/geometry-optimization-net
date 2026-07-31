# pointgeo.jl — shared generator for the mirror-symmetric point-contact mesh.
# 3x3 grid of transfinite blocks; y-partition symmetric about 0 so the mesh is
# exactly mirror-symmetric. Narrow source/drain centered on left/right walls,
# narrow probe_A/probe_B centered on top/bottom walls.
function write_point_geo(path; L=1.0, W=0.6, xPL=0.425, xPR=0.575, wc=0.12, h=0.03)
    xs = (0.0, xPL, xPR, L)
    ys = (-W/2, -wc/2, wc/2, W/2)
    Nx = ntuple(i -> max(2, round(Int,(xs[i+1]-xs[i])/h)+1), 3)
    Ny = ntuple(j -> max(2, round(Int,(ys[j+1]-ys[j])/h)+1), 3)
    pid(i,j) = (i-1)*4 + j
    H(i,j) = (j-1)*3 + i          # horizontal line P(i,j)->P(i+1,j)
    V(i,j) = 12 + (i-1)*3 + j     # vertical line   P(i,j)->P(i,j+1)
    open(path,"w") do io
        Printf.@printf(io,"lc=%.6f;\n", h)
        for i in 1:4, j in 1:4
            Printf.@printf(io,"Point(%d)={%.10f,%.10f,0,lc};\n", pid(i,j), xs[i], ys[j])
        end
        for j in 1:4, i in 1:3
            Printf.@printf(io,"Line(%d)={%d,%d};\n", H(i,j), pid(i,j), pid(i+1,j))
        end
        for i in 1:4, j in 1:3
            Printf.@printf(io,"Line(%d)={%d,%d};\n", V(i,j), pid(i,j), pid(i,j+1))
        end
        sid=0
        for i in 1:3, j in 1:3
            sid+=1
            Printf.@printf(io,"Curve Loop(%d)={%d,%d,%d,%d};\n", sid,
                    H(i,j), V(i+1,j), -H(i,j+1), -V(i,j))
            Printf.@printf(io,"Plane Surface(%d)={%d};\n", sid, sid)
        end
        for i in 1:3, j in 1:4; Printf.@printf(io,"Transfinite Curve{%d}=%d;\n",H(i,j),Nx[i]); end
        for i in 1:4, j in 1:3; Printf.@printf(io,"Transfinite Curve{%d}=%d;\n",V(i,j),Ny[j]); end
        for s in 1:9; Printf.@printf(io,"Transfinite Surface{%d};\n", s); end
        print(io,"Recombine Surface{"); print(io, join(1:9,",")); println(io,"};")
        println(io,"Physical Surface(\"domain\")={",join(1:9,","),"};")
        Printf.@printf(io,"Physical Curve(\"contact_source\")={%d};\n", V(1,2))
        Printf.@printf(io,"Physical Curve(\"contact_drain\") ={%d};\n", V(4,2))
        Printf.@printf(io,"Physical Curve(\"probe_A\")={%d};\n", H(2,4))
        Printf.@printf(io,"Physical Curve(\"probe_B\")={%d};\n", H(2,1))
        walls = [H(1,1),H(3,1),H(1,4),H(3,4),V(1,1),V(1,3),V(4,1),V(4,3)]
        println(io,"Physical Curve(\"walls\")={",join(walls,","),"};")
    end
    return path
end

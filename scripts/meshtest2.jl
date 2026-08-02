using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using Gmsh: gmsh
using Printf
L=1.0;W=0.6;W2=W/2;xPL=0.425;xPR=0.575;wc=0.12;h=0.06;R0=0.12;xc=0.5;n=48
geo=joinpath(@__DIR__,"..","runs","fixed_mesh","mt2.geo")
open(geo,"w") do io
  @printf(io,"lc=%.4f;\n",h)
  P=[(0.0,-W2),(0.0,-wc/2),(0.0,wc/2),(0.0,W2),(xPL,W2),(xPR,W2),(L,W2),(L,wc/2),(L,-wc/2),(L,-W2),(xPR,-W2),(xPL,-W2)]
  for (i,(x,y)) in enumerate(P); @printf(io,"Point(%d)={%.6f,%.6f,0,lc};\n",i,x,y); end
  for i in 1:12; @printf(io,"Line(%d)={%d,%d};\n",i,i,i==12 ? 1 : i+1); end
  println(io,"Curve Loop(1)={",join(1:12,","),"};")
  pid=100; th=range(0,2pi,length=n+1)[1:end-1]
  for t in th; @printf(io,"Point(%d)={%.6f,%.6f,0,lc};\n",pid,xc+R0*cos(t),R0*sin(t)); pid+=1; end
  for k in 0:n-1; @printf(io,"Line(%d)={%d,%d};\n",200+k,100+k,100+(k+1)%n); end
  println(io,"Curve Loop(2)={",join(200:200+n-1,","),"};")
  println(io,"Plane Surface(1)={1,2};")
end
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo)
gmsh.option.setNumber("Mesh.Algorithm",8)           # Frontal-Delaunay for quads
gmsh.option.setNumber("Mesh.RecombineAll",1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm",1)
println("meshing (Algorithm 8)..."); flush(stdout)
gmsh.model.mesh.generate(2)
_,tags,_=gmsh.model.mesh.getElements(2)
@printf("MESHED8: %d elements\n", sum(length.(tags))); flush(stdout)
gmsh.finalize()

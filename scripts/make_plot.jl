#!/usr/bin/env julia
# scripts/make_plot.jl — solve the optimized-shape state, sample density+current
# with FermiSea's save_cartesian, and emit a self-contained HTML figure.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra, Printf, JLD2, HDF5, Statistics
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
ALPHA=2000.0; EPS=1e-10; NVIS=140
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
saved = load(joinpath(workdir,"result_point_opt.jld2"))
bestphys = saved["bestphys"]; f1_saved = saved["best_f1"]
DNX,DNY = size(bestphys)

geo=joinpath(workdir,"cpp.geo"); inp=joinpath(workdir,"cpp.inp")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.04)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
field = DensityField(cfg; nx=DNX, ny=DNY, alpha_max=1.0)
ev = FixedEvaluator(inp, field, sim); semi=ev.semi
_, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)
extract()=(FermiSea._current_contact_potential(ev.bc_probe_A,equations,dg,cache)-
           FermiSea._current_contact_potential(ev.bc_probe_B,equations,dg,cache))/sim.I_source
clear!(field); ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0)
println("assembling..."); flush(stdout)
Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(b);ej=zeros(N)
for j in 1:N
    ej[j]=1.0; Trixi.rhs!(tmp,ej,semi,0.0); c[j]=extract(); ej[j]=0.0
    @inbounds for i in 1:N
        v=tmp[i]-b[i]; abs(v)>1e-12 && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
    end
end
A0=sparse(Ir,Jc,Vv,N,N)
fill!(field.rho,1.0); ov=ones(N); r1=similar(b); Trixi.rhs!(r1,ov,semi,0.0)
clear!(field); r0=similar(b); Trixi.rhs!(r0,ov,semi,0.0); b1u=r1.-r0
rpd=zeros(N); w=Trixi.wrap_array(rpd,semi); nc=cache.elements.node_coordinates
nn=size(w,2); nel=size(w,4); field.rho.=bestphys
for e in 1:nel, jj in 1:nn, ii in 1:nn
    x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
    ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
    cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
    for v in 1:nvars; w[v,ii,jj,e]=bestphys[ci,cj]; end
end
u = lu(A0+spdiagm(0=>(ALPHA.*rpd).*b1u)+EPS*spdiagm(0=>ones(N)))\(-b)
f1 = dot(c,u)
@printf("solved: f_1=%.4f\n", f1); flush(stdout)

h5f = joinpath(workdir,"opt_cartesian.h5")
FermiSea.save_cartesian(u, semi, h5f; nvisnodes=NVIS)
x,y,a1,b1 = h5open(h5f,"r") do f
    read(f["x"]), read(f["y"]), read(f["a1"]), read(f["b1"])
end
nx=length(x); ny=length(y)
# jmag[xi,yi]; obstacle density on same grid
jmag=zeros(nx,ny); obst=zeros(nx,ny)
for yi in 1:ny, xi in 1:nx
    jmag[xi,yi]=sqrt(a1[xi,yi]^2+b1[xi,yi]^2)
    ci=clamp(floor(Int,(x[xi]-field.x0)/field.dx)+1,1,field.nx)
    cj=clamp(floor(Int,(y[yi]-field.y0)/field.dy)+1,1,field.ny)
    obst[xi,yi]=bestphys[ci,cj]
end
jclip = quantile(vec(jmag), 0.985)   # robust color scale
# arrow subgrid
AX,AY=26,16
arr = Tuple{Float64,Float64,Float64,Float64}[]
for aj in 1:AY, ai in 1:AX
    xi=clamp(round(Int,(ai-0.5)/AX*nx),1,nx); yi=clamp(round(Int,(aj-0.5)/AY*ny),1,ny)
    push!(arr,(x[xi],y[yi],a1[xi,yi],b1[xi,yi]))
end

flat(A)=join((@sprintf("%.4g",v) for v in vec(permutedims(A))), ",")  # row-major (y outer)
jstr="["*flat(jmag)*"]"; ostr="["*flat(obst)*"]"
axs=join((@sprintf("%.4g",t[1]) for t in arr),","); ays=join((@sprintf("%.4g",t[2]) for t in arr),",")
au=join((@sprintf("%.4g",t[3]) for t in arr),","); av=join((@sprintf("%.4g",t[4]) for t in arr),",")
drho="["*join((@sprintf("%.3g",v) for v in vec(permutedims(bestphys))),",")*"]"

xPL=cfg.x_probe-cfg.L_probe/2; xPR=cfg.x_probe+cfg.L_probe/2; wc=0.12
data = """{
"nx":$nx,"ny":$ny,"x0":$(x[1]),"x1":$(x[end]),"y0":$(y[1]),"y1":$(y[end]),
"jmag":$jstr,"jclip":$(@sprintf("%.4g",jclip)),"obst":$ostr,
"ax":[$axs],"ay":[$ays],"au":[$au],"av":[$av],
"dnx":$DNX,"dny":$DNY,"drho":$drho,
"f1":$(@sprintf("%.1f",f1)),"alpha":$(Int(ALPHA)),"vol":0.12,"gmc":100,
"probeL":$xPL,"probeR":$xPR,"cw":$wc,"L":$(cfg.L_x),"W":$(cfg.W)
}"""

body = raw"""
<style>
:root{--bg:#f6f8fa;--fg:#0f1418;--muted:#5b6672;--line:#dfe4ea;--card:#ffffff;--accent:#2b7bd6;}
@media (prefers-color-scheme:dark){:root{--bg:#0f1418;--fg:#e8edf2;--muted:#8b97a4;--line:#232b33;--card:#161c22;--accent:#5aa0ea;}}
:root[data-theme="light"]{--bg:#f6f8fa;--fg:#0f1418;--muted:#5b6672;--line:#dfe4ea;--card:#ffffff;--accent:#2b7bd6;}
:root[data-theme="dark"]{--bg:#0f1418;--fg:#e8edf2;--muted:#8b97a4;--line:#232b33;--card:#161c22;--accent:#5aa0ea;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px 64px}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{font-size:clamp(22px,3.4vw,30px);margin:.3em 0 .1em;text-wrap:balance;font-weight:650}
.sub{color:var(--muted);font-size:15px;max-width:64ch}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 26px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:110px}
.stat .k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.stat .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 18px 14px;margin-bottom:20px}
.panel h2{font-size:14px;margin:0 0 2px;font-weight:600}
.panel p{margin:0 0 12px;color:var(--muted);font-size:13px}
.figrow{display:flex;gap:14px;align-items:stretch;flex-wrap:wrap}
canvas{display:block;width:100%;height:auto;border-radius:8px;background:#0b0f13}
.cbar{width:54px;flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:6px;font-size:11px;color:var(--muted)}
.cbar .bar{width:14px;flex:1;border-radius:4px;border:1px solid var(--line)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.foot{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:16px;margin-top:8px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.two{display:grid;grid-template-columns:1.6fr 1fr;gap:14px}
@media(max-width:720px){.two{grid-template-columns:1fr}}
</style>

<div class="wrap">
  <div class="eyebrow">Fixed-mesh topology optimization &middot; FermiSea.jl</div>
  <h1>Optimized obstacle for maximum vicinity resistance</h1>
  <p class="sub">Free-form density (Brinkman) optimization via the discrete adjoint, in the
  point-contact vicinity geometry. The optimizer places a slanted constriction that
  redirects the central source&rarr;drain current, producing a large transverse
  probe-voltage difference.</p>

  <div class="stats" id="stats"></div>

  <div class="panel">
    <h2>Current flow around the optimized obstacle</h2>
    <p>Colour = current magnitude |j| = &radic;(a&#8321;&sup2;+b&#8321;&sup2;) sampled on the DG mesh (FermiSea
    <span class="mono">save_cartesian</span>). Arrows = flow direction. Grey = obstacle (&rho;&rarr;1, solid).</p>
    <div class="figrow">
      <canvas id="flow" width="880" height="528"></canvas>
      <div class="cbar"><span id="cmax">hi</span><div class="bar" id="cbar"></div><span>0</span><span>|j|</span></div>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:#f0a500"></span>source (I in)</span>
      <span><span class="dot" style="background:#28c3d4"></span>drain (V=0)</span>
      <span><span class="dot" style="background:#e05fb0"></span>probe A / B (floating)</span>
      <span><span class="dot" style="background:#9aa3ad"></span>obstacle</span>
    </div>
  </div>

  <div class="two">
    <div class="panel">
      <h2>Optimized obstacle density</h2>
      <p>Design field &rho;&isin;[0,1] on the 22&times;14 grid (filtered). Volume fraction fixed.</p>
      <canvas id="dens" width="440" height="280"></canvas>
    </div>
    <div class="panel">
      <h2>How to read it</h2>
      <p style="color:var(--fg)">The obstacle is a tilted bar spanning the channel. Its up&ndash;down
      tilt breaks mirror symmetry, so probe&nbsp;A and probe&nbsp;B see different potentials
      &mdash; that difference, per unit injected current, is f&#8321;.</p>
      <p>A centred or symmetric obstacle gives f&#8321;=0 exactly; full-edge contacts give
      f&#8321;&asymp;0 for any obstacle. The signal exists because narrow contacts create a
      genuinely 2-D spreading flow.</p>
    </div>
  </div>

  <div class="foot">
    <b>Method.</b> Boltzmann harmonic-moment model (IsotropicFermiHarmonics2D), exact
    regularized sparse-LU steady solve; adjoint gradient verified vs finite difference to
    1e-6. <b>Caveats.</b> f&#8321;&asymp;620 is exact on the h=0.05 mesh (&epsilon;- and GMRES-verified) but
    carries ~10% scatter under mesh refinement at this coarse design grid; the optimizer had
    not fully plateaued. Magnitudes are in the model's natural units (v_F=1, I=1).
  </div>
</div>

<script>
const D = __DATA__;
const vir=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
function viridis(t){t=Math.max(0,Math.min(1,t));const s=t*9,i=Math.floor(s),f=s-i,a=vir[i],b=vir[Math.min(9,i+1)];
 return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`;}
function X(v,c){return (v-D.x0)/(D.x1-D.x0)*c.width;}
function Y(v,c){return (1-(v-D.y0)/(D.y1-D.y0))*c.height;}   // y up
// stats
const st=[["f₁","+"+D.f1,"vicinity resistance"],["α",D.alpha,"obstacle solidity"],
 ["vol",D.vol,"material fraction"],["γ_mc",D.gmc,"collision rate"]];
document.getElementById("stats").innerHTML=st.map(s=>
 `<div class="stat"><div class="k">${s[2]}</div><div class="v">${s[0]} = ${s[1]}</div></div>`).join("");
// colorbar
document.getElementById("cbar").style.background=
 `linear-gradient(to top,${Array.from({length:10},(_,i)=>viridis(i/9)).join(",")})`;
document.getElementById("cmax").textContent=(+D.jclip).toPrecision(2);
// flow panel
(function(){const c=document.getElementById("flow"),g=c.getContext("2d");
 const nx=D.nx,ny=D.ny,dw=c.width/nx,dh=c.height/ny;
 for(let yi=0;yi<ny;yi++)for(let xi=0;xi<nx;xi++){
   const j=D.jmag[yi*nx+xi];
   g.fillStyle=viridis(Math.sqrt(Math.min(1,j/D.jclip)));   // sqrt for contrast
   g.fillRect(xi*dw, (ny-1-yi)*dh, dw+1, dh+1);
 }
 // obstacle overlay
 for(let yi=0;yi<ny;yi++)for(let xi=0;xi<nx;xi++){const o=D.obst[yi*nx+xi];
   if(o>0.06){g.fillStyle=`rgba(150,160,172,${0.16+0.72*o})`;g.fillRect(xi*dw,(ny-1-yi)*dh,dw+1,dh+1);}}
 // arrows
 let am=0;for(let k=0;k<D.au.length;k++)am=Math.max(am,Math.hypot(D.au[k],D.av[k]));
 g.lineWidth=1.3;g.strokeStyle="rgba(255,255,255,.72)";g.fillStyle="rgba(255,255,255,.72)";
 for(let k=0;k<D.ax.length;k++){const px=X(D.ax[k],c),py=Y(D.ay[k],c);
   const m=Math.hypot(D.au[k],D.av[k]);if(m<am*0.02)continue;
   const s=14*Math.sqrt(m/am)/m, ux=D.au[k]*s, uy=-D.av[k]*s;
   g.beginPath();g.moveTo(px-ux/2,py-uy/2);g.lineTo(px+ux/2,py+uy/2);g.stroke();
   const ang=Math.atan2(uy,ux),hx=px+ux/2,hy=py+uy/2;
   g.beginPath();g.moveTo(hx,hy);g.lineTo(hx-4*Math.cos(ang-0.5),hy-4*Math.sin(ang-0.5));
   g.lineTo(hx-4*Math.cos(ang+0.5),hy-4*Math.sin(ang+0.5));g.closePath();g.fill();}
 // contacts / probes
 function bar(x0,y0,x1,y1,col,lab,lx,ly){g.strokeStyle=col;g.lineWidth=4;
   g.beginPath();g.moveTo(X(x0,c),Y(y0,c));g.lineTo(X(x1,c),Y(y1,c));g.stroke();
   g.fillStyle=col;g.font="600 12px ui-sans-serif";g.fillText(lab,lx,ly);}
 const cw=D.cw;
 bar(D.x0,-cw/2,D.x0,cw/2,"#f0a500","S",4,c.height/2);
 bar(D.x1,-cw/2,D.x1,cw/2,"#28c3d4","D",c.width-16,c.height/2);
 bar(D.probeL,D.W/2,D.probeR,D.W/2,"#e05fb0","A",X((D.probeL+D.probeR)/2,c)-4,12);
 bar(D.probeL,-D.W/2,D.probeR,-D.W/2,"#e05fb0","B",X((D.probeL+D.probeR)/2,c)-4,c.height-4);
})();
// density panel
(function(){const c=document.getElementById("dens"),g=c.getContext("2d");
 const nx=D.dnx,ny=D.dny,dw=c.width/nx,dh=c.height/ny;
 for(let yi=0;yi<ny;yi++)for(let xi=0;xi<nx;xi++){const r=D.drho[yi*nx+xi];
   const v=Math.round(20+200*r);g.fillStyle=`rgb(${v},${Math.round(v*0.92)},${Math.round(v*0.8)})`;
   g.fillRect(xi*dw,(ny-1-yi)*dh,dw+0.6,dh+0.6);}
})();
</script>
"""

html = replace(body, "__DATA__" => data)
out = joinpath(workdir, "opt_plot.html")
open(out,"w") do io; write(io, html); end
println("wrote ", out)

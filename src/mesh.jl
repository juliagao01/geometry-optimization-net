"""
    Mesh

Tiny wrapper around `Gmsh.jl` that turns one of our `.geo` files into a
Trixi-readable `.inp` mesh. Equivalent to the script in the FermiSea.jl
mesh-generation tutorial but parameterized and headless.
"""
module Mesh

using Gmsh: gmsh

export geo_to_inp

"""
    geo_to_inp(geo_path, inp_path; verbose=false)

Mesh a `.geo` file with the Quasi-structured Quad algorithm and recombine
to quads, exactly as FermiSea.jl's tutorial does. Writes `inp_path` (Abaqus
format) ready for `Trixi.P4estMesh{2}`.
"""
function geo_to_inp(geo_path::AbstractString, inp_path::AbstractString;
                    verbose::Bool=false)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", verbose ? 1 : 0)
        gmsh.open(geo_path)
        gmsh.option.setNumber("Mesh.Algorithm", 11)    # quasi-structured quad
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(inp_path)
    finally
        gmsh.finalize()
    end
    return inp_path
end

end # module

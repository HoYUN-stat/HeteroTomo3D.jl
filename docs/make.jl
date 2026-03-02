using HeteroTomo3D
using Documenter

DocMeta.setdocmeta!(HeteroTomo3D, :DocTestSetup, :(using HeteroTomo3D); recursive=true)

makedocs(;
    modules=[HeteroTomo3D],
    authors="YUN Ho <ho.yun@epfl.ch> and contributors",
    sitename="HeteroTomo3D.jl",
    format=Documenter.HTML(;
        canonical="https://HoYUN-stat.github.io/HeteroTomo3D.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quaternions" => "quaternions.md",
        "Tomographic Kernels" => "tomokernel.md",
        "Forward Operations" => "forward.md",
    ],
)

deploydocs(;
    repo="github.com/HoYUN-stat/HeteroTomo3D.jl",
    devbranch="main",
)

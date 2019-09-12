using Documenter, YaoExtensions

makedocs(;
    modules=[YaoExtensions],
    format=Documenter.HTML(assets=String[]),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/QuantumBFS/YaoExtensions.jl/blob/{commit}{path}#L{line}",
    sitename="YaoExtensions.jl",
    authors="JinGuo Liu, XiuZhe Luo",
)

deploydocs(;
    repo="github.com/QuantumBFS/YaoExtensions.jl",
)

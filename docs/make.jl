using Documenter, YaoExtensions

makedocs(;
    modules=[YaoExtensions],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/GiggleLiu/YaoExtensions.jl/blob/{commit}{path}#L{line}",
    sitename="YaoExtensions.jl",
    authors="JinGuo Liu",
    assets=String[],
)

deploydocs(;
    repo="github.com/GiggleLiu/YaoExtensions.jl",
)

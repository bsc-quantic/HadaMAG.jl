using Documenter
using HadaMAG

makedocs(
    modules = [HadaMAG],
    sitename = "HadaMAG.jl",
    pages = Any["Home"=>"index.md", "Manual"=>["State"=>"manual/State.md", "Helpers"=>"manual/Helpers.md"]],
    pagesonly = true,
    format = Documenter.HTML(),
    warnonly = true,
)

deploydocs(;
    repo = "github.com/bsc-quantic/HadaMAG.jl.git",
    devbranch = "master",
    push_preview = true,
)

using Documenter
using HadaMAG

makedocs(
    modules = [HadaMAG],
    sitename = "HadaMAG.jl",
    pages = Any[
        "Home"=>"index.md",
        "Manual"=>["Exact SRE"=>"manual/Exact_SRE.md",
        "Custom FHT Library"=>"manual/CustomFHT.md",],
        # "Manual"=>["Exact SRE"=>"manual/Exact_SRE.md",
        #            "Monte Carlo SRE"=>"manual/Monte_Carlo_SRE.md",
        #            "Mana Computation"=>"manual/Mana_Computation.md",],
        "API"=>["State"=>"api/State.md", "Helpers"=>"api/Helpers.md"],
    ],
    pagesonly = true,
    format = Documenter.HTML(),
    warnonly = true,
)

deploydocs(;
    repo = "github.com/bsc-quantic/HadaMAG.jl.git",
    devbranch = "master",
    push_preview = true,
)

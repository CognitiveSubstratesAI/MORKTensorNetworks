using Documenter
using MORKTensorNetworks

DocMeta.setdocmeta!(
    MORKTensorNetworks, :DocTestSetup, :(using MORKTensorNetworks); recursive=true
)

makedocs(;
    modules=[MORKTensorNetworks],
    authors="CognitiveSubstrates AI",
    repo=Remotes.GitHub("CognitiveSubstratesAI", "MORKTensorNetworks"),
    sitename="MORKTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cognitivesubstratesai.github.io/MORKTensorNetworks/stable/",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Architecture" => "architecture.md"
    ],
    # Pages link to source files / audit records outside docs/src; tolerate warnings.
    warnonly=true
)

deploydocs(; repo="github.com/CognitiveSubstratesAI/MORKTensorNetworks", devbranch="main")

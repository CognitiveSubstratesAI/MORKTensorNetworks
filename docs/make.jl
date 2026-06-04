using Documenter
using MORKTensorNetworks

DocMeta.setdocmeta!(
    MORKTensorNetworks, :DocTestSetup, :(using MORKTensorNetworks); recursive=true
)

# NOTE: repo currently lives at sivaji1012/MORKTensorNetworks (audit-then-migrate);
# it moves to CognitiveSubstratesAI at migration — update repo/deploydocs then.
makedocs(;
    modules=[MORKTensorNetworks],
    authors="CognitiveSubstrates AI",
    repo=Remotes.GitHub("sivaji1012", "MORKTensorNetworks"),
    sitename="MORKTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sivaji1012.github.io/MORKTensorNetworks/stable/",
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

deploydocs(; repo="github.com/sivaji1012/MORKTensorNetworks", devbranch="main")

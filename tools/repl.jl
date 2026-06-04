#!/usr/bin/env julia
# tools/repl.jl — development REPL
#
# Interactive:
#   julia --project=. -i tools/repl.jl
#
# Scripted:
#   printf 'include("test/runtests.jl")\n' | julia --project=. tools/repl.jl
#   printf 't()\n' | julia --project=. tools/repl.jl

try
    using Revise
catch
end

using MORKTensorNetworks

t(path=joinpath(@__DIR__, "..", "test", "runtests.jl")) = include(path)

if isinteractive()
    println("MORKTensorNetworks v0.1.0 loaded.")
    println("  t()  — run full test suite")
end

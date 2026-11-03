# main.jl

include("src41/LoquatSPH.jl")

using .LoquatSPH

function main()
    args = ARGS
    
    if length(args) < 1
        println("Error! Please give the project name!")
        println("Usage: julia main.jl <project_name> [device_id]")
        exit(1)
    end
    
    project_name = args[1]
    device_id = 0
    
    if length(args) >= 2
        try
            device_id = parse(Int, args[2])
        catch
            println("Error! Invalid device ID format: $(args[2])")
            exit(1)
        end
    end
    
    try
        RunSimulation(project_name, device_id)
    catch e
        println("\n------------------------------------------------------------")
        println("Simulation Terminated with Error:")
        showerror(stderr, e, catch_backtrace())
        println("\n------------------------------------------------------------")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
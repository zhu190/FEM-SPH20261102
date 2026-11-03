# main.jl

# ==============================================================================
# LOQUAT-SPH Julia Entry Point
# ==============================================================================

# Include the main module source
# Assuming main.jl is in the project root and source is in ./src/
include("../src41/LoquatSPH.jl")

# Bring the module into scope
using .LoquatSPH

function main()
    # 1. Parse Command Line Arguments
    args = ARGS
    
    if length(args) < 1
        println("Error! Please give the project name!")
        println("Usage: julia main.jl <project_name> [device_id]")
        exit(1)
    end
    
    project_name = args[1]
    
    # Default device ID is 0
    device_id = 0
    
    # Optional: GPU Device ID
    if length(args) >= 2
        try
            device_id = parse(Int, args[2])
        catch
            println("Error! Invalid device ID format: $(args[2])")
            exit(1)
        end
    end
    
    # 2. Start Simulation
    # Calls the main loop defined in src/LoquatSPH.jl
    try
        RunSimulation(project_name, device_id)
    catch e
        # Catch and print any runtime errors nicely
        println("\n------------------------------------------------------------")
        println("Simulation Terminated with Error:")
        showerror(stderr, e, catch_backtrace())
        println("\n------------------------------------------------------------")
        exit(1)
    end
end

# Execute the main function if this file is run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
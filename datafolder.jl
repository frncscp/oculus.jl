using Flux, Images, Random
using Flux: onehotbatch, DataLoader
using CUDA
using Suppressor
using ProgressBars

Random.seed!(420) #for reproduction, hehe

struct datafolder
    folder::String #where the data is stored
    dims::Tuple #image dimensions, like 224x224x3 (there is not support for greyscale images)
    postive_class::String #the class you actually want to detect
    
    #percentage of the set that is going to be used in subsets
    train::Float16
    test::Float16 

    #X and y respectively
    features::Array{Float16, 4}
    targets::Vector{Int64}
end

function get_length(folder) #assuming you have a main folder that contains subfolders with files
    len = 0
    for subfolder in readdir(folder)
        len += length(readdir(joinpath(folder, subfolder)))
    end
    return len
end

function datafolder(folder::String, dims::Tuple, postive_class::String, train::Float16, test::Float16)
    N = get_length(folder)
    features = Array{Float16, 4}(undef, dims[1], dims[2], dims[3], N)
    targets = Vector{Int64}()

    for subfolder in readdir(folder)
        class = subfolder == postive_class ? 1 : 0
        len = length(readdir(joinpath(folder, subfolder)))

        println("Transferring data from $subfolder folder...")
        pbar = ProgressBar(total = len)
        
        for (i, filename) in enumerate(readdir(joinpath(folder, subfolder)))
            ProgressBars.update(pbar)
            filename = joinpath(folder, subfolder, filename)

            @suppress_err begin
            image = load(filename)
            image = imresize(image, (dims[1], dims[2]))
            arr = channelview(image)
            arr = convert(Array{Float16, dims[3]}, arr)
            if size(arr)[1] == 4 #if its not rgba instead of rgb
                arr = arr[1:3, :, :]
            end
            perm_arr = PermutedDimsArray(arr, (2, 3, 1))
            features[:, :, :, i] = Float16.(perm_arr) #X (features)
            push!(targets, copy(class)) #y (targets)
            end
        end
    end
    return datafolder(folder, dims, postive_class, train, test, features, targets)
end

Base.show(io::IO, obj::datafolder) = print(io, "Dataset size: $(size(obj.features))\nNumber of features: $(size(obj.targets))\nNumber of elements: $(get_length(obj.folder))")

function loader(data::datafolder, batchsize::Int64=16, imsize::Tuple= (224, 224, 3))
    yhot = onehotbatch(data.targets, 0:1) # encoding 2 classes into [0, 1]
    DataLoader((data.features, yhot); batchsize, shuffle=true) |> gpu
end

function dataset_from_datafolder(datadir::String, dims::Tuple, postive_class::String, batch_size::Int64 = 16 , train::Float16 = Float16(0.6), test::Float16 = Float16(0.4))
    df = datafolder(datadir, dims, postive_class, Float16(train), Float16(test))
    return loader(df, batch_size)
end
include("datafolder.jl")
using Flux, LIBSVM, MLJ
using Flux: crossentropy, train!
using OutlierDetection

#= 

this code is composed by three models that work as one:

- a convnet as a feature extractor
- a PCA model to trim down the size of the features
- a svm as a classifier, exploiting its anomaly detection capabilities

=#

horus = Chain(
    #color transformation
    Conv((1, 1), 3 => 10, pad=(1,1)),
    Conv((1, 1), 10 => 3, pad=(1,1)),

    #broad receptive fields

    #block 1
    Conv((3, 3), 3 => 256, pad=(1,1)),
    BatchNorm(256),
    Conv((3, 3), 256 => 256, pad=(1,1)),
    elu,
    LayerNorm(256),
    MaxPool((3, 3), pad=(1,1)),

    #block 2
    Conv((5, 5), 256 => 128, pad=(1,1)),
    Conv((5, 5), 128 => 64, pad=(1,1)),
    Dropout(0.2),
    LayerNorm(64),
    MaxPool((5, 5), pad=(1,1)),

    #block 3
    Conv((7, 7), 64 => 32, pad=(1,1)),
    BatchNorm(32),
    Conv((7, 7), 32 => 64, pad=(1,1)),
    elu,
    LayerNorm(64),
    MaxPool((7, 7), pad=(1,1)),

    #heavy block
    Conv((3, 3), 64 => 128, pad=(1,1)),
    Conv((5, 5), 128 => 256, pad=(1,1)),
    Conv((7, 7), 256 => 512, pad=(1,1)),
    Dropout(0.4),
    LayerNorm(512),
    MaxPool((7, 7), pad=(1,1)),
    x -> mean(x, dims=(1, 2)),
) |> gpu

λ = 1e-4

horus



#clf = OneClassSVM()
#doc("PCA", pkg="MLJ")
#doc("OneClassSVM", pkg="LIBSVM")

#=
patacon730 = dataset_from_datafolder(
    "C:\\Users\\franc\\Downloads\\archive\\patacon-730",
    (224, 224, 3),
    "Patacon-True",
)=#

#pmodel = ProbabilisticDetector(clf)
#pmach = machine(pmodel, X) |> fit!
#y_prob = predict(pmach, Xnew)
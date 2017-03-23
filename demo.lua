-- demo.lua

require 'nn'
require 'RandomBinaryConvolution'

local RandomBinaryConvolution = nn.RandomBinaryConvolution
local Convolution = nn.SpatialConvolution

local sz = 3 -- convoltuional kernel size
local nChInp = 10 -- number of input channels
local nChOut = 20 -- number of output channels
local nChInt = 20 -- number of intermediate channels for lbc, could be more or less than nChOut
kSparsity = 0.5

-- define LBC module
local lbc = nn.Sequential()
lbc:add(RandomBinaryConvolution(nChInp,nChInt,sz,sz,1,1,(sz-1)/2,(sz-1)/2),kSparsity)
lbc:add(nn.ReLU())
lbc:add(Convolution(nChInt,nChOut,1,1))

-- define standard convolutional module
local cnn = nn.Sequential()
cnn:add(Convolution(nChInp,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2))

-- input data
local batchSize = 16
local resolution = 32
local input = torch.rand(batchSize,nChInp,resolution,resolution)

-- cnn output
local outCNN = cnn:forward(input)

-- lbc output
local outLBC = lbc:forward(input)

print(outCNN)
print(outLBC)
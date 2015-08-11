
require 'nn'
require 'image'
require 'constants.lua'


-- creating the model for linear regression
-- Try one: 1-D without convolution
model = nn.Sequential() 

--[[
ninputs = IMG_DIM*IMG_DIM; nhidden = 100; noutputs = MAX_FEATURE
model:add(nn.Linear(ninputs, nhidden))
model:add(nn.ReLU())
model:add(nn.Linear(nhidden, noutputs)) 
--]]


--[[
-- hidden units, filter sizes (for ConvNet only):
nfeats=1
nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)
noutputs = MAX_FEATURE
-- a typical convolutional network, with locally-normalized hidden
-- units, and L2-pooling

-- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
-- work on this dataset (`). In particular
-- the use of LP-pooling (with P=2) has a very positive impact on
-- generalization. Normalization is not done exactly as proposed in
-- the paper, and low-level (first layer) features are not fed to
-- the classifier.
-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(21*21*nstates[2])) -- nstates[2]*filtsize*filtsize))
model:add(nn.Linear(21*21*nstates[2], nstates[3])) 
model:add(nn.Tanh())
model:add(nn.Linear(nstates[3], noutputs))
--]]


-- hidden units, filter sizes (for ConvNet only):
nfeats=1
nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)
noutputs = MAX_FEATURE
-- a typical modern convolution network (conv+relu+pool)
model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : standard 2-layer neural network
model:add(nn.View(nstates[2]*21*21))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2]*21*21, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))

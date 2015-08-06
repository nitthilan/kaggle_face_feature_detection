
require 'nn'
require 'optim'
require 'constants.lua'


-- Loading the pre-processed data 
local feature_data = torch.load(FILEPATH_DATA_DIR.."feature_data.raw", 'binary')
local image_data = torch.load(FILEPATH_DATA_DIR.."image_data.raw", 'binary')

local csvFile = io.open(FILEPATH_DATA_DIR..'training.csv', 'r')  
local header = csvFile:read()
csvFile:close()
local feature_map = header:split(',')


-- finding out how many valid values are there for each feature
for i=1,MAX_FEATURE do
	local byte_vec = torch.ne(feature_data:select(2,i), -1.0)
	print (feature_map[i], torch.sum(byte_vec))
end

-- finding out all images which have all the 30 features
local num_images = 0
local image_id_map = {}
for i=1,MAX_TRAIN_IMG do
	local byte_vec = torch.ne(feature_data:select(1,i), -1.0)
	if torch.sum(byte_vec) == MAX_FEATURE then
		num_images = num_images + 1
		image_id_map[num_images] = i
	end
end
print ("Num images with all the 30 feature vectors ", num_images)
-- Calculating the 80% of total images as the training data set and the rest as the test data set to validate the training

local num_images_training = (80*num_images)/100
local num_images_validating = num_images - num_images_training
print ("Num train images", num_images_training, "Num validating images", num_images_validating)

-- creating the model for linear regression
-- Try one: 1-D without convolution
model = nn.Sequential() 
ninputs = IMG_DIM*IMG_DIM; nhidden = 100; noutputs = MAX_FEATURE
model:add(nn.Linear(ninputs, nhidden))
model:add(nn.ReLU())
model:add(nn.Linear(nhidden, noutputs)) 



----------------------------------------------------------------------
-- 3. Define a loss function, to be minimized.

-- In that example, we minimize the Mean Square Error (MSE) between
-- the predictions of our linear model and the groundtruth available
-- in the dataset.

-- Torch provides many common criterions to train neural networks.

criterion = nn.MSECriterion()


----------------------------------------------------------------------
-- 4. Train the model

-- To minimize the loss defined above, using the linear model defined
-- in 'model', we follow a stochastic gradient descent procedure (SGD).

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the 
-- entire training set is too costly.

-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these 
-- parameters by doing so:
x, dl_dx = model:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our model, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > num_images_training then _nidx_ = 1 end

   local image_id = image_id_map[_nidx_]
   local inputs = image_data[image_id]
   local target = feature_data[image_id]

   -- print (image_id, target)

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = .01, --1e-3,
   learningRateDecay = .001, --1e-4,
   weightDecay = 0,
   momentum = .9
}



-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation.

-- we cycle 1e4 times over our training data
for i = 1,1e4 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,num_images_training do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      -- _,fs = optim.sgd(feval,x,sgd_params)
      _,fs = optim.nag(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / num_images_training
   print('current loss = ' .. current_loss)
   -- Validating the trained model using validating data set
   local validation_loss = 0.0
   for i = num_images_training,num_images do
      local image_id = image_id_map[i]
      local inputs = image_data[image_id]
      local target = feature_data[image_id]
      local myPrediction = model:forward(inputs)
      local error = target - myPrediction
      -- local mse1 = math.sqrt(torch.mean(torch.pow(error, 2)))
      local mse = torch.norm(error)/math.sqrt(MAX_FEATURE);
      -- print ('mse', mse, mse1)
      validation_loss = validation_loss + mse
   end
   print('current validation loss', validation_loss/num_images_validating)

end



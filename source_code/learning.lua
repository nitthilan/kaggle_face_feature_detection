require 'torch'
require 'nn'
require 'optim'
require 'constants.lua'


-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: mlp | convnet')
   cmd:option('-type', 'cuda', 'type: double | cuda')
   cmd:option('-numdata', 'all_features', 'numdata: all_images | all_features')
   cmd:option('-batch_size', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:text()
   opt = cmd:parse(arg or {})
end

if(opt.type == 'cuda') then
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.manualSeedAll(1)
end

torch.setnumthreads(10)--opt.threads)
torch.manualSeed(1)--opt.seed)

-- Loading the pre-processed data 
local feature_data = torch.load(FILEPATH_DATA_DIR.."feature_data.raw", 'binary')
local image_data = torch.load(FILEPATH_DATA_DIR.."image_data.raw", 'binary')
-- image_data = image_data:float()

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

if opt.numdata == 'all_features' then
   for i=1,MAX_TRAIN_IMG do
   	local byte_vec = torch.ne(feature_data:select(1,i), -1.0)
   	if torch.sum(byte_vec) == MAX_FEATURE then
   		num_images = num_images + 1
   		image_id_map[num_images] = i
   	end
   end
else
   for i=1,MAX_TRAIN_IMG do
      num_images = num_images + 1
      image_id_map[num_images] = i
   end
end
print ("Num images with all the 30 feature vectors ", num_images)
-- Calculating the 80% of total images as the training data set and the rest as the test data set to validate the training


local num_images_training = math.floor((80*num_images)/100)
local num_images_validating = num_images - num_images_training
print ("Num train images", num_images_training, "Num validating images", num_images_validating)

-- shuffle at each epoch
local shuffle_idx -- = torch.randperm(trsize)


dofile("model.lua")







----------------------------------------------------------------------
-- 3. Define a loss function, to be minimized.

-- In that example, we minimize the Mean Square Error (MSE) between
-- the predictions of our linear model and the groundtruth available
-- in the dataset.

-- Torch provides many common criterions to train neural networks.

local criterion = nn.MSECriterion()
-- criterion.sizeAverage = false


if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end
-- model:float()


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
   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()
   local loss_x = 0

   for batch_num = 1,opt.batch_size do 
      -- select a new training sample
      _nidx_ = (_nidx_ or 0) + 1
      if _nidx_ > num_images_training then _nidx_ = 1 end

      if _nidx_ % 50 == 0 then
         collectgarbage()
      end
      -- print (_nidx_)

      local image_id = image_id_map[shuffle_idx[_nidx_]]
      local inputs
      if opt.model == 'mlp' then
         inputs = image_data[image_id]
      else
         inputs = image_data[image_id]:view(1, 96,96)
      end
      
      -- print (inputs:dim(), inputs:size())
      local target = feature_data[image_id]
      
      if opt.type == 'cuda' then
         inputs = inputs:cuda()
         target = target:cuda()
      end

      -- print (image_id, target)


      -- Logic to make the predicted output to target output so that it does not affect the error calculation
      -- 
      local loss
      if opt.numdata == 'all_images' then
         local forward_output = model:forward(inputs)   
         local byte_vec_fea = torch.ne(target, -1.0)
         local byte_vec_non_fea = torch.eq(target, -1.0)
         local zeroed_target, selected_output, equalised_target
         if opt.type == 'cuda' then
            zeroed_target = torch.cmul(target:cuda(), byte_vec_fea:cuda())
            selected_output = torch.cmul(forward_output:cuda(), byte_vec_non_fea:cuda())
            equalised_target = torch.add(zeroed_target:cuda(), selected_output:cuda())
         else
            zeroed_target = torch.cmul(target:double(), byte_vec_fea:double())
            selected_output = torch.cmul(forward_output:double(), byte_vec_non_fea:double())
            equalised_target = torch.add(zeroed_target:double(), selected_output:double())
         end
         -- print(byte_vec_fea, byte_vec_non_fea, forward_output, zeroed_target, selected_output, equalised_target)
         -- evaluate the loss function and its derivative wrt x, for that sample

         loss = criterion:forward(forward_output, equalised_target)
         model:backward(inputs, criterion:backward(model.output, equalised_target))
      else
         loss = criterion:forward(model:forward(inputs), target)
         model:backward(inputs, criterion:backward(model.output, target))
      end
      loss_x = loss_x + loss
   end
   loss_x = loss_x/opt.batch_size
   dl_dx = dl_dx:div(opt.batch_size)

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
for epoch = 1,1e4 do

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:training()
   
   -- shuffle at each epoch
   shuffle_idx = torch.randperm(num_images_training)
   -- print (shuffle_idx)
   
   -- this variable is used to estimate the average loss
   current_loss = 0

   -- local vars
   local time = sys.clock()

   -- an epoch is a full loop over our training data
   for img_id = 1,num_images_training,opt.batch_size do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      --_,fs = optim.sgd(feval,x,sgd_params)
      _,fs = optim.nag(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + math.sqrt(fs[1])
      -- print ("image error ", fs[1])
   end

   -- time taken
   time = sys.clock() - time
   time = time / num_images_training
   -- report average error on epoch
   current_loss = current_loss * opt.batch_size / num_images_training
   print(epoch..' current loss = ' .. current_loss)
   -- print("==> time to learn 1 sample = " .. (time*1000) .. ' ms')
   
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   -- Validating the trained model using validating data set
   local validation_loss = 0.0
   time = sys.clock()
   for i = num_images_training,num_images do
      local image_id = image_id_map[i]
      if opt.model == 'mlp' then
         inputs = image_data[image_id]
      else
         inputs = image_data[image_id]:view(1, 96,96)
      end
      
      if opt.type == 'cuda' then
         inputs = inputs:cuda()
      end
      local target = feature_data[image_id]
      local forward_output = model:forward(inputs)
      
      local byte_vec_fea = torch.ne(feature_data:select(1,image_id), -1.0)
      local byte_vec_non_fea = torch.eq(feature_data:select(1,image_id), -1.0)
      local zeroed_target, selected_output, equalised_target
      if opt.type == 'cuda' then
         zeroed_target = torch.cmul(target:cuda(), byte_vec_fea:cuda())
         selected_output = torch.cmul(forward_output:cuda(), byte_vec_non_fea:cuda())
         equalised_target = torch.add(zeroed_target:cuda(), selected_output:cuda())
      else
         zeroed_target = torch.cmul(target:double(), byte_vec_fea:double())
         selected_output = torch.cmul(forward_output:double(), byte_vec_non_fea:double())
         equalised_target = torch.add(zeroed_target:double(), selected_output:double())
      end
      local error = equalised_target - forward_output
      -- local mse1 = math.sqrt(torch.mean(torch.pow(error, 2)))
      local mse = torch.norm(error)/math.sqrt(torch.sum(byte_vec_fea));
      -- print ('mse', mse, mse1)
      validation_loss = validation_loss + mse
   end
   print(epoch..' current validation loss '.. validation_loss/num_images_validating)
   -- time taken
   time = sys.clock() - time
   time = time / num_images_training
   -- print("==> time to validate 1 sample = " .. (time*1000) .. ' ms')

   -- saving the model for using it later
   modsav = model:clone('weight', 'bias');
   torch.save(FILEPATH_DATA_DIR.."trained_model_"..(epoch - math.floor(epoch/10)*10)..".t7",modsav)
   -- torch.save(FILEPATH_DATA_DIR.."trained_model_full.t7",model)
   
   --[[
   local savedModel = torch.load(FILEPATH_DATA_DIR.."trained_model.t7")
   local validation_loss = 0.0
   for i = num_images_training,num_images do
      local image_id = image_id_map[i]
      local inputs = image_data[image_id]
      local target = feature_data[image_id]
      local myPrediction = savedModel:forward(inputs)
      local error = target - myPrediction
      -- local mse1 = math.sqrt(torch.mean(torch.pow(error, 2)))
      local mse = torch.norm(error)/math.sqrt(MAX_FEATURE);
      -- print ('mse', mse, mse1)
      validation_loss = validation_loss + mse
   end
   print(epoch..' current saved validation loss '.. validation_loss/num_images_validating)
   ]]--



end



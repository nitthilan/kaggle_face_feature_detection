require 'nn'
require 'csvigo'
require 'constants.lua'

local csvFile = io.open(FILEPATH_TRAIN, 'r')  
local header = csvFile:read()
local feature_map = header:split(',')

local inv_feature_map = {}
for i=1,30 do
	inv_feature_map[feature_map[i]] = i
end

local testImageFile = csvigo.load(FILEPATH_TEST)
local testFeatureFile = csvigo.load(FILEPATH_TEST_FEATURE)

-- local image_data = torch.Tensor(MAX_TEST_IMG, IMG_DIM*IMG_DIM)
local feature_data = torch.Tensor(MAX_TEST_IMG, MAX_FEATURE)

-- print (testImageFile["Image"][2])

local savedModel = torch.load(FILEPATH_DATA_DIR.."trained_model_9.t7")
local validation_loss = 0.0
for i = 1,MAX_TEST_IMG do
	local x = testImageFile["Image"][i]
	local image = x:split(' ')
	local inputs = torch.Tensor(image)/MAX_PIXEL_VAL
  	local myPrediction = savedModel:forward(inputs)
  	feature_data[i] = torch.Tensor(myPrediction)
end

-- RowId,ImageId,FeatureName,Location
local testOutputFile = csvigo.File(FILEPATH_TEST_OUTPUT, "w")
testOutputFile:write({"RowId","Location"})
function trim1(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end
for i=1,MAX_TEST_OUTPUT do
	local imageId = testFeatureFile["ImageId"][i]
	local featureId = inv_feature_map[trim1(testFeatureFile["FeatureName"][i])]
	print(imageId, trim1(testFeatureFile["FeatureName"][i]))

	local location = feature_data[imageId][featureId]*96
	if(location > 95) then location = 95; end
	testOutputFile:write({i,location})	
end

testOutputFile:close()


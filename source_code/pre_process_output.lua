require 'constants.lua'

local csvFile = io.open(FILEPATH_TEST, 'r')  
local header = csvFile:read()
local feature_map = header:split(',')


local image_data = torch.Tensor(MAX_TEST_IMG, IMG_DIM*IMG_DIM)

for i=1,total_images do
	local x = csvigoFile[feature_map[2] ][i]
	local image = x:split(' ')
	image_data[i] = torch.Tensor(image)/MAX_PIXEL_VAL
end

torch.save("../data/test_feature.raw", feature_data, 'binary')
torch.save("../data/test_image_data.raw", image_data, 'binary')

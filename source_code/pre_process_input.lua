
require 'csvigo'
require 'constants.lua'



local csvFile = io.open(FILEPATH_TRAIN, 'r')  
local header = csvFile:read()



-- Mapping image id to 30 features. For features not present filling with -1
local feature_data = torch.Tensor(MAX_TRAIN_IMG,MAX_FEATURE)
-- There are 7049 samples each of 96x96 dimension
local image_data = torch.Tensor(MAX_TRAIN_IMG,IMG_DIM*IMG_DIM)
local feature_map = header:split(',')
local csvigoFile = csvigo.load(FILEPATH_TRAIN)
-- print (feature_data)


for i=1,MAX_TRAIN_IMG do
	local x = csvigoFile[feature_map[MAX_FEATURE+1] ][i]
	local image = x:split(' ')
	image_data[i] = torch.Tensor(image)/MAX_PIXEL_VAL

	image_feature = {}

	for j=1,MAX_FEATURE do
		local point_info = csvigoFile[feature_map[j]]
		local x = tonumber(point_info[i])
		if(x~=nil) then
			image_feature[j] = x/IMG_DIM
		else
			image_feature[j] = -1
		end
	end
	feature_data[i] = torch.Tensor(image_feature)
	print("num images", i)
end

 print (feature_data[10])
-- print (image_data[10])

torch.save(FILEPATH_DATA_DIR.."feature_data.raw", feature_data, 'binary')
torch.save(FILEPATH_DATA_DIR.."image_data.raw", image_data, 'binary')
torch.save(FILEPATH_DATA_DIR.."feature_data.ascii", feature_data, 'ascii')
torch.save(FILEPATH_DATA_DIR.."image_data.ascii", image_data, 'ascii')

--[[
for j=1,30 do
	local point_info = csvigoFile[feature_map[j] ]
	local image_index = {}
	table.setn(image_index, 30)

	for i=1,7049 do
		local x = tonumber(point_info[i])
		if(x~=nil) then
			image_index[i] = x
		end
	end
	feature_data[feature_map[j] ] = image_index
	print ("num features ", j)
end
]]--
--[[

for key, val in ipairs(feature_map) do
    feature_data[val] = {}
end



local i = 0  
for line in csvFile:lines('*l') do  
  i = i + 1
  data[i] = {}
  local l = line:split(',')
  local image = (l[31]:split(' '))
  local image_num = {}
  for i = 1,9216 do
  	image_num[i] = tonumber(image[i])
  end
  -- print (image)
  image_data[i] = torch.Tensor(image_num)
  --print (image)
  for j=1,30 do
  	local x = tonumber(l[j])
  	if(x ~= nil) then
  		feature_data[feature_map[j] ][i] = x
  	end
  end
  --if i == 1 then break end
  print(i)

end

-- print (feature_data)
-- print (feature_data["right_eyebrow_outer_end_y"])

csvFile:close()
--]]

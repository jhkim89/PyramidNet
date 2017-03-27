--  Implementation of "Deep Pyramidal Residual Networks" 

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local iChannels
   local alpha = 4.75
   local function round(x)
      return math.floor(x+0.5)
   end

   local function shortcut(nInputPlane, nOutputPlane, stride)
      -- Strided, zero-padded identity shortcut
      local short = nn.Sequential()
      if stride == 2 then
         short:add(nn.SpatialAveragePooling(2, 2, 2, 2))
      end
      if nInputPlane ~= nOutputPlane then
         short:add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))
      else
	       short:add(nn.Identity())
      end
      return short
   end

   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(SBatchNorm(nInputPlane))
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))
      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
   end
	
   local function bottleneck(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(SBatchNorm(nInputPlane))
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n*4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
   end

	
   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      for i=1,count do
         s:add(block(features, stride))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
   -- Configurations for ResNet:
   --  num. residual blocks, num features, residual block function
   local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      [34]  = {{3, 4, 6, 3}, 512, basicblock},
      [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      [200] = {{3, 24, 36, 3}, 2048, bottleneck},
   }

   assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
   local def, nFeatures, block = table.unpack(cfg[depth])
   iChannels = 64
   Channeltemp = 64
   local addrate = alpha^(1/(def[1]+def[2]+def[3]+def[4]))
   print(' | ResNet-' .. depth .. ' ImageNet')

   model:add(Convolution(3,64,7,7,2,2,3,3))
   model:add(SBatchNorm(64))
   model:add(ReLU(true))
   model:add(Max(3,3,2,2,1,1))
   Channeltemp = Channeltemp * addrate
   model:add(bottleneck(round(Channeltemp), 1, 1, 'first'))
   for i=2,def[1] do
      Channeltemp = Channeltemp * addrate
      model:add(bottleneck(round(Channeltemp), 1, 1))
   end
   Channeltemp = Channeltemp * addrate
   model:add(bottleneck(round(Channeltemp), 2, 1))
   for i=2,def[2] do
      Channeltemp = Channeltemp * addrate
      model:add(bottleneck(round(Channeltemp), 1, 1))
   end
   Channeltemp = Channeltemp * addrate
   model:add(bottleneck(round(Channeltemp), 2, 1))
   for i=2,def[3] do
      Channeltemp = Channeltemp * addrate
      model:add(bottleneck(round(Channeltemp), 1, 1))
   end
   Channeltemp = Channeltemp * addrate
   model:add(bottleneck(round(Channeltemp), 2, 1))
   for i=2,def[4] do
      Channeltemp = Channeltemp * addrate
      model:add(bottleneck(round(Channeltemp), 1, 1))
   end
   model:add(nn.Copy(nil, nil, true))
   model:add(SBatchNorm(iChannels))
   model:add(ReLU(true))
   model:add(Avg(7, 7, 1, 1))
   model:add(nn.View(iChannels):setNumInputDims(3))
   model:add(nn.Linear(iChannels, 1000))

   elseif opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
   local n = (depth - 2) / 6
   iChannels = 16
   local startChannel = 16
   local Channeltemp = 16
   addChannel = alpha^(1/(3*n))
   print(' | PyramidNet-' .. depth .. ' CIFAR-10')

   model:add(Convolution(3,16,3,3,1,1,1,1))
   model:add(SBatchNorm(iChannels))

   Channeltemp = startChannel
   startChannel = startChannel * addChannel
   model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   for i=2,n do 
      Channeltemp = startChannel
      startChannel = startChannel * addChannel
      model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   end

   Channeltemp = startChannel
   startChannel = startChannel * addChannel
   model:add(layer(basicblock, round(startChannel), 1, 2, 1))
   for i=2,n do 
      Channeltemp = startChannel
      startChannel = startChannel * addChannel
      model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   end
   Channeltemp = startChannel
   startChannel = startChannel * addChannel
   model:add(layer(basicblock, round(startChannel), 1, 2, 1))
   for i=2,n do 
      Channeltemp = startChannel
      startChannel = startChannel * addChannel
      model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   end
   model:add(nn.Copy(nil, nil, true))
   model:add(SBatchNorm(iChannels))
   model:add(ReLU(true))
   model:add(Avg(8, 8, 1, 1))
   model:add(nn.View(iChannels):setNumInputDims(3))
   if opt.dataset == 'cifar10' then
      model:add(nn.Linear(iChannels, 10))
   elseif opt.dataset == 'cifar100' then
      model:add(nn.Linear(iChannels, 100))
   end
   else
      error('invalid dataset: ' .. opt.dataset)
   end
		
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel

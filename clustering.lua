require 'torch'    -- https://github.com/torch/torch7
require 'image'    -- https://github.com/torch/image


torch.manualSeed(123)
math.randomseed(123)
math.randomseed(os.clock() * 1048576.0) ; math.randomseed(os.clock() * 1048576.0 * math.random()) ; math.randomseed(os.clock() * 1048576.0 * math.random())

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Test clustering.')
cmd:text()
cmd:option('-d', 'data.csv', 'Data file name.')
cmd:option('-n', 50000, 'Number of points.')
cmd:option('-r', 250, 'Values range.')
cmd:option('-i', '/media/sf_VMo/clusters.png', 'Output image name.')
cmd:option('-s', 2, 'Image scale.')
cmd:text()
local params = cmd:parse(arg)


local cluster_colors = torch.FloatTensor({ -- RGB
                                  {255,   0,   0}, -- каждый
                                  {255, 128,   0}, -- охотник
                                  {255, 255,   0}, -- желает
                                  {  0, 255,   0}, -- знать
                                  {  0, 255, 255}, -- где
                                  {  0,   0, 255}, -- сидит
                                  {255,   0, 255}  -- фазан
                                }) / 255
local img = torch.FloatTensor(3, params.r * params.s, params.r * params.s):fill(1)
local xy_points = torch.ByteTensor(params.r, params.r):fill(0)
local xy_cluster = torch.LongTensor(params.r, params.r):fill(0)
local cluster_centers = torch.LongTensor(cluster_colors:size(1), 2):fill(0)
local csv_data = torch.FloatTensor(params.n, 2)
local frame = 0


local function show_points()
  local s = params.s
  for x = 1, xy_points:size(2), 1 do
    for y = 1, xy_points:size(1), 1 do
      if xy_points[y][x] ~= 0 then
        local c = xy_cluster[y][x]
        local xo = (x - 1) * s
        local yo = (y - 1) * s
        if c == 0 then
          img[{{}, {yo + 1, yo + s}, {xo + 1, xo + s}}] = 0
        else
          img[{{}, {yo + 1, yo + s}, {xo + 1, xo + s}}] = cluster_colors[c]:view(img:size(1), 1, 1):expand(img:size(1), s, s)
        end
      end
    end
  end
end

local function show_centers()
  local s = params.s
  for i = 1, cluster_centers:size(1), 1 do
    local x = cluster_centers[i][1]
    if x ~= 0 then
      local y = cluster_centers[i][2]
      local xo = (x - 1) * s
      local yo = (y - 1) * s
--[[
      if xy_points[y][x] ~= 0 then
        local c = xy_cluster[y][x]
        if c == i then -- black, point color
          img[{{}, {yo + 1, yo + s}, {xo + 1, xo + s}}] = 0
        else           -- cluster color, point color
          img[{{}, {yo + 1, yo + s}, {xo + 1, xo + s}}] = cluster_colors[i]:view(img:size(1), 1, 1):expand(img:size(1), s, s)
        end
        if c == 0 then -- unassigned point
          img[{{}, {yo + 2, yo + s - 1}, {xo + 2, xo + s - 1}}] = 0
        else           -- point in cluster
          img[{{}, {yo + 2, yo + s - 1}, {xo + 2, xo + s - 1}}] = cluster_colors[c]:view(img:size(1), 1, 1):expand(img:size(1), s - 2, s - 2)
        end
      else -- empty
        img[{{}, {yo + 1, yo + s}, {xo + 1, xo + s}}] = cluster_colors[i]:view(img:size(1), 1, 1):expand(img:size(1), s, s)
        img[{{}, {yo + 2, yo + s - 1}, {xo + 2, xo + s - 1}}] = 1
      end
--]]
img[{{}, {yo + 1, yo + s}, {xo + 1, xo + s}}] = 0
    end
  end
end

local function save_image(i, u)
  local up ; if u == nil then up = true else up = u end
  local filename = params.i
  local ext = paths.extname(filename)
  local filename = string.format('%s/%s_%03d.%s', paths.dirname(filename), paths.basename(filename, ext), frame, ext)
  print("Saving ", filename)
  image.save(filename, i)
  if up then frame = frame + 1 end
end

local function generate_data(t) -- 0 = random
  local t = t or 0
  local w = math.min(xy_points:size(1), xy_points:size(2))
  local p = torch.ByteTensor(w, w):fill(0)
  if t == 0 then -- random fill
    for i = 1, csv_data:size(1), 1 do
      local x, y
      repeat
        x, y = math.random(w), math.random(w)
      until p[y][x] == 0
      p[y][x] = 1
      csv_data[i][1] = x
      csv_data[i][2] = y
    end
  end -- type
end


local function main(params)

  generate_data()

  --save_image(img)

  -- placing points
  for i = 1, csv_data:size(1), 1 do
    xy_points[csv_data[i][2]][csv_data[i][1]] = 1
    --show_points() ; save_image(img)
  end
  --show_points() ; save_image(img)

  -- initial centers
  local w = math.min(xy_points:size(1), xy_points:size(2))
  --cluster_centers[1][1] = csv_data[1][1]
  --cluster_centers[1][2] = csv_data[1][2]
  for i = 1, cluster_centers:size(1), 1 do
    local x, y
    local m = false
    repeat
      x, y = math.random(w), math.random(w)
      for j = 1, i - 1, 1 do
        if (cluster_centers[j][1] == x) and (cluster_centers[j][2] == y) then
          m = true
          break
        end
      end
    until not m
    cluster_centers[i][1] = x
    cluster_centers[i][2] = y
    --show_centers() ; save_image(img)
  end
  --show_centers() ; save_image(img)
  --print(cluster_centers)

  -- filling clusters
repeat
  xy_cluster:fill(0)
  for i = 1, csv_data:size(1), 1 do
    local px, py = csv_data[i][1], csv_data[i][2]
    if xy_cluster[py][px] == 0 then
      local ccd = torch.FloatTensor(cluster_centers:size(1))
      local cpd = torch.FloatTensor(cluster_centers:size(1))
      for j = 1, cluster_centers:size(1), 1 do
        local cx, cy = cluster_centers[j][1], cluster_centers[j][2]
        ccd[j] = ( (px - cx) ^ 2 + (py - cy) ^ 2 ) ^ 0.5 -- Euclidian
      end
      local m = ccd:min()
      for j = 1, ccd:size(1), 1 do
        if ccd[j] == m then
          xy_cluster[py][px] = j
          --print(ccd) ; print("px,py: ", px,py, "cx,cy: ", cluster_centers[j][1],cluster_centers[j][2], "min: ", m, "; cluster: ", j)
          break
        end
      end
      --xy_cluster[py][px] = math.random(cluster_colors:size(1))
    end
    --show_points() ; show_centers() ; save_image(img)
  end
  img:fill(1); show_points() ; show_centers() ; save_image(img)

  local moved = false
  for i = 1, cluster_centers:size(1), 1 do
    local cx, cy, cc = 0, 0, 0
    for j = 1, csv_data:size(1), 1 do
      local px, py = csv_data[j][1], csv_data[j][2]
      if xy_cluster[py][px] == i then
        cx = cx + px
        cy = cy + py
        cc = cc + 1
      end
    end
    if cc ~= 0 then
      local x, y = cx / cc, cy / cc
      if (math.abs(cluster_centers[i][1] - x) > 0.5) or (math.abs(cluster_centers[i][2] - y) > 0.5) then
        cluster_centers[i][1] = x
        cluster_centers[i][2] = y
        moved = true
        --print("<->")
      end
    end
  end
  img:fill(1); show_points() ; show_centers() ; save_image(img)
until not moved

end


main(params)

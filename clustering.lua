
require 'torch'    -- https://github.com/torch/torch7
require 'image'    -- https://github.com/torch/image


-- -6 - random
-- -5 - red
-- -4 - none
-- -3 - Chebyshev
-- -2 - Euclid
-- -1 - Manhattan
-- 0+ - Minkowski
local metric = -2


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Test clustering.')
cmd:text()
--cmd:option('-d', 'data.csv', 'Data file name.')
cmd:option('-n', 50000, 'Number of points.')
cmd:option('-v', 250, 'Values range.')
cmd:option('-i', '/media/sf_VMo/clusters.png', 'Output image name.')
cmd:option('-s', 2, 'Image scale.')
cmd:option('-r', '', 'Random seed.')
cmd:option('-t', 7, 'Stable frames, 2N-2.')
cmd:option('-m', metric, [[
Metric: -6 - random clusters
             -5 - 1st (red) cluster
             -4 - unassigned clusters
             -3 - Chebyshev
             -2 - Euclid
             -1 - Manhattan
           >= 0 - Minkowski
    ]])
cmd:text()
local params = cmd:parse(arg)
metric = params.m

if params.r == '' then
  math.randomseed(os.clock() * 1048576.0) ; math.randomseed(os.clock() * 1048576.0 * math.random()) ; math.randomseed(os.clock() * 1048576.0 * math.random())
else
  math.randomseed(params.r) ; torch.manualSeed(params.r)
end

local cluster_colors = torch.FloatTensor({ -- RGB
                                  {255,   0,   0}, -- каждый
                                  {255, 128,   0}, -- охотник
                                  {255, 255,   0}, -- желает
                                  {  0, 255,   0}, -- знать
                                  {  0, 255, 255}, -- где
                                  {  0,   0, 255}, -- сидит
                                  {255,   0, 255}  -- фазан
                                }) / 255
local img = torch.FloatTensor(3, params.v * params.s, params.v * params.s):fill(1)
local xy_points = torch.ByteTensor(params.v, params.v):fill(0)
local xy_cluster = torch.LongTensor(params.v, params.v):fill(0)
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
  local w = img:size(2)
  for i = 1, cluster_centers:size(1), 1 do
    local x = cluster_centers[i][1]
    if x ~= 0 then
      local y = cluster_centers[i][2]
      local xl = (x - 1) * s - 1 -- -2
      local xr = xl + s + 3      -- +2
      local yl = (y - 1) * s - 1 -- -2
      local yr = yl + s + 3      -- +2
      xl, xr, yl, yr = math.max(1, xl), math.min(w, xr), math.max(1, yl), math.min(w, yr) -- limit to image
      --print("yl,yr,xl,xr: ", yl, yr, xl, xr, "+-1: ", yl + 1, yr - 1, xl + 1, xr - 1, "cw,ch: ", cw, ch)
      img[{{}, {yl, yr}, {xl, xr}}] = 0
      img[{{}, {yl + 1, yr - 1}, {xl + 1, xr - 1}}] = cluster_colors[i]:view(img:size(1), 1, 1):expand(img:size(1), yr - yl - 1, xr - xl - 1)
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
  --cluster_centers[1][1], cluster_centers[1][2] = csv_data[1][1], csv_data[1][2]
  --cluster_centers[1][1], cluster_centers[1][2] = w, w
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

  -- correcting clusters
  local stable = 0
  repeat
    -- assigning points to clusters
    xy_cluster:fill(0)
    for i = 1, csv_data:size(1), 1 do
      local px, py = csv_data[i][1], csv_data[i][2]
      if metric == -6 then         -- случайный
        xy_cluster[py][px] = math.random(cluster_colors:size(1))
      elseif metric == -5 then     -- первый
        xy_cluster[py][px] = 1
      elseif metric == -4 then     -- не назначать
      elseif metric >= -3 then
        local ccd = torch.FloatTensor(cluster_centers:size(1)):fill(0)
        --local cpd = torch.FloatTensor(cluster_centers:size(1)):fill(0)
        for j = 1, cluster_centers:size(1), 1 do
          local cx, cy = cluster_centers[j][1], cluster_centers[j][2]
          if metric == -3 then     -- Чебышёв
            ccd[j] = math.max( math.abs(px - cx), math.abs(py - cy) )
          elseif metric == -2 then -- Евклид
            ccd[j] = math.sqrt( (px - cx) ^ 2 + (py - cy) ^ 2 )
          elseif metric == -1 then -- Манхэттен
            ccd[j] = ( math.abs(px - cx) + math.abs(py - cy) )
          elseif metric >= 0 then  -- Минковский
            -- inf = Chebyshev
            -- 2 = Euclidian
            -- 1 = Manhattan
            ccd[j] = ( math.abs(px - cx) ^ metric + math.abs(py - cy) ^ metric ) ^ (1 / metric)
          end
        end
        local m = ccd:min()
        for j = 1, ccd:size(1), 1 do
          if ccd[j] == m then
            xy_cluster[py][px] = j
            --print(ccd) ; print("px,py: ", px,py, "cx,cy: ", cluster_centers[j][1],cluster_centers[j][2], "min: ", m, "; cluster: ", j)
            break
          end
        end
      end
      --show_points() ; show_centers() ; save_image(img)
    end
    img:fill(1); show_points() ; show_centers() ; save_image(img)

    -- centers of cluster, averaging coordinates of cluster points
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
          stable = 0
        end
      end
    end
    img:fill(1); show_points() ; show_centers() ; save_image(img)
    stable = stable + 1

  until stable >= params.t

end


main(params)

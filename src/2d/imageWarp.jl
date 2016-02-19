using Images

function imwarp(
    pts::Array{Float64,2},
    img::Image,
    nCols::Int64,
    nRows::Int64,
  )

  newImg::Image = copyproperties(img, Array(typeof(img[1,1]), nCols, nRows))
  x::Float64 = 0.0
  y::Float64 = 0.0
  x0::Int64 = 0
  y0::Int64 = 0
  x1::Int64 = 0
  y1::Int64 = 0
  xx::Float64 = 0.0
  yy::Float64 = 0.0
  for i::Int64=1:nCols
    for j::Int64=1:nRows
      x = pts[i+(j-1)*nRows,1]
      y = pts[i+(j-1)*nRows,2]

      x0 = floor(Int64, x)
      y0 = floor(Int64, y)
      x1 = ceil(Int64, x)
      y1 = ceil(Int64, y)

      if (x0 < 0 || y0 < 0 || x1 >= nCols || y1 >= nRows)
        newImg.data[i,j] = 0*img[1,1]
        continue
      end

      xx = x-x0
      yy = y-y0
      newImg.data[i, j] =
        img.data[x0+1, y0+1]*(1-xx)*(1-yy)+
        img.data[x0+1, y1+1]*xx*(1-yy)+
        img.data[x1+1, y0+1]*(1-xx)*yy+
        img.data[x1+1, y1+1]*xx*yy
    end
  end
  newImg
end
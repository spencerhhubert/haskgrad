module Main where

import qualified MyLib (someFunc)
import Linear
  
vec1 = [11..15]
vec3 = [101..105]

mat4 = [vec1, vec3]

mat5 = generate_matrix 100 42 6.9
mat7 = multiply mat4 mat5

main :: IO ()
main = show_matrix mat7

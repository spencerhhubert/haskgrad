module Main where

import Haskgrad
import Linear
  
vec1 = [11..15]
vec3 = [101..105]

mat4 = [vec1, vec3]

mat5 = generate_matrix 100 42 6.9
mat7 = multiply mat4 mat5

x = [1..4]
w1 = generate_matrix 2 4
z1 = w1 `dot` x + b 
a1 = map sigmoid input_layer




main :: IO ()
main = print $ show hidden_layer1

--let's learn a simple function
float_vector = [0.00,0.01..10.00]

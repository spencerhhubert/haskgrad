module Main where

import Haskgrad
import Vector
import Matrix
  
x = [1..4]
b1 = [1..2]
w1 = generateFloatMatrix 2 4
z1 = w1 `dot` x + b1
a1 = map sigmoid input_layer




main :: IO ()
main = print $ show "ok"

--let's learn a simple function
float_vector = [0.00,0.01..10.00]

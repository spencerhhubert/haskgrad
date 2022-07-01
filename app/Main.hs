module Main where

import Haskgrad
import Vector
import Matrix
import MyRandom

input_size = 4
hidden_size = 2
output_size = 1

x = generateFloatMatrix input_size 1
b1 = generateFloatMatrix hidden_size 1
w1 = generateFloatMatrix hidden_size input_size
z2 = (w1 `multiply` x) `addMatrix` b1
a2 = mapMatrix sigmoid z2
b2 = generateFloatMatrix hidden_size 1
w2 = generateFloatMatrix hidden_size hidden_size
z3 = (w2 `multiply` a2) `addMatrix` b2
a3 = mapMatrix sigmoid z3
w3 = generateFloatMatrix output_size hidden_size
s = w3 `multiply` a3

some_shapes = [(Shape 4 1), (Shape 2 1), (Shape 2 1), (Shape 2 1)]
new_network = initNet some_shapes

input :: Matrix Float
input = [[1.0..4.0]]

propped = propForward input new_network sigmoid

main :: IO()
main = do
        print $ show propped
        print s

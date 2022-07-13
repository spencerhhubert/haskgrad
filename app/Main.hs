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
y = [16.0]
c = mse (head s) y

dcdw = mse' (head s) y

input :: Matrix Float
input = [[1.0..4.0]]

expected = [10.0..14.0]

--propped = propForward input new_network sigmoid
--cost = mse (expected) (head propped)

--eo = (mse' expected (head propped)) `multiply` (relu' zo)
--eh = ([eo] `multiply` (tail new_network)) * reluPrime 

main :: IO()
main = print "ok"
--    print $ show propped
--    print cost
--    print dcdw


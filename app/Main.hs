module Main where

import Haskgrad
import Vector
import Matrix
import MyRandom


--
--messing around
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
--end
--

--more experimenting
input = [[1.0..4.0]]
expected = [[10.0..14.0]]

actFunc = ActFunc (relu, relu')
costFunc = CostFunc (mse, mse')
slices = [Slice (4,1) actFunc, Slice (10,1) actFunc, Slice (10,1) actFunc, Slice (8,1) actFunc, Slice (6,1) actFunc, Slice (4,1) actFunc]
arch = Arch slices costFunc
network = initNet arch

propped = propForward input network


main :: IO()
main = print 10

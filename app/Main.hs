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
expected = [[1000.0..1003.0]]

actFunc = ActFunc (relu, relu')
costFunc = CostFunc (mse, mse')
slices = [Slice (4,1) actFunc, Slice (10,1) actFunc, Slice (10,1) actFunc, Slice (8,1) actFunc, Slice (6,1) actFunc, Slice (4,1) actFunc]
arch = Arch slices costFunc
net = initNet arch

propped = propForward input net
output1 = grabOutput propped
grads = gradients propped input expected
descended = descend propped 1.0 grads 

grabWeights :: NN -> Int -> Matrix Float
grabWeights (NN layers _) li = grab $ layers !! li where grab (Layer w _ _ _ _) = w 

listthem (NN layers _) = map (\(Layer w b y z c) -> map (\x -> show $ dimensions x) [w,b,y,z]) layers

trained = train net input expected 1 1

main :: IO()
main = do
    print $ grabOutput trained
    print $ grabOutput (propForward input trained)
    --print $ loss trained expected $ grabOutput trained
    print $ passThrough input net
    showMatrix $ grabWeights net 2
    print $ passThrough input grads
    showMatrix $ grabWeights grads 2
    print $ passThrough input descended
    showMatrix $ grabWeights descended 2
    --print $ passThrough input trained
    

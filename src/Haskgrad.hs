module Haskgrad where

import Vector
import Matrix
import MyRandom

data Shape = Shape Int Int
	deriving (Show, Eq)
type Architecture = [Shape]
data Layer = Layer (Matrix Float) (Matrix Float)
	deriving (Show, Eq)
data NN = NN [Layer]
	deriving (Show, Eq)

rfm = randomFloatMatrix

initNet :: Architecture -> NN
initNet arch = NN $ appendNet arch where
	appendNet :: [Shape] -> [Layer]
	appendNet arch
		| length arch == 1 || length arch == 1 = [Layer (rfm 0 0) (rfm 0 0)]
		| length arch == 2 = [Layer (weights (head arch) (jump arch)) (bias $ jump arch)]
		| otherwise = Layer (weights (head arch) (jump arch)) (bias (jump arch)) : appendNet (tail arch)
	weights x y = rfm (height y) (height x)
	bias x = rfm (height x) (depth x)

type ActivationFunc = (Float -> Float)

propForward :: Matrix Float -> NN -> ActivationFunc -> Matrix Float
propForward input (NN layers) act_func = foldl step input layers where
	step :: Matrix Float -> Layer -> Matrix Float
	step x l = mapMatrix act_func $ ((weights l) `multiply` x) `addMatrix` (bias l)
	weights (Layer w b) = w
	bias (Layer w b) = b

jump :: [a] -> a
jump x = head $ tail x

justMid :: [a] -> [a]
justMid x = tail $ init x

height :: Shape -> Int
height (Shape x y) = x

depth :: Shape -> Int
depth (Shape x y) = y

merge :: [a] -> [a] -> [a]
merge xs     []     = xs
merge []     ys     = ys
merge (x:xs) (y:ys) = x : y : merge xs ys

sigmoid :: Float -> Float
sigmoid x = 1 / (1 + (exp (-x)))

sigmoidPrime :: Float -> Float
sigmoidPrime x = (exp x) / (((exp x) + 1)^2)

gradients :: NN -> Layer -> NN
gradients network finalLayer = muhNetwork

descend :: NN -> NN -> Float -> NN
descend network gradients learningRate = muhNetwork

--junk
l1 = Layer (rfm 4 4) (rfm 4 1)
muhNetwork = NN [l1]

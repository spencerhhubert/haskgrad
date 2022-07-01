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

--layer has
--weight matrix and bias vector (1d matrix)

--A network is a list of layers
--a layer is a Tensor of some dimension
--the dimension requires as many arguments as there are dimensions
--I'm pretty sure there has to be some logic to the shapes of the layers
--like you can't have an 9x12 layer follow a 2x4 one.

--these should be changed to natural numbers, instead of Int. Or jk, why even stop
--at three dimensions? if it's written right, why doesn't the math transfer to
--n dimensions
--this will only work for matrices right now, and we're treating vectors as nx1 matrices
--maybe this dumb logic applies to any dimension. A vector with n units is an
--nx1x1 tensor hehe

data Shape = Shape Int Int
	deriving (Show, Eq)
data Layer = Layer (Matrix Float) (Matrix Float)
	deriving (Show, Eq)
data NN = NN [Layer]
	deriving (Show, Eq)

rfm = randomFloatMatrix

--this only does the weight matrices but I think the hard part is done

type Architecture = [Shape]

makeNet :: Architecture -> NN
makeNet arch = NN $ appendNet arch

appendNet :: [Shape] -> [Layer]
appendNet arch
	| length arch == 1 || length arch == 1 = [(Layer (rfm 0 0) (rfm 0 0))]
	| length arch == 2 = [Layer (weights (head arch) (jump arch)) (bias (jump arch))]
	| otherwise = (Layer (weights (head arch) (jump arch)) (bias (jump arch))) : (appendNet (tail arch))

weights x y = rfm (height y) (height x)
bias x = rfm (height x) (depth x)

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



--is there a way to define it from both ends at the same time such that if you made a network with only in and out, it would make a single weight matrix of that size?
--
	

--the weight matrix for a given layer is m x n where m=coming layer n=this layer

muhShapes = [(Shape 4 1), (Shape 2 1), (Shape 2 1), (Shape 1 1)]

l1 = Layer (rfm 4 4) (rfm 4 1)
muhNetwork = NN [l1]

forwardProp :: Matrix Float -> NN -> Matrix Float
forwardProp x y = generateFloatMatrix 10 10

gradients :: NN -> Layer -> NN
gradients network finalLayer = muhNetwork

descend :: NN -> NN -> Float -> NN
descend network gradients learningRate = muhNetwork




lair = Layer (rfm 4 4) (rfm 4 1)

--what we essentially have is a list of tuples of the same size
--
--first generate list of weight matrices
--
--




some_shapes = [(Shape 4 1), (Shape 2 1), (Shape 2 1), (Shape 1 1)]

main :: IO()
main = print $ makeNet some_shapes

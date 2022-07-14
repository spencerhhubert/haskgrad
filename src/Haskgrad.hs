module Haskgrad where

import Vector
import Matrix
import MyRandom

data Slice = Slice (Int, Int) ActFunc
data Arch = Arch [Slice] CostFunc
--                 weight         bias           output without activation
data Layer = Layer (Matrix Float) (Matrix Float) (Matrix Float) ActFunc
data NN = NN [Layer] CostFunc
--                     function           derivative
data ActFunc = ActFunc ((Float -> Float), (Float -> Float))
data CostFunc = CostFunc ((Vector Float -> Vector Float -> Float), (Vector Float -> Vector Float -> Float))

rfm = randomFloatMatrix
emptyMat = rfm 0 0
nothingFunc = ActFunc ((\x -> x), (\x -> x))

initNet :: Arch -> NN
initNet (Arch slices costFunc) = NN (appendNet slices) costFunc where
        appendNet :: [Slice] -> [Layer]
        appendNet slices
                | length slices == 0 || length slices == 1 = [Layer emptyMat emptyMat emptyMat nothingFunc] --nothing layer
                | length slices == 2 = [constructLayer]
                | otherwise = constructLayer : appendNet (tail slices)

	constructLayer = Layer (weights (head slices) (jump slices)) (bias $ jump slices) (zed $ jump slices) (getActFunc $ jump slices)
        weights x y = rfm (height y) (height x)
        bias x = rfm (height x) (depth x)
	zed x = rfm (height x) (depth x)
        getActFunc (Slice shape actFunc) = actFunc

step :: Layer -> Layer -> Layer
step (Layer w0 b0 z0 a0) (Layer w1 b1 z1 (ActFunc (act, act'))) = Layer w1 b1 (mapMatrix act $ (w1 `multiply` z0) `addMatrix` b1) $ ActFunc (act, act')

propForward :: Matrix Float -> NN -> NN
--construct a fake input "layer" from the input matrix. this is not pleasant
propForward input (NN layers costFunc) = NN (scanl step (Layer emptyMat emptyMat input nothingFunc) layers) costFunc where

apply :: Matrix Float -> NN -> Matrix Float
apply input (NN layers costFunc) = grabZed $ step (Layer emptyMat emptyMat input nothingFunc) $ last layers where
    grabZed (Layer _ _ z _) = z

--actual primary function for sending data through the network without preserving information for gradients
passThrough :: Matrix Float -> NN -> Matrix Float
passThrough x nn = apply x (propForward x nn)

--layer error is the following error times the following weight matrix times the derivative of the activation function of the inputs to that layer (the value after the matrix multiply with the previous weights)
--so do we need to store the values from before the activation function?

--gradients :: NN -> Matrix Float -> Matrix Float -> NN
--gradients (NN layers (CostFunc (cost, cost'))) actual desired = scanr error (cost' actual desired) layers where
--        error (Layer w b (ActFunc (act, act'))) = act' z
        --think need store z (before activation) values in layer :(

--foldr :: (a -> b -> b) -> b -> [a] -> b

descend :: NN -> NN -> Float -> NN
descend network gradients learningRate = muhNetwork

mse :: Vector Float -> Vector Float -> Float
mse actual desired = (/) (sum $ map (\x -> x^2) (actual `sub` desired) :: Float) l where
    l = fromIntegral $ length actual :: Float

mse' :: Vector Float -> Vector Float -> Float
mse' actual desired = (*) (2/l) (sum (actual `sub` desired) :: Float) where
    l = fromIntegral $ length actual :: Float

muhNetwork = NN [Layer (rfm 1 1) (rfm 1 1) (rfm 1 1) $ ActFunc (relu, relu')] $ CostFunc (mse, mse')

jump :: [a] -> a
jump x = head $ tail x

justMid :: [a] -> [a]
justMid x = tail $ init x

height :: Slice -> Int
height (Slice dims func) = fst dims

depth :: Slice -> Int
depth (Slice dims func) = snd dims

merge :: [a] -> [a] -> [a]
merge xs     []     = xs
merge []     ys     = ys
merge (x:xs) (y:ys) = x : y : merge xs ys

sigmoid :: Float -> Float
sigmoid x = 1 / (1 + (exp (-x)))

sigmoid' :: Float -> Float
sigmoid' x = (exp x) / (((exp x) + 1)^2)

relu :: Float -> Float
relu x = max 0.0 x

relu' :: Float -> Float
relu' x
    | x < 0 = 0.0
    | otherwise = 1.0

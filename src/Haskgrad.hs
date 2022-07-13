module Haskgrad where

import Vector
import Matrix
import MyRandom

data Shape = Shape Int Int
        deriving (Show, Eq)
type Architecture = [Shape]
--                     function           derivative
data ActFunc = ActFunc ((Float -> Float), (Float -> Float))
data CostFunc = CostFunc ((Vector Float -> Vector Float -> Float), (Vector Float -> Vector Float -> Float))

data Slice = Slice (Int, Int) ActFunc
data Arch = Arch [Slice] CostFunc
--                 weight         bias
data Layer = Layer (Matrix Float) (Matrix Float) ActFunc
data NN = NN [Layer] CostFunc

rfm = randomFloatMatrix

initNet :: Arch -> NN
initNet (Arch slices costFunc) = NN (appendNet slices) costFunc where
        appendNet :: [Slice] -> [Layer]
        appendNet slices
                | length slices == 0 || length slices == 1 = [Layer (rfm 0 0) (rfm 0 0) nothingFunc] --nothing layer
                | length slices == 2 = [Layer (weights (head slices) (jump slices)) (bias $ jump slices) (actFunc $ jump slices)]
                | otherwise = Layer (weights (head slices) (jump slices)) (bias $ jump slices) (actFunc $ jump slices) : appendNet (tail slices)
        weights x y = rfm (height y) (height x)
        bias x = rfm (height x) (depth x)
        actFunc (Slice shape funcs) = funcs
        nothingFunc = ActFunc ((\x -> x), (\x -> x))

propForward :: Matrix Float -> NN -> Matrix Float
propForward input (NN layers costFunc) = foldl step input layers where
        step :: Matrix Float -> Layer -> Matrix Float
        step x (Layer w b (ActFunc (f, f'))) = (mapMatrix f (w `multiply` x)) `addMatrix` b

--layer error is the following error times the following weight matrix times the derivative of the activation function of the inputs to that layer (the value after the matrix multiply with the previous weights)
--so do we need to store the values from before the activation function?

--gradients :: NN -> Matrix Float -> Matrix Float -> NN
--gradients network actual desired = where
--        foldr func (cost' actual desired)

--foldr :: (a -> b -> b) -> b -> [a] -> b

descend :: NN -> NN -> Float -> NN
descend network gradients learningRate = muhNetwork

mse :: Vector Float -> Vector Float -> Float
mse actual desired = (/) (sum $ map (\x -> x^2) (actual `sub` desired) :: Float) l where
    l = fromIntegral $ length actual :: Float

mse' :: Vector Float -> Vector Float -> Float
mse' actual desired = (*) (2/l) (sum (actual `sub` desired) :: Float) where
    l = fromIntegral $ length actual :: Float

muhNetwork = NN [Layer (rfm 1 1) (rfm 1 1) $ ActFunc (relu, relu')] $ CostFunc (mse, mse')

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

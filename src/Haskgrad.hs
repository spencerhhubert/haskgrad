module Haskgrad where

import Vector
import Matrix
import MyRandom

data Slice = Slice (Int, Int) ActFunc
data Arch = Arch [Slice] CostFunc
--                 weight         bias           output         output without activation
data Layer = Layer (Matrix Float) (Matrix Float) (Matrix Float) (Matrix Float) ActFunc
data NN = NN [Layer] CostFunc
--                     function           derivative
data ActFunc = ActFunc ((Float -> Float), (Float -> Float))
data CostFunc = CostFunc ((Vector Float -> Vector Float -> Float), (Vector Float -> Vector Float -> Vector Float))

showLayer (Layer w b y z c) = map (\x -> show $ dimensions x) [w,b,y,z]

rfm = randomFloatMatrix
emptyMat = rfm 0 0
nothingFunc = ActFunc ((\x -> x), (\x -> x))

initNet :: Arch -> NN
initNet (Arch slices costFunc) = NN (appendNet slices) costFunc

appendNet :: [Slice] -> [Layer]
appendNet slices
    | length slices == 0 || length slices == 1 = [Layer emptyMat emptyMat emptyMat emptyMat nothingFunc] --nothing layer
    | length slices == 2 = [constructLayer]
    | otherwise = constructLayer : appendNet (tail slices)
    where
    constructLayer = Layer (weights (head slices) (jump slices)) (bias $ jump slices) (out $ jump slices) (zed $ jump slices) (getActFunc $ jump slices)
    weights x y = rfm (height y) (height x)
    bias x = rfm (height x) (depth x)
    out x = rfm (height x) (depth x)
    zed x = rfm (height x) (depth x)
    getActFunc (Slice shape actFunc) = actFunc

step :: Layer -> Layer -> Layer
step (Layer w0 b0 y0 z0 a0) (Layer w1 b1 y1 z1 (ActFunc (act, act'))) = Layer w1 b1 (compute) (mapMatrix act compute) $ ActFunc (act, act') where
   compute = (w1 `multiply` z0) `addMatrix` b1

propForward :: Matrix Float -> NN -> NN
--construct a fake input "layer" from the input matrix. this is not pleasant
propForward input (NN layers costFunc) = NN (scanl1 step layersChopped) costFunc where
    layersChopped = (step emptyLayer $ head layers) : (tail layers)
    emptyLayer = Layer emptyMat emptyMat emptyMat input nothingFunc

grabOutput :: NN -> Matrix Float
grabOutput (NN layers _) = grabZed $ last layers where
    grabZed (Layer _ _ _ z _) = z

--actual primary function for sending data through the network without preserving information for gradients
passThrough :: Matrix Float -> NN -> Matrix Float
passThrough x nn = grabOutput (propForward x nn)

gradients :: NN -> Matrix Float -> Matrix Float -> Matrix Float -> NN
gradients (NN layers (CostFunc (cost, cost'))) input actual desired = NN (reverse (tuneWeights (reverse layers) initialError)) $ CostFunc (cost, cost') where
    initialError :: Matrix Float
    initialError = [cost' (head actual) (head desired)] `hadamard` (mapMatrix (getAct' $ last layers) (getZed $ last layers))
    tuneWeights :: [Layer] -> Matrix Float -> [Layer]
    tuneWeights layers' error
        | length layers' == 1 = [constructInputLayer $ head layers']
        | otherwise = constructLayer (head layers') (jump layers') : tuneWeights (tail layers') constructError 
        where
            constructLayer :: Layer -> Layer -> Layer
            constructLayer (Layer w0 b0 y0 z0 c0) (Layer w1 b1 y1 z1 c1) = Layer (error `multiply` y1) b0 y0 z0 c0
            constructError :: Matrix Float
            constructError = (transpose $ weights $ head layers') `hadamard` (mapMatrix (getAct' $ head layers') (getZed $ head layers'))
            constructInputLayer :: Layer -> Layer
            constructInputLayer (Layer w b y z c) = Layer (error `multiply` input) b y z c --this is all quite messy

weights :: Layer -> Matrix Float
weights (Layer w _ _ _ _) = w
output :: Layer -> Matrix Float --should change to vector
output (Layer _ _ y _ _) = y
getAct' :: Layer -> (Float -> Float)
getAct' (Layer _ _ _ _ (ActFunc (act, act'))) = act'
getZed (Layer _ _ _ z _) = z --these functions really suck

--todo: need to either remove this vector nonsense or make it so multiplies can take vectors. this might require changing vector and matrix
--to tensor and make multiply take a tensor type

descend :: NN -> NN -> Float -> NN
descend network gradients learningRate = zipWithNN (-) network (mapNN (\x -> x*learningRate) gradients)

mapNN :: (Float -> Float) -> NN -> NN
mapNN f (NN layers costFunc) = NN (map (\(Layer w b y z c) -> Layer (mapMatrix f w) b y z c) layers) costFunc

zipWithNN :: (Float -> Float -> Float) -> NN -> NN -> NN
zipWithNN f (NN layers0 costFunc0) (NN layers1 costFunc1) = NN thing costFunc1 where
    thing = zipWith func layers0 layers1
    func = (\(Layer w0 b0 y0 z0 c0) (Layer w1 b1 y1 z1 c1) -> Layer (zipWithMatrix f w0 w1) b0 y0 z0 c0)

mse :: Vector Float -> Vector Float -> Float
mse actual desired = (/) (sum $ map (\x -> x^2) (actual `sub` desired) :: Float) l where
    l = fromIntegral $ length actual :: Float

mse' :: Vector Float -> Vector Float -> Vector Float
mse' actual desired = scale (actual `sub` desired) (-2)

muhNetwork = NN [Layer (rfm 1 1) (rfm 1 1) (rfm 1 1) (rfm 1 1) $ ActFunc (relu, relu')] $ CostFunc (mse, mse')

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

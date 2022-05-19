import Errors
import Control.Monad.Except

--data Scalar a = Scalar a
data Vector a = Vector [a]        deriving (Read, Show)
data Matrix a = Matrix [Vector a] deriving (Read, Show)
--data Tensor a = Vector a | Matrix a | Tensor [Matrix a] deriving (Read, Show)

--class Tensor a where
--    add :: a -> a -> a

--add :: Num a, Ten b => b a -> b a -> b a

dot :: Num a => Vector a -> Vector a -> ThrowsError a
dot (Vector x) (Vector y) = if (length x /= length y)
        then throwError $ Dimensions (length x) (length y) 0 "Vectors must be the same length"
        else return $ sum (zipWith (*) x y)
dot _ _ = throwError $ Args "bad args, should be two vectors of same length and type"

times :: Num a => Matrix a -> Matrix a -> ThrowsError (Matrix a)
times (Matrix x) (Matrix y) = return $ Matrix x

--what we want is an add, multiply, transpose, inverse, and so on function defined for all tensors
--really whether it's a vector, matrix, or tensor is dependent on the dimensionality...
--maybe I should think about how I actually want to use this stuff...
--there are going to be times we want to type match on vectors, matries, and tensors

--what type signitures and what type safety do I want...

--need to find a way to make tensor datatype again...


--data Tensor a = Scalar a | Vector [Scalar a] | Matrix [[a]] | Tensor [[[a]]]
--    deriving (Read, Show)

--this could be written to take a list of arguments and add of them, but perhaps that deserves another function...
-- add :: Num a => Tensor a -> Tensor a -> Tensor a
-- add (Vector x) (Vector y) = Vector $ zipWith (+) x y
--add (Matrix x) (Matrix y) = Matrix $ map (uncurry add) (zip x y)

vec1 = Vector [1..6]
vec2 = Vector [10..15]
vec3 = Vector [21..106]

mat1 = Matrix [vec1, vec2]
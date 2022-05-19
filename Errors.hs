module Errors where

data TensorError = Dimensions Int Int Int String
    | Args String
    | Default String
    deriving (Read, Show)


type ThrowsError = Either TensorError

extractValue :: ThrowsError a -> a
extractValue (Right val) = val
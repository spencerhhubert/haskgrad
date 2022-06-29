module Haskgrad where

sigmoid :: Float -> Float
sigmoid x = 1 / (1 + (exp (-x)))

sigmoidPrime :: Float -> Float
sigmoidPrime x = (exp x) / (((exp x) + 1)^2)


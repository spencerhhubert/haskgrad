module Haskgrad where

sigmoid :: Float -> Float
sigmoid x = 1 / (1 + (exp (-x)))


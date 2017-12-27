factorial :: Integer -> Integer
factorial n = product [1..n]

circumference :: Float -> Float
circumference r = 2 * pi * r

factorial' :: (Integral a) => a -> a
factorial' 0 = 1
factorial' n = n * factorial' (n - 1)

addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

sum :: (Num a) => [a] -> a
sum []     = 0
sum (x:xs) = x + sum xs

max :: (Ord a) => a -> a -> a
max a b
    | a < b     = a
    | otherwise = b

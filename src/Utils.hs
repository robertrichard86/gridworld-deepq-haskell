module Utils
  ( chunksOf
  , movingAverage
  , formatDuration
  , progressBar
  , safeMaximum
  , safeMinimum
  ) where

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs =
  let (chunk, rest) = splitAt n xs
  in chunk : chunksOf n rest

movingAverage :: Int -> [Double] -> [Double]
movingAverage _ [] = []
movingAverage window xs
  | length xs < window = [sum xs / fromIntegral (length xs)]
  | otherwise =
      let chunks = zip [0..] xs
          avg i  = let slice = take window (drop i xs)
                   in sum slice / fromIntegral (length slice)
      in map avg [0 .. length xs - window]

formatDuration :: Double -> String
formatDuration seconds
  | seconds < 60    = showF seconds ++ "s"
  | seconds < 3600  = let mins = truncate (seconds / 60) :: Int
                          secs = seconds - fromIntegral mins * 60
                      in show mins ++ "m " ++ showF secs ++ "s"
  | otherwise       = let hours = truncate (seconds / 3600) :: Int
                          remaining = seconds - fromIntegral hours * 3600
                          mins = truncate (remaining / 60) :: Int
                      in show hours ++ "h " ++ show mins ++ "m"
  where
    showF x = let w = truncate x :: Int
                  f = abs (round ((x - fromIntegral w) * 10)) :: Int
              in show w ++ "." ++ show f

progressBar :: Int -> Int -> Int -> String
progressBar width current total =
  let ratio    = fromIntegral current / fromIntegral total :: Double
      filled   = round (ratio * fromIntegral width) :: Int
      empty'   = width - filled
      pct      = round (ratio * 100) :: Int
  in "[" ++ replicate filled '█' ++ replicate empty' '░' ++ "] " ++ show pct ++ "%"

safeMaximum :: (Ord a) => [a] -> Maybe a
safeMaximum [] = Nothing
safeMaximum xs = Just (maximum xs)

safeMinimum :: (Ord a) => [a] -> Maybe a
safeMinimum [] = Nothing
safeMinimum xs = Just (minimum xs)

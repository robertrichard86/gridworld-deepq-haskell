module ReplayBuffer
  ( emptyBuffer
  , addExperience
  , sampleBatch
  , bufferSize
  ) where

import System.Random (StdGen, randomR)
import Types

emptyBuffer :: Int -> ReplayBuffer
emptyBuffer maxSize = ReplayBuffer
  { rbBuffer  = []
  , rbMaxSize = maxSize
  }

addExperience :: Experience -> ReplayBuffer -> ReplayBuffer
addExperience exp' buf =
  let newBuffer = exp' : rbBuffer buf
      trimmed   = take (rbMaxSize buf) newBuffer
  in buf { rbBuffer = trimmed }

sampleBatch :: StdGen -> Int -> ReplayBuffer -> ([Experience], StdGen)
sampleBatch gen batchSize buf =
  let experiences = rbBuffer buf
      n           = length experiences
      actualBatch = min batchSize n
  in sampleN gen actualBatch experiences []

sampleN :: StdGen -> Int -> [Experience] -> [Experience] -> ([Experience], StdGen)
sampleN gen 0 _ acc = (acc, gen)
sampleN gen n xs acc =
  let (idx, gen') = randomR (0, length xs - 1) gen
      selected    = xs !! idx
  in sampleN gen' (n - 1) xs (selected : acc)

bufferSize :: ReplayBuffer -> Int
bufferSize = length . rbBuffer

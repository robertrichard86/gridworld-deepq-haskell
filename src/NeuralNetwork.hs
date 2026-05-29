module NeuralNetwork
  ( initNetwork
  , forward
  , backward
  , updateWeights
  , predict
  , networkLoss
  ) where

import Numeric.LinearAlgebra
  ( Matrix, Vector, (#>)
  , cmap, outer, tr', scale, sumElements
  , konst, size, fromList, toList
  )
import qualified Numeric.LinearAlgebra as LA
import System.Random (StdGen, split, randomRs)
import Types

initNetwork :: StdGen -> Int -> Int -> Int -> (Network, StdGen)
initNetwork gen inputSize hiddenSize outputSize =
  let (gen1, gen2) = split gen
      (gen3, gen4) = split gen2
      xavierHidden = sqrt (2.0 / fromIntegral inputSize)
      xavierOutput = sqrt (2.0 / fromIntegral hiddenSize)
      w1Vals = take (hiddenSize * inputSize) (randomRs (-1.0, 1.0) gen1)
      w1 = scale xavierHidden $ (hiddenSize LA.>< inputSize) w1Vals
      b1 = konst 0.0 hiddenSize
      w2Vals = take (outputSize * hiddenSize) (randomRs (-1.0, 1.0) gen3)
      w2 = scale xavierOutput $ (outputSize LA.>< hiddenSize) w2Vals
      b2 = konst 0.0 outputSize
  in (Network w1 b1 w2 b2, gen4)

relu :: Vector Double -> Vector Double
relu = cmap (max 0)

reluDerivative :: Vector Double -> Vector Double
reluDerivative = cmap (\x -> if x > 0 then 1.0 else 0.0)

forward :: Network -> Vector Double -> (Vector Double, Vector Double, Vector Double)
forward net input =
  let z1     = (netWeights1 net #> input) + netBias1 net
      hidden = relu z1
      z2     = (netWeights2 net #> hidden) + netBias2 net
  in (z2, hidden, z1)

predict :: Network -> Vector Double -> Vector Double
predict net input =
  let (output, _, _) = forward net input
  in output

backward :: Network -> Vector Double -> Vector Double -> (Matrix Double, Vector Double, Matrix Double, Vector Double)
backward net input targetQ =
  let (output, hidden, z1) = forward net input
      outputError = output - targetQ
      dW2 = outer outputError hidden
      dB2 = outputError
      hiddenError = tr' (netWeights2 net) #> outputError
      reluGrad    = reluDerivative z1
      hiddenDelta = hiddenError * reluGrad
      dW1 = outer hiddenDelta input
      dB1 = hiddenDelta
  in (dW1, dB1, dW2, dB2)

updateWeights :: Double -> Network -> (Matrix Double, Vector Double, Matrix Double, Vector Double) -> Network
updateWeights lr net (dW1, dB1, dW2, dB2) =
  let clipGrad v = cmap (max (-1.0) . min 1.0) v
      clipGradM m = cmap (max (-1.0) . min 1.0) m
  in Network
    { netWeights1 = netWeights1 net - scale lr (clipGradM dW1)
    , netBias1    = netBias1 net - scale lr (clipGrad dB1)
    , netWeights2 = netWeights2 net - scale lr (clipGradM dW2)
    , netBias2    = netBias2 net - scale lr (clipGrad dB2)
    }

networkLoss :: Vector Double -> Vector Double -> Double
networkLoss predicted target =
  let diff = predicted - target
      squared = cmap (\x -> x * x) diff
  in sumElements squared / fromIntegral (size diff)

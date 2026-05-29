module Agent
  ( selectAction
  , decayEpsilon
  , defaultAgentConfig
  ) where

import Numeric.LinearAlgebra (Vector, maxIndex)
import System.Random (StdGen, randomR)
import Types
import NeuralNetwork (predict)

selectAction :: StdGen -> Double -> Network -> Vector Double -> (Action, StdGen)
selectAction gen epsilon net state =
  let (r, gen') = randomR (0.0 :: Double, 1.0) gen
  in if r < epsilon
     then randomAction gen'
     else
       let qValues = predict net state
           bestIdx = maxIndex qValues
       in (indexToAction bestIdx, gen')

randomAction :: StdGen -> (Action, StdGen)
randomAction gen =
  let (idx, gen') = randomR (0, numActions - 1) gen
  in (indexToAction idx, gen')

decayEpsilon :: AgentConfig -> AgentConfig
decayEpsilon cfg =
  let newEps = max (acEpsilonMin cfg) (acEpsilon cfg * acEpsilonDecay cfg)
  in cfg { acEpsilon = newEps }

defaultAgentConfig :: AgentConfig
defaultAgentConfig = AgentConfig
  { acEpsilon      = 1.0
  , acEpsilonMin   = 0.01
  , acEpsilonDecay = 0.995
  , acGamma        = 0.99
  , acLearningRate = 0.001
  , acBatchSize    = 32
  }

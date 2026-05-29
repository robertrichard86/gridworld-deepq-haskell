module DQN
  ( trainDQN
  , trainStep
  , runEpisode
  , defaultDQNConfig
  ) where

import Numeric.LinearAlgebra (Vector, fromList, toList, maxElement)
import System.Random (StdGen)
import Data.List (foldl')
import Types
import Environment
import NeuralNetwork
import ReplayBuffer
import Agent

data TrainState = TrainState
  { tsNet        :: Network
  , tsBuffer     :: ReplayBuffer
  , tsAgentCfg   :: AgentConfig
  , tsGen        :: StdGen
  , tsAllStats   :: [TrainingStats]
  }

defaultDQNConfig :: DQNConfig
defaultDQNConfig = DQNConfig
  { dqnNumEpisodes = 500
  , dqnMaxSteps    = 100
  , dqnHiddenSize  = 64
  , dqnInputSize   = stateSize
  , dqnOutputSize  = numActions
  , dqnAgentConfig = defaultAgentConfig
  , dqnBufferSize  = 10000
  }

trainDQN :: DQNConfig -> StdGen -> (Network, [TrainingStats])
trainDQN cfg gen =
  let (net0, gen') = initNetwork gen
                       (dqnInputSize cfg)
                       (dqnHiddenSize cfg)
                       (dqnOutputSize cfg)
      buf0         = emptyBuffer (dqnBufferSize cfg)
      initState    = TrainState net0 buf0 (dqnAgentConfig cfg) gen' []
      finalState   = foldl' (trainEpisode cfg) initState [1 .. dqnNumEpisodes cfg]
  in (tsNet finalState, reverse (tsAllStats finalState))

trainEpisode :: DQNConfig -> TrainState -> Int -> TrainState
trainEpisode cfg ts episodeNum =
  let env0    = resetEnv mkGridWorld
      state0  = stateToVector env0
      (net', buf', gen', totalReward, steps, losses, reachedGoal) =
        runEpisode cfg (tsNet ts) (tsBuffer ts) (tsAgentCfg ts) (tsGen ts)
                   env0 state0 0.0 0 [] (dqnMaxSteps cfg)
      avgLoss = if null losses then 0.0 else sum losses / fromIntegral (length losses)
      stats   = TrainingStats
        { tsEpisode     = episodeNum
        , tsTotalReward  = totalReward
        , tsSteps        = steps
        , tsEpsilon      = acEpsilon (tsAgentCfg ts)
        , tsAverageLoss  = avgLoss
        , tsReachedGoal  = reachedGoal
        }
      newCfg  = decayEpsilon (tsAgentCfg ts)
  in TrainState net' buf' newCfg gen' (stats : tsAllStats ts)

runEpisode :: DQNConfig -> Network -> ReplayBuffer -> AgentConfig -> StdGen
           -> GridWorld -> Vector Double
           -> Double -> Int -> [Double] -> Int
           -> (Network, ReplayBuffer, StdGen, Double, Int, [Double], Bool)
runEpisode cfg net buf agentCfg gen env state totalReward steps losses maxSteps
  | steps >= maxSteps = (net, buf, gen, totalReward, steps, losses, False)
  | gwDone env        = (net, buf, gen, totalReward, steps, losses, totalReward > 0)
  | otherwise =
      let (action, gen1)   = selectAction gen (acEpsilon agentCfg) net state
          (env', reward, done) = stepEnv env action
          nextState        = stateToVector env'
          experience       = Experience state action reward nextState done
          buf'             = addExperience experience buf
          (net', gen2, loss) = if bufferSize buf' >= acBatchSize agentCfg
                               then trainStep agentCfg net buf' gen1
                               else (net, gen1, 0.0)
          newLosses        = if loss > 0.0 then loss : losses else losses
      in runEpisode cfg net' buf' agentCfg gen2 env' nextState
                    (totalReward + reward) (steps + 1) newLosses maxSteps

trainStep :: AgentConfig -> Network -> ReplayBuffer -> StdGen -> (Network, StdGen, Double)
trainStep cfg net buf gen =
  let (batch, gen') = sampleBatch gen (acBatchSize cfg) buf
      (net', totalLoss) = foldl' (updateFromExperience cfg) (net, 0.0) batch
      avgLoss = totalLoss / fromIntegral (length batch)
  in (net', gen', avgLoss)

updateFromExperience :: AgentConfig -> (Network, Double) -> Experience -> (Network, Double)
updateFromExperience cfg (net, accLoss) exp' =
  let currentQ  = predict net (expState exp')
      currentQL = toList currentQ
      nextQ     = predict net (expNextState exp')
      maxNextQ  = if expDone exp' then 0.0 else maxElement nextQ
      actionIdx = actionToIndex (expAction exp')
      targetVal = expReward exp' + acGamma cfg * maxNextQ
      targetQ   = fromList [ if i == actionIdx then targetVal else currentQL !! i
                            | i <- [0 .. numActions - 1]
                            ]
      grads     = backward net (expState exp') targetQ
      net'      = updateWeights (acLearningRate cfg) net grads
      loss      = networkLoss (predict net (expState exp')) targetQ
  in (net', accLoss + loss)

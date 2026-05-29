{-# LANGUAGE FlexibleInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}
module Types
  ( Position
  , GridSize
  , Action(..)
  , allActions
  , actionToIndex
  , indexToAction
  , numActions
  , CellType(..)
  , GridWorld(..)
  , Experience(..)
  , ReplayBuffer(..)
  , Network(..)
  , AgentConfig(..)
  , TrainingStats(..)
  , DQNConfig(..)
  , Symbols(..)
  , defaultSymbols
  ) where

import Numeric.LinearAlgebra (Matrix, Vector)
import GHC.Generics (Generic)
import Control.DeepSeq (NFData(..))

type Position = (Int, Int)
type GridSize = (Int, Int)

data Action = Up | Down | MoveLeft | MoveRight
  deriving (Show, Eq, Ord, Enum, Bounded, Generic)

instance NFData Action

allActions :: [Action]
allActions = [minBound .. maxBound]

actionToIndex :: Action -> Int
actionToIndex = fromEnum

indexToAction :: Int -> Action
indexToAction = toEnum

numActions :: Int
numActions = length allActions

data CellType = Empty | Obstacle | Goal | Start
  deriving (Show, Eq, Ord, Generic)

instance NFData CellType

data GridWorld = GridWorld
  { gwSize       :: GridSize
  , gwAgentPos   :: Position
  , gwGoalPos    :: Position
  , gwObstacles  :: [Position]
  , gwStartPos   :: Position
  , gwDone       :: Bool
  } deriving (Show, Eq, Generic)

instance NFData GridWorld

instance NFData (Vector Double) where
  rnf v = v `seq` ()

instance NFData (Matrix Double) where
  rnf m = m `seq` ()

data Experience = Experience
  { expState      :: Vector Double
  , expAction     :: Action
  , expReward     :: Double
  , expNextState  :: Vector Double
  , expDone       :: Bool
  } deriving (Show, Generic)

instance NFData Experience

data ReplayBuffer = ReplayBuffer
  { rbBuffer   :: [Experience]
  , rbMaxSize  :: Int
  } deriving (Show, Generic)

instance NFData ReplayBuffer

data Network = Network
  { netWeights1  :: Matrix Double
  , netBias1     :: Vector Double
  , netWeights2  :: Matrix Double
  , netBias2     :: Vector Double
  } deriving (Show, Generic)

instance NFData Network

data AgentConfig = AgentConfig
  { acEpsilon      :: Double
  , acEpsilonMin   :: Double
  , acEpsilonDecay :: Double
  , acGamma        :: Double
  , acLearningRate :: Double
  , acBatchSize    :: Int
  } deriving (Show, Generic)

instance NFData AgentConfig

data TrainingStats = TrainingStats
  { tsEpisode       :: Int
  , tsTotalReward   :: Double
  , tsSteps         :: Int
  , tsEpsilon       :: Double
  , tsAverageLoss   :: Double
  , tsReachedGoal   :: Bool
  } deriving (Show, Generic)

instance NFData TrainingStats

data DQNConfig = DQNConfig
  { dqnNumEpisodes  :: Int
  , dqnMaxSteps     :: Int
  , dqnHiddenSize   :: Int
  , dqnInputSize    :: Int
  , dqnOutputSize   :: Int
  , dqnAgentConfig  :: AgentConfig
  , dqnBufferSize   :: Int
  } deriving (Show, Generic)

instance NFData DQNConfig

data Symbols = Symbols
  { symAgent    :: String
  , symGoal     :: String
  , symObstacle :: String
  , symEmpty    :: String
  , symPath     :: String
  } deriving (Show, Eq, Generic)

instance NFData Symbols

defaultSymbols :: Symbols
defaultSymbols = Symbols
  { symAgent    = "\129497"
  , symGoal     = "\128142"
  , symObstacle = "\128293"
  , symEmpty    = "\11036"
  , symPath     = "\128994"
  }
